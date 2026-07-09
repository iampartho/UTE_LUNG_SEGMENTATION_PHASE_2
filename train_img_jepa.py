"""
Phase 1, step 2: train the IMAGE encoder E_img + cross-modal predictor P against
the FROZEN Step-1 mask encoder E_mask.

Objective (the cross-modal energy Phase 2 will steer):
    loss = || P(E_img(GIN(CT_img))) - E_mask(mask) ||_1

Fixed choices (all for consistency with Step 1, so the trained objective IS the
deployed energy):
    * E_mask = the Step-1 EMA target encoder, frozen, LayerNorm'd target -- same
      as phase0_energy_vs_dice.py. Loaded from MASK_CKPT below.
    * Canonical 256^3 grid, patch 32 -> 8^3 = 512 tokens (inherited from E_mask).
    * PAIRED geometric shape-aug (anisotropic scale -> elongation, elastic warp
      -> extent) applied identically to image AND mask -- the SAME family/params
      the Step-1 mask encoder was trained under, so both encoders see a symmetric
      shape distribution. (dataset_img_jepa.py + datasets_causality transforms.)
    * GIN appearance augmentation applied to the IMAGE only, on the GPU, at 80%
      probability -- mirrors causality_train.py. This is the appearance-invariance
      that must let the CT-trained image branch transfer to true UTE at test time.

Train on CT only (no UTE labels, no UTE images in training -> source-free story);
validate on UTE. E_img never sees UTE in training, so lowest UTE val L1 is an
honest cross-modal generalisation criterion for model selection (same logic as
Step 1). The artefacts kept are E_img and P (and, for self-contained Phase-2
reload, the frozen E_mask state is saved alongside).
"""
import os
import sys
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset_img_jepa import ImgJEPADataset, MaskGaussianBlur
from datasets_causality import RandomAnisoScale, RandomElastic, PadOrCrop, ToTensor
from model_img_jepa import ImgJEPA, count_parameters
from gin_with_log_capability import CausalityAugmentation3D


# ---------------------------------------------------------------------------
# Hyperparameters / paths
# ---------------------------------------------------------------------------
# CT-only train (no UTE), UTE val (model selection on cross-modal generalisation).
TRAIN_CSV = './ids/only_copd_1.25mm.csv'
VAL_CSV = './ids/only_ute_1.25mm.csv'
PATH_REPLACE = None              # optional (old, new) CSV path remap; None = verbatim

# Frozen Step-1 mask encoder (the new-aug LATEST checkpoint chosen in Phase 1 Step 1).
MASK_CKPT = './save_models/latest_mask_jepa_ct256_new_aug_optimised.pth'
MASK_WHICH = 'target_encoder'    # EMA target -> matches the Phase-0 energy
TARGET_NORM = True               # LayerNorm the E_mask target (matches Step 1)

# Canonical input (native UTE size). The token grid / patch are inherited from
# E_mask's config inside ImgJEPA; this is only used by the dataset's PadOrCrop.
INPUT_SIZE = (256, 256, 256)

# E_img (ViT) -- defaults mirror the Step-1 encoder shape; embed_dim defaults to
# the mask encoder width inside ImgJEPA so the latents are comparable.
IMG_EMBED_DIM = 384
IMG_DEPTH = 10
IMG_HEADS = 6
# Cross-modal predictor P (deterministic full-token map).
PRED_DIM = 384
PRED_DEPTH = 6
PRED_HEADS = 6
MLP_RATIO = 4.0

# --- Appearance augmentation (GIN, image-only, GPU, train-only) -------------
GIN_PROB = 0.9

# --- PAIRED geometric shape-aug (image + mask together), matching Step 1 ----
# Aniso scale runs on the native grid (before the 256^3 fit); elastic runs after
# the fit, on the padded grid. Mask softened with sigma=1.0 as the last step.
SCALE_PROB = 0.5
SCALE_RANGE = (0.80, 1.25)       # per-axis, non-isotropic -> varies elongation
ELASTIC_PROB = 0.5
ELASTIC_ALPHA = 12.0             # displacement magnitude in voxels
ELASTIC_CTRL = 8                 # control-grid res; smaller => smoother global bends
GAUSSIAN_SIGMA = 1.0             # mask softening (match E_mask training input)
CLEAN_MASKS = True
CLEAN_KEEP_FRAC = 0.10

# At 256^3 each sample is heavy: small per-step batch + gradient accumulation.
BATCH_SIZE = 2
ACCUM_STEPS = 8                  # effective batch ~ BATCH_SIZE * ACCUM_STEPS
NUM_WORKERS = 8

EPOCHS = 500
BASE_LR = 5e-4
WARMUP_EPOCHS = 15
FINAL_LR = 1e-6
WEIGHT_DECAY = 0.04

LOG_DIR = './log'
SAVE_DIR = './save_models'
TAG = 'img_jepa_ct256_new_aug'
TRAIN_LOG_CSV = os.path.join(LOG_DIR, f'train_metrics_{TAG}.csv')
TEST_LOG_CSV = os.path.join(LOG_DIR, f'test_metrics_{TAG}.csv')
BEST_WEIGHT_PATH = os.path.join(SAVE_DIR, f'best_{TAG}.pth')
LATEST_WEIGHT_PATH = os.path.join(SAVE_DIR, f'latest_{TAG}.pth')


global_best_loss = np.inf


def build_config():
    """Single source of truth for the model config (saved into checkpoints)."""
    return dict(
        mask_ckpt=MASK_CKPT,
        mask_which=MASK_WHICH,
        target_norm=TARGET_NORM,
        input_size=INPUT_SIZE,
        img_embed_dim=IMG_EMBED_DIM,
        img_depth=IMG_DEPTH,
        img_heads=IMG_HEADS,
        pred_dim=PRED_DIM,
        pred_depth=PRED_DEPTH,
        pred_heads=PRED_HEADS,
        mlp_ratio=MLP_RATIO,
    )


def lr_at(epoch):
    if epoch < WARMUP_EPOCHS:
        return BASE_LR * float(epoch + 1) / float(WARMUP_EPOCHS)
    progress = (epoch - WARMUP_EPOCHS) / max(1, (EPOCHS - WARMUP_EPOCHS))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return FINAL_LR + (BASE_LR - FINAL_LR) * cosine


def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            'config': build_config(),
            'aug': {
                'gin_prob': GIN_PROB,
                'clean_masks': CLEAN_MASKS,
                'clean_keep_frac': CLEAN_KEEP_FRAC,
                'scale_prob': SCALE_PROB,
                'scale_range': SCALE_RANGE,
                'elastic_prob': ELASTIC_PROB,
                'elastic_alpha': ELASTIC_ALPHA,
                'elastic_ctrl': ELASTIC_CTRL,
                'gaussian_sigma': GAUSSIAN_SIGMA,
            },
            # Trainable artefacts.
            'encoder': model.encoder.state_dict(),       # E_img
            'predictor': model.predictor.state_dict(),   # P
            # Frozen E_mask copied in so Phase 2 can reload the whole energy from
            # this one file (also still recoverable from MASK_CKPT['target_encoder']).
            'mask_encoder': model.mask_encoder.state_dict(),
            'mask_config': model.mask_cfg,
        },
        path,
    )


def build_model(device):
    model = ImgJEPA(
        mask_ckpt_path=MASK_CKPT,
        device=device,
        img_embed_dim=IMG_EMBED_DIM,
        img_depth=IMG_DEPTH,
        img_heads=IMG_HEADS,
        pred_dim=PRED_DIM,
        pred_depth=PRED_DEPTH,
        pred_heads=PRED_HEADS,
        mlp_ratio=MLP_RATIO,
        mask_which=MASK_WHICH,
    ).to(device)
    return model


def train(dataloader, model, augmentor, optimizer, epoch, df, device):
    print(f"\n\n========================== Training Epoch {epoch}===================\n\n")
    model.train()
    model.mask_encoder.eval()      # frozen target stays in eval

    running = 0.0
    idx = 0
    optimizer.zero_grad()

    for batch, (X, M, fname) in enumerate(dataloader):
        X, M = X.to(device), M.to(device)

        # GIN appearance aug on the IMAGE only (GPU, no grad through GIN), at 80%.
        use_gin = np.random.rand() < GIN_PROB
        with torch.no_grad():
            X_aug = augmentor(X) if use_gin else X

        pred, target = model(X_aug, M, target_norm=TARGET_NORM)
        loss = torch.nn.functional.l1_loss(pred, target)

        # Gradient accumulation: scale so the accumulated grad matches a true mean.
        (loss / ACCUM_STEPS).backward()

        if (batch + 1) % ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        running += loss.item()
        idx += 1
        print(f"\n Epoch {epoch} Batch {idx}/{len(dataloader)} "
              f"L1: {loss.item():.6f} ({running/idx:.6f}) gin={int(use_gin)}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Flush any remaining accumulated gradients (last partial accumulation window).
    if len(dataloader) % ACCUM_STEPS != 0:
        optimizer.step()
        optimizer.zero_grad()

    df.at[epoch, 'epoch'] = epoch
    df.at[epoch, 'loss'] = running / idx

    print(f"\n\n\n\n Epoch {epoch} Training Ended \n")
    print('\n\n Current Learning rate is : ', optimizer.param_groups[0]['lr'], '\n\n\n')
    print(f"Epoch {epoch} train L1: {running/idx:>7f}")
    print("\n\n\n\n\n")


def test(dataloader, model, epoch, df, device):
    print(f"\n\n\n ======================= Validation Epoch {epoch} ====================\n\n\n")
    model.eval()

    running = 0.0
    idx = 0

    with torch.no_grad():
        for X, M, _ in dataloader:
            X, M = X.to(device), M.to(device)

            # No GIN at validation: honest test of whether the CT-trained image
            # branch transfers to true UTE appearance.
            pred, target = model(X, M, target_norm=TARGET_NORM)
            loss = torch.nn.functional.l1_loss(pred, target).item()

            running += loss
            idx += 1
            print(f"Epoch {epoch} Batch {idx}/{len(dataloader)} "
                  f"L1: {loss:.6f} ({running/idx:.6f})")

        avg_loss = running / idx
        df.at[epoch, 'epoch'] = epoch
        df.at[epoch, 'loss'] = avg_loss

        print(f"\n\n\n\n Epoch {epoch} Validation Ended \n")
        print(f"Epoch {epoch} val L1: {avg_loss:>7f}")
        print("\n\n\n\n\n")

        global global_best_loss
        if avg_loss < global_best_loss:
            print(f"\n L1 improved from {global_best_loss:.6f} to {avg_loss:.6f}. "
                  f"Saving the current image-encoder + predictor weights\n\n\n\n\n")
            global_best_loss = avg_loss
            save_checkpoint(model, BEST_WEIGHT_PATH)
            print(f"Image-JEPA weights saved to {BEST_WEIGHT_PATH}")


def main():
    print("Image-encoder cross-modal JEPA training (CT-only train, UTE val, fixed 256^3)")

    # Train: paired geometric aug ON. Order = aniso(native) -> fit 256^3 ->
    # elastic(padded) -> mask blur -> tensor (mirrors the Step-1 mask pipeline).
    train_transform = transforms.Compose([
        RandomAnisoScale(prob=SCALE_PROB, scale_range=SCALE_RANGE),
        PadOrCrop(output_size=INPUT_SIZE, image_padval=-1, mask_padval=0),
        RandomElastic(prob=ELASTIC_PROB, alpha=ELASTIC_ALPHA, ctrl=ELASTIC_CTRL),
        MaskGaussianBlur(sigma=GAUSSIAN_SIGMA),
        ToTensor(),
    ])
    # Val: deterministic. UTE is already 256^3 so PadOrCrop is a no-op; mask still
    # cleaned (in the dataset) + softened so E_mask sees its training distribution.
    val_transform = transforms.Compose([
        PadOrCrop(output_size=INPUT_SIZE, image_padval=-1, mask_padval=0),
        MaskGaussianBlur(sigma=GAUSSIAN_SIGMA),
        ToTensor(),
    ])

    training_data = ImgJEPADataset(
        csv_file=TRAIN_CSV,
        transform=train_transform,
        training=True,                 # CT: HU clip before [-1, 1]
        clean_mask=CLEAN_MASKS,
        clean_keep_frac=CLEAN_KEEP_FRAC,
        path_replace=PATH_REPLACE,
    )
    test_data = ImgJEPADataset(
        csv_file=VAL_CSV,
        transform=val_transform,
        training=False,                # UTE: no HU clip
        clean_mask=CLEAN_MASKS,
        clean_keep_frac=CLEAN_KEEP_FRAC,
        path_replace=PATH_REPLACE,
    )

    print(device)

    model = build_model(device)
    grid = model.encoder.grid
    print(f"[img-jepa] frozen E_mask from: {MASK_CKPT} (which={MASK_WHICH})")
    print(f"[img-jepa] tokens per volume: {model.encoder.num_patches} (grid {grid})")
    print(f"[img-jepa] E_img params:    {count_parameters(model.encoder)/1e6:.2f}M")
    print(f"[img-jepa] predictor params:{count_parameters(model.predictor)/1e6:.2f}M")
    print(f"[img-jepa] E_mask params:   {count_parameters(model.mask_encoder)/1e6:.2f}M (frozen)")

    # Optimised dataloaders: pin host memory, persistent workers, prefetch so the
    # heavy CPU mask/image augmentation overlaps GPU compute. (Both the paired
    # geometric transforms and the dataset draw randomness from per-call
    # np.random.default_rng()/np.random, so workers stay diverse.)
    train_dataloader = DataLoader(
        training_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )
    test_dataloader = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )
    print('data loaded')

    # GIN augmentor: random (replay-style) appearance transform, used under
    # no_grad (no parameters optimised), image-only -- same as causality_train.py.
    augmentor = CausalityAugmentation3D(in_channels=1).to(device)

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY
    )

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    df_train = pd.DataFrame(columns=['epoch', 'loss'])
    df_test = pd.DataFrame(columns=['epoch', 'loss'])

    for t in range(EPOCHS):
        lr = lr_at(t)
        for g in optimizer.param_groups:
            g['lr'] = lr

        train(train_dataloader, model, augmentor, optimizer, t, df_train, device)
        test(test_dataloader, model, t, df_test, device)

        save_checkpoint(model, LATEST_WEIGHT_PATH)
        print(f"[per-epoch] saved image-JEPA weights to {LATEST_WEIGHT_PATH}")

        df_train.to_csv(TRAIN_LOG_CSV, index=False)
        df_test.to_csv(TEST_LOG_CSV, index=False)

        sys.stdout.flush()

    print("Done!")
    print(f'Best image-JEPA weights (lowest UTE val L1 = {global_best_loss:.6f}) saved at {BEST_WEIGHT_PATH}')


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
