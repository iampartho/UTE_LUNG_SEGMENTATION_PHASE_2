"""
Phase 1, step 1: train a FRESH mask encoder with the V-JEPA objective.

Fresh-start choices vs. the old ``train_vjepa.py``:
    * CT masks ONLY (no UTE anywhere) -> defensible "no MRI labels used" story.
    * Fixed input size 256^3 (the native UTE size). UTE needs no resize/crop at
      Phase 2; CT is fit to 256^3 by the dataset (centre pad/crop at 1.25mm by
      default, preserving physical scale). See dataset_mask_jepa.py.
    * Patch size 32 -> 256/32 = 8 -> 8^3 = 512 tokens per volume (boundary
      granularity comparable to the old 96^3 / patch-16 run, but at native scale).

Domain-generalisation augmentation (added after Phase-0 diagnostics):
    * The Phase-0 shape analysis showed the CT-only encoder treats UTE lung masks
      as off-distribution along a compactness/inflation axis (UTE lungs are more
      elongated and have lower bbox extent), and that axis drives the mask-only
      energy. An anatomy-vs-artifact check confirmed the gap is real shape, not
      the UTE speck/hole artifact. So we (a) clean every mask (largest comps +
      fill holes) and (b) apply a SYMMETRIC shape augmentation on training masks
      that varies exactly that axis: anisotropic per-axis scaling (elongation) +
      elastic deformation (extent). This widens the encoder's notion of a
      "plausible lung" to include UTE-like shapes, so a correct UTE mask is no
      longer penalised as high-energy purely for being UTE-shaped. Validation
      (UTE masks) is cleaned but NOT augmented. See dataset_mask_jepa.py.

The artefact we keep is the trained *encoder* (and its EMA target): in Phase 1
step 2 it becomes the FROZEN target/mask encoder E_mask, into whose latent the
image-encoder + predictor learn to regress. Model selection is on lowest
validation L1 (predictions live in latent space, so there is no IoU here).
"""
import os
import sys
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset_mask_jepa import MaskJEPADataset, MaskToTensor
from model_vjepa import VJEPA, generate_masks, count_parameters


# ---------------------------------------------------------------------------
# Hyperparameters / paths
# ---------------------------------------------------------------------------
# CT-only train/val splits (no UTE). copd_test is held out from copd_train,
# giving an unbiased CT set for monitoring and for the final Phase-0 check.
TRAIN_CSV = './ids/only_copd_1.25mm.csv'
VAL_CSV = './ids/only_ute_1.25mm.csv'
# Optional path remap if the CSV roots differ on this machine, e.g.
# ("/Shared/", "/Volumes/"). Leave None to use the CSV paths verbatim.
PATH_REPLACE = None

# Fixed canonical input (the native UTE size). Each dim divisible by PATCH_SIZE.
INPUT_SIZE = (256, 256, 256)
PATCH_SIZE = (32, 32, 32)        # 256/32 = 8 -> 8^3 = 512 tokens
CT_FIT_MODE = 'pad_crop'         # 'pad_crop' (preserve 1.25mm scale) | 'resize'

# Encoder ~width-384 depth-10; predictor lightweight (mirrors the old run).
ENCODER_DIM = 384
ENCODER_DEPTH = 10
ENCODER_HEADS = 6
PREDICTOR_DIM = 192
PREDICTOR_DEPTH = 4
PREDICTOR_HEADS = 4
MLP_RATIO = 4.0

# V-JEPA multi-block masking.
MASK_RATIO = 0.6
NUM_BLOCKS = 4
BLOCK_SCALE = (0.05, 0.25)
BLOCK_ASPECT = (0.75, 1.5)
TARGET_NORM = True

# Gaussian softening of the binary mask (match Phase-2 soft predictions).
GAUSSIAN_SIGMA = 1.0

# --- Mask hygiene + domain-generalisation augmentation ---------------------
# Cleaning (both train & val): keep major connected component(s) + fill holes.
# Removes the UTE speck/hole artifact; the real CT<->UTE shape gap survives it.
CLEAN_MASKS = True
CLEAN_KEEP_FRAC = 0.10
# SYMMETRIC shape augmentation (TRAIN ONLY). Broadens the CT shape distribution
# along the compactness/inflation axis that separates UTE from CT:
#   anisotropic scale -> elongation ; elastic warp -> extent. (Isotropic scale is
#   intentionally avoided: it changes neither.)
AUGMENT_TRAIN = True
SCALE_PROB = 0.5
SCALE_RANGE = (0.80, 1.25)   # per-axis, non-isotropic -> varies elongation
ELASTIC_PROB = 0.5
ELASTIC_ALPHA = 12.0         # displacement magnitude in voxels
ELASTIC_CTRL = 8            # control-grid res; smaller => smoother global bends

# At 256^3 each sample is heavy; keep the per-step batch small and recover an
# effective batch via gradient accumulation. EMA/optimiser step every ACCUM_STEPS.
BATCH_SIZE = 2
ACCUM_STEPS = 8                  # effective batch ~ BATCH_SIZE * ACCUM_STEPS
# NUM_WORKERS = 2                 # (previous) — restore this line to revert
NUM_WORKERS = 8

EPOCHS = 400
BASE_LR = 5e-4
WARMUP_EPOCHS = 15
FINAL_LR = 1e-6
WEIGHT_DECAY = 0.04
EMA_START = 0.999
EMA_END = 1.0

LOG_DIR = './log'
SAVE_DIR = './save_models'
TRAIN_LOG_CSV = os.path.join(LOG_DIR, 'train_metrics_mask_jepa_ct256_new_aug_optimised.csv')
TEST_LOG_CSV = os.path.join(LOG_DIR, 'test_metrics_mask_jepa_ct256_new_aug_optimised.csv')
BEST_WEIGHT_PATH = os.path.join(SAVE_DIR, 'best_mask_jepa_ct256_new_aug_optimised.pth')
LATEST_WEIGHT_PATH = os.path.join(SAVE_DIR, 'latest_mask_jepa_ct256_new_aug_optimised.pth')


global_best_loss = np.inf


def build_config():
    """Single source of truth for the model config (saved into checkpoints)."""
    return dict(
        in_chans=1,
        input_size=INPUT_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=ENCODER_DIM,
        depth=ENCODER_DEPTH,
        num_heads=ENCODER_HEADS,
        mlp_ratio=MLP_RATIO,
        pred_dim=PREDICTOR_DIM,
        pred_depth=PREDICTOR_DEPTH,
        pred_num_heads=PREDICTOR_HEADS,
    )


def lr_at(epoch):
    if epoch < WARMUP_EPOCHS:
        return BASE_LR * float(epoch + 1) / float(WARMUP_EPOCHS)
    progress = (epoch - WARMUP_EPOCHS) / max(1, (EPOCHS - WARMUP_EPOCHS))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return FINAL_LR + (BASE_LR - FINAL_LR) * cosine


def ema_at(epoch):
    progress = epoch / max(1, (EPOCHS - 1))
    return EMA_START + (EMA_END - EMA_START) * progress


def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            'config': build_config(),
            'fit_mode': CT_FIT_MODE,
            'aug': {
                'clean_masks': CLEAN_MASKS,
                'clean_keep_frac': CLEAN_KEEP_FRAC,
                'augment_train': AUGMENT_TRAIN,
                'scale_prob': SCALE_PROB,
                'scale_range': SCALE_RANGE,
                'elastic_prob': ELASTIC_PROB,
                'elastic_alpha': ELASTIC_ALPHA,
                'elastic_ctrl': ELASTIC_CTRL,
            },
            'encoder': model.encoder.state_dict(),
            'target_encoder': model.target_encoder.state_dict(),
            'predictor': model.predictor.state_dict(),
        },
        path,
    )


def _masks_for(batch_size, grid, device):
    return generate_masks(
        batch_size, grid,
        mask_ratio=MASK_RATIO,
        num_blocks=NUM_BLOCKS,
        block_scale=BLOCK_SCALE,
        block_aspect=BLOCK_ASPECT,
        device=device,
    )


def train(dataloader, model, optimizer, epoch, grid, momentum, df, device):
    print(f"\n\n========================== Training Epoch {epoch}===================\n\n")
    model.train()
    model.target_encoder.eval()

    running = 0.0
    idx = 0
    optimizer.zero_grad()

    for batch, (X, fname) in enumerate(dataloader):
        X = X.to(device)
        b = X.shape[0]

        keep_idx, mask_idx = _masks_for(b, grid, device)

        pred, target = model(X, keep_idx, mask_idx, target_norm=TARGET_NORM)
        loss = torch.nn.functional.l1_loss(pred, target)

        # Gradient accumulation: scale so the accumulated grad matches a true mean.
        (loss / ACCUM_STEPS).backward()

        if (batch + 1) % ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            model.update_target(momentum)  # EMA only on actual optimiser steps

        running += loss.item()
        idx += 1
        print(f"\n Epoch {epoch} Batch {idx}/{len(dataloader)} "
              f"L1: {loss.item():.6f} ({running/idx:.6f})")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Flush any remaining accumulated gradients (last partial accumulation window).
    if len(dataloader) % ACCUM_STEPS != 0:
        optimizer.step()
        optimizer.zero_grad()
        model.update_target(momentum)

    df.at[epoch, 'epoch'] = epoch
    df.at[epoch, 'loss'] = running / idx

    print(f"\n\n\n\n Epoch {epoch} Training Ended \n")
    print('\n\n Current Learning rate is : ', optimizer.param_groups[0]['lr'],
          ' | EMA momentum: ', momentum, '\n\n\n')
    print(f"Epoch {epoch} train L1: {running/idx:>7f}")
    print("\n\n\n\n\n")


def test(dataloader, model, epoch, grid, df, device):
    print(f"\n\n\n ======================= Validation Epoch {epoch} ====================\n\n\n")
    model.eval()

    running = 0.0
    idx = 0

    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device)
            b = X.shape[0]

            keep_idx, mask_idx = _masks_for(b, grid, device)

            pred, target = model(X, keep_idx, mask_idx, target_norm=TARGET_NORM)
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
                  f"Saving the current mask-encoder weights\n\n\n\n\n")
            global_best_loss = avg_loss
            save_checkpoint(model, BEST_WEIGHT_PATH)
            print(f"Mask-encoder weights saved to {BEST_WEIGHT_PATH}")


def main():
    print("Mask-encoder V-JEPA pretraining (CT-only, fixed 256^3)")

    training_data = MaskJEPADataset(
        csv_file=TRAIN_CSV,
        input_size=INPUT_SIZE,
        fit_mode=CT_FIT_MODE,
        gaussian_sigma=GAUSSIAN_SIGMA,
        clean_mask=CLEAN_MASKS,
        clean_keep_frac=CLEAN_KEEP_FRAC,
        augment=AUGMENT_TRAIN,           # train: shape augmentation ON
        scale_prob=SCALE_PROB,
        scale_range=SCALE_RANGE,
        elastic_prob=ELASTIC_PROB,
        elastic_alpha=ELASTIC_ALPHA,
        elastic_ctrl=ELASTIC_CTRL,
        path_replace=PATH_REPLACE,
        transform=transforms.Compose([MaskToTensor()]),
    )
    test_data = MaskJEPADataset(
        csv_file=VAL_CSV,
        input_size=INPUT_SIZE,
        fit_mode=CT_FIT_MODE,
        gaussian_sigma=GAUSSIAN_SIGMA,
        clean_mask=CLEAN_MASKS,
        augment=False,                   # val: deterministic, no augmentation
        path_replace=PATH_REPLACE,
        transform=transforms.Compose([MaskToTensor()]),
    )

    print(device)

    model = VJEPA(**build_config()).to(device)
    grid = model.encoder.grid  # fixed (8, 8, 8) for 256^3 / patch 32
    print(f"[mask-jepa] tokens per volume: {model.encoder.num_patches} (grid {grid})")
    print(f"[mask-jepa] encoder params:   {count_parameters(model.encoder)/1e6:.2f}M")
    print(f"[mask-jepa] predictor params: {count_parameters(model.predictor)/1e6:.2f}M")

    # --- Previous (un-optimised) dataloaders — uncomment to revert -----------
    # train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True,
    #                               drop_last=True, num_workers=NUM_WORKERS)
    # test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False,
    #                              num_workers=NUM_WORKERS)
    # --- Optimised dataloaders: pin host memory, keep workers alive across
    # epochs, and prefetch batches so the heavy CPU mask augmentation overlaps
    # with GPU compute instead of stalling it. (This dataset draws all its
    # randomness from a per-call np.random.default_rng(), so workers stay
    # diverse without a worker_init_fn.)
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

    params = list(model.encoder.parameters()) + list(model.predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=BASE_LR, weight_decay=WEIGHT_DECAY)

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    df_train = pd.DataFrame(columns=['epoch', 'loss'])
    df_test = pd.DataFrame(columns=['epoch', 'loss'])

    for t in range(EPOCHS):
        lr = lr_at(t)
        for g in optimizer.param_groups:
            g['lr'] = lr
        momentum = ema_at(t)

        train(train_dataloader, model, optimizer, t, grid, momentum, df_train, device)
        test(test_dataloader, model, t, grid, df_test, device)

        save_checkpoint(model, LATEST_WEIGHT_PATH)
        print(f"[per-epoch] saved mask-encoder weights to {LATEST_WEIGHT_PATH}")

        df_train.to_csv(TRAIN_LOG_CSV, index=False)
        df_test.to_csv(TEST_LOG_CSV, index=False)

        sys.stdout.flush()

    print("Done!")
    print(f'Best mask-encoder weights (lowest val L1 = {global_best_loss:.6f}) saved at {BEST_WEIGHT_PATH}')


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
