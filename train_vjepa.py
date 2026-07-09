"""
V-JEPA pretraining (3D lung-anatomy adaptation) -- the replacement for the
ACNN Stage-1 (``train_AE.py``).

This script mirrors the structure / logging conventions of ``train_AE.py``: a
``train(...)`` loop, a ``test(...)`` validation loop, model saving on
improvement, per-epoch CSV logs.

Objective (V-JEPA 2, Figure 2 left, Eq. 1):
    minimise_{theta, phi}  || P_phi(dy, E_theta(x)) - sg(E_theta_bar(y)) ||_1
where x is the masked (visible-token) view, y is the full volume, dy are the
learnable mask tokens, E_theta_bar is the EMA target encoder and sg is
stop-grad. The loss is applied only to the masked tokens.

There is no reconstruction / IoU metric here (predictions live in latent space),
so model selection is on the *lowest validation L1 loss*. The saved checkpoint
stores both the context encoder and the EMA target encoder plus the full config,
so Stage-2 (causality_train_with_ACNN.py) can rebuild and load the encoder.
"""
import os
import sys
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset_vjepa import VJEPAShapeDataset, MaskToTensor
from model_vjepa import VJEPA, generate_masks, count_parameters


# ---------------------------------------------------------------------------
# Hyperparameters / paths (mirroring train_AE.py for side-by-side comparison).
# ---------------------------------------------------------------------------
TRAIN_CSV = './ids/causality_train_ute_copd.csv'
VAL_CSV = './ids/causality_test_ute_copd.csv'

# Canonical fixed input. Each dim must be divisible by PATCH_SIZE. This SAME
# constant must be used in causality_train_with_ACNN.py (VJEPA_INPUT_SIZE) so the
# Stage-2 segmenter resizes its predictions to the matching size before passing
# them through the frozen encoder.
INPUT_SIZE = (96, 96, 96)
PATCH_SIZE = (16, 16, 16)        # 96/16 = 6 -> 6^3 = 216 tokens

# Encoder ~19M params (within the 10-20M target), predictor ~2M params.
ENCODER_DIM = 384
ENCODER_DEPTH = 10
ENCODER_HEADS = 6
PREDICTOR_DIM = 192
PREDICTOR_DEPTH = 4
PREDICTOR_HEADS = 4
MLP_RATIO = 4.0 # this is the ratio of the hidden dimension to the embedding dimension in the MLP

# V-JEPA multi-block masking (Sec. 2.1 / Table 9, adapted to 3 spatial axes).
MASK_RATIO = 0.6                 # fraction of tokens masked (predicted)
NUM_BLOCKS = 4
BLOCK_SCALE = (0.05, 0.25)       # per-block volume fraction
BLOCK_ASPECT = (0.75, 1.5)       # per-axis aspect ratio (paper: 0.75-1.5)
TARGET_NORM = True               # LayerNorm the EMA targets (JEPA stability)

# Gaussian softening of the binary mask (matches Stage-2 soft predictions).
GAUSSIAN_SIGMA = 1.0

BATCH_SIZE = 16
EPOCHS = 400
BASE_LR = 5e-4
WARMUP_EPOCHS = 15
FINAL_LR = 1e-6
WEIGHT_DECAY = 0.04              # paper Table 9
EMA_START = 0.999               # paper Table 10 (start)
EMA_END = 1.0                   # paper Table 10 (final)

LOG_DIR = './log'
SAVE_DIR = './save_models'
TRAIN_LOG_CSV = os.path.join(LOG_DIR, 'train_metrics_vjepa.csv')
TEST_LOG_CSV = os.path.join(LOG_DIR, 'test_metrics_vjepa.csv')
BEST_WEIGHT_PATH = os.path.join(SAVE_DIR, 'best_vjepa.pth')
LATEST_WEIGHT_PATH = os.path.join(SAVE_DIR, 'latest_vjepa.pth')


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
    """Warmup -> cosine-decay learning rate (paper's warmup-constant-decay
    family; we use a cosine cooldown which behaves comparably for short runs)."""
    if epoch < WARMUP_EPOCHS:
        return BASE_LR * float(epoch + 1) / float(WARMUP_EPOCHS)
    progress = (epoch - WARMUP_EPOCHS) / max(1, (EPOCHS - WARMUP_EPOCHS))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return FINAL_LR + (BASE_LR - FINAL_LR) * cosine


def ema_at(epoch):
    """Linear EMA momentum ramp from EMA_START to EMA_END (paper Table 10)."""
    progress = epoch / max(1, (EPOCHS - 1))
    return EMA_START + (EMA_END - EMA_START) * progress


def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            'config': build_config(),
            'encoder': model.encoder.state_dict(),
            'target_encoder': model.target_encoder.state_dict(),
            'predictor': model.predictor.state_dict(),
        },
        path,
    )


def train(dataloader, model, optimizer, epoch, grid, momentum, df, device):
    """V-JEPA training loop for a single epoch."""
    print(f"\n\n========================== Training Epoch {epoch}===================\n\n")
    model.train()
    model.target_encoder.eval()

    train_metrics = {"loss": 0.0}
    idx = 0

    for batch, (X, fname) in enumerate(dataloader):
        X = X.to(device)
        b = X.shape[0]

        keep_idx, mask_idx = generate_masks(
            b, grid,
            mask_ratio=MASK_RATIO,
            num_blocks=NUM_BLOCKS,
            block_scale=BLOCK_SCALE,
            block_aspect=BLOCK_ASPECT,
            device=device,
        )

        pred, target = model(X, keep_idx, mask_idx, target_norm=TARGET_NORM)
        loss = torch.nn.functional.l1_loss(pred, target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # EMA update of the target encoder (after the optimizer step).
        model.update_target(momentum)

        train_metrics["loss"] += loss.item()
        idx += 1
        print(f"\n Epoch {epoch} Batch {idx}/{len(dataloader)} "
              f"L1: {loss.item():.6f} ({train_metrics['loss']/idx:.6f})")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df.at[epoch, 'epoch'] = epoch
    df.at[epoch, 'loss'] = train_metrics["loss"] / idx

    print(f"\n\n\n\n Epoch {epoch} Training Ended \n")
    print('\n\n Current Learning rate is : ', optimizer.param_groups[0]['lr'],
          ' | EMA momentum: ', momentum, '\n\n\n')
    print(f"Epoch {epoch} train L1: {train_metrics['loss']/idx:>7f}")
    print("\n\n\n\n\n")


def test(dataloader, model, epoch, grid, df, device):
    """Validation loop. Saves the model on lowest validation L1 loss."""
    print(f"\n\n\n ======================= Validation Epoch {epoch} ====================\n\n\n")
    model.eval()

    test_metrics = {"running_loss": 0.0}
    idx = 0

    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device)
            b = X.shape[0]

            keep_idx, mask_idx = generate_masks(
                b, grid,
                mask_ratio=MASK_RATIO,
                num_blocks=NUM_BLOCKS,
                block_scale=BLOCK_SCALE,
                block_aspect=BLOCK_ASPECT,
                device=device,
            )

            pred, target = model(X, keep_idx, mask_idx, target_norm=TARGET_NORM)
            loss = torch.nn.functional.l1_loss(pred, target).item()

            test_metrics["running_loss"] += loss
            idx += 1
            print(f"Epoch {epoch} Batch {idx}/{len(dataloader)} "
                  f"L1: {loss:.6f} ({test_metrics['running_loss']/idx:.6f})")

        df.at[epoch, 'epoch'] = epoch
        df.at[epoch, 'loss'] = test_metrics["running_loss"] / idx

        avg_loss = test_metrics["running_loss"] / idx
        print(f"\n\n\n\n Epoch {epoch} Validation Ended \n")
        print(f"Epoch {epoch} val L1: {avg_loss:>7f}")
        print("\n\n\n\n\n")

        global global_best_loss
        if avg_loss < global_best_loss:
            print(f"\n L1 improved from {global_best_loss:.6f} to {avg_loss:.6f}. "
                  f"Saving the current V-JEPA weights\n\n\n\n\n")
            global_best_loss = avg_loss
            save_checkpoint(model, BEST_WEIGHT_PATH)
            print(f"V-JEPA weights saved to {BEST_WEIGHT_PATH}")


def main():
    print("V-JEPA pretraining (3D lung-anatomy adaptation)")

    training_data = VJEPAShapeDataset(
        csv_file=TRAIN_CSV,
        transform=transforms.Compose([MaskToTensor()]),
        input_size=INPUT_SIZE,
        gaussian_sigma=GAUSSIAN_SIGMA,
    )
    test_data = VJEPAShapeDataset(
        csv_file=VAL_CSV,
        transform=transforms.Compose([MaskToTensor()]),
        input_size=INPUT_SIZE,
        gaussian_sigma=GAUSSIAN_SIGMA,
    )

    print(device)

    model = VJEPA(**build_config()).to(device)
    grid = model.encoder.grid
    print(f"[V-JEPA] tokens per volume: {model.encoder.num_patches} (grid {grid})")
    print(f"[V-JEPA] encoder params:   {count_parameters(model.encoder)/1e6:.2f}M")
    print(f"[V-JEPA] predictor params: {count_parameters(model.predictor)/1e6:.2f}M")

    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    print('data loaded')

    # Only encoder + predictor are optimised; the target encoder is EMA-updated.
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
        print(f"[per-epoch] saved V-JEPA weights to {LATEST_WEIGHT_PATH}")

        df_train.to_csv(TRAIN_LOG_CSV, index=False)
        df_test.to_csv(TEST_LOG_CSV, index=False)

        sys.stdout.flush()

    print("Done!")
    print(f'Best V-JEPA weights (lowest val L1 = {global_best_loss:.6f}) saved at {BEST_WEIGHT_PATH}')


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
