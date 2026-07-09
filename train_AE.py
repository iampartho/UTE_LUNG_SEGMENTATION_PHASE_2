"""
Stage-1 of the ACNN approach: train the shape autoencoder on lung masks.

This script mirrors the structure / logging conventions of
``causality_train.py``: a ``train(...)`` function, a ``test(...)`` function
that drives validation, model saving on improvement, and per-epoch CSV logs.

Key differences vs. the segmenter training script:
    * The dataset returns (corrupted_mask, clean_mask, fname) -- the AE
      reconstructs the *clean* mask from the *corrupted* one (denoising AE).
    * The reconstruction loss is BCE-with-logits, matching the
      cross-entropy used in the paper for label-map reconstruction.
    * Best weight is selected based on the *highest validation IoU*
      (the user's preference) rather than the lowest validation loss.
"""
import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from monai.losses import TverskyLoss  # kept for optional comparison experiment

from dataset_AE import ShapeAEDataset, MaskToTensor
from model_AE import ShapeAE


# ---------------------------------------------------------------------------
# Hyperparameters / paths (adjust freely; mirroring the conventions in
# causality_train.py for ease of side-by-side comparison)
# ---------------------------------------------------------------------------
TRAIN_CSV = './ids/causality_train_ute_copd.csv'
VAL_CSV = './ids/causality_test_ute_copd.csv'

# Paper-faithful AE: resize every mask to AE_INPUT_SIZE before training.
# Each dim must be divisible by 8 (3 stride-2 downsamples in the encoder).
# This SAME constant must also be used in causality_train_with_ACNN.py so
# the Stage-2 segmenter resizes its predictions to the matching size before
# passing them through the frozen encoder.
AE_INPUT_SIZE = (96, 96, 96)

CODE_DIM = 64
BASE_CHANNELS = 16

NOISE_STD = 0.1
FLIP_PROB = 0.1

BATCH_SIZE = 64
EPOCHS = 200
LR = 1e-4
WEIGHT_DECAY = 1e-5  # paper Eq. 1 lambda_2 (light L2 regularisation)

LOG_DIR = './log'
SAVE_DIR = './save_models'
TRAIN_LOG_CSV = os.path.join(LOG_DIR, 'train_metrics_shape_ae.csv')
TEST_LOG_CSV = os.path.join(LOG_DIR, 'test_metrics_shape_ae.csv')
BEST_WEIGHT_PATH = os.path.join(SAVE_DIR, 'best_shape_ae.pth')
LATEST_WEIGHT_PATH = os.path.join(SAVE_DIR, 'latest_shape_ae.pth')


global_best_iou = -1.0


def calculate_iou_precision(pred_masks, true_masks, threshold=0.5):
    """Same semantics as in causality_train.py."""
    pred_mask = (pred_masks > threshold).astype(np.uint8)
    true_mask = (true_masks > threshold).astype(np.uint8)

    intersection = np.sum(np.logical_and(pred_mask, true_mask))
    union = np.sum(np.logical_or(pred_mask, true_mask))

    iou = intersection / union if union > 0 else 0.0
    precision = intersection / (np.sum(pred_mask) + 1e-6)
    return [iou, precision]


def train(dataloader, model, optimizer, epoch, criterion, df, device):
    """Training loop for a single epoch.

    Args:
        dataloader: yields (corrupted_mask, clean_mask, filename) tuples.
        model: ShapeAE instance.
        optimizer: torch optimizer.
        epoch: current epoch index (int).
        criterion: loss function for mask reconstruction. Default is BCEWithLogitsLoss
            (paper Sec. II-B, III-B cross-entropy). Swap to TverskyLoss for comparison.
        df: pandas DataFrame to log per-epoch metrics into.
        device: torch device.
    """
    print(f"\n\n========================== Training Epoch {epoch}===================\n\n")
    model.train()

    train_metrics = {"loss": 0, "IOU": 0, "precision": 0}
    idx = 0

    for batch, (X, y, fname) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        recon_logits, _ = model(X)
        loss = criterion(recon_logits, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pred_np = torch.sigmoid(recon_logits).squeeze().detach().cpu().numpy()
        label_np = y.squeeze().detach().cpu().numpy()
        pred_np[pred_np > 0.5] = 1
        pred_np[pred_np <= 0.5] = 0

        metrics = calculate_iou_precision(pred_np, label_np)

        train_metrics["IOU"] += metrics[0]
        train_metrics["precision"] += metrics[1]
        train_metrics["loss"] += loss.item()

        idx += 1
        print(f"\n Epoch {epoch} Batch {idx}/{len(dataloader)} loss: {loss.item():.4f} ({train_metrics['loss']/idx:1.4f}) \
            Precision: {metrics[1]:.4f} ({train_metrics['precision']/idx:.5f}) \
            IOU: {metrics[0]:.4f} ({train_metrics['IOU']/idx:.5f})")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df.at[epoch, 'epoch'] = epoch
    df.at[epoch, 'loss'] = train_metrics["loss"] / idx
    df.at[epoch, 'IOU'] = train_metrics['IOU'] / idx
    df.at[epoch, 'precision'] = train_metrics['precision'] / idx

    print(f"\n\n\n\n Epoch {epoch} Training Ended \n")
    print('\n\n Current Learning rate is : ', optimizer.param_groups[0]['lr'], '\n\n\n')
    print(f"Epoch {epoch} train loss: {train_metrics['loss']/idx:>7f} \
        Precision: {train_metrics['precision']/idx:>7f} \
        IOU: {train_metrics['IOU']/idx:>7f}")
    print("\n\n\n\n\n")


def test(dataloader, model, epoch, criterion, df, device):
    """Validation loop. Saves the model on highest IoU."""
    print(f"\n\n\n ======================= Validation Epoch {epoch} ====================\n\n\n")
    model.eval()

    test_metrics = {"running_loss": 0, "IOU": 0, "precision": 0}
    idx = 0

    with torch.no_grad():
        for X, y, _ in dataloader:
            X, y = X.to(device), y.to(device)

            recon_logits, _ = model(X)
            loss = criterion(recon_logits, y).item()

            pred_np = torch.sigmoid(recon_logits).squeeze().detach().cpu().numpy()
            label_np = y.squeeze().detach().cpu().numpy()

            metrics = calculate_iou_precision(pred_np, label_np)

            test_metrics["IOU"] += metrics[0]
            test_metrics["precision"] += metrics[1]
            test_metrics["running_loss"] += loss

            idx += 1
            print(f"Epoch {epoch} Batch {idx}/{len(dataloader)} loss: {loss:.4f} ({test_metrics['running_loss']/idx:.5f}) \
                Precision: {metrics[1]:.4f} ({test_metrics['precision']/idx:.5f}) \
                IOU: {metrics[0]:.4f} ({test_metrics['IOU']/idx:.5f})")

        df.at[epoch, 'epoch'] = epoch
        df.at[epoch, 'loss'] = test_metrics["running_loss"] / idx
        df.at[epoch, 'IOU'] = test_metrics['IOU'] / idx
        df.at[epoch, 'precision'] = test_metrics['precision'] / idx

        avg_iou = test_metrics['IOU'] / idx

        print(f"\n\n\n\n Epoch {epoch} Validation Ended \n")
        print(f"Epoch {epoch} loss: {test_metrics['running_loss']/idx:>7f} \
             Test  Precision: {test_metrics['precision']/idx:>7f} Test  IOU: {test_metrics['IOU']/idx:>7f} ")
        print("\n\n\n\n\n")

        global global_best_iou
        if avg_iou > global_best_iou:
            print(f"\n IoU improved from {global_best_iou:.5f} to {avg_iou:.5f} . Saving the current AE weight\n\n\n\n\n")
            global_best_iou = avg_iou
            os.makedirs(os.path.dirname(BEST_WEIGHT_PATH), exist_ok=True)
            torch.save(model.state_dict(), BEST_WEIGHT_PATH)
            print(f"AE weight saved to {BEST_WEIGHT_PATH}")


def main():
    print("ShapeAE (Stage-1 of ACNN)")

    training_data = ShapeAEDataset(
        csv_file=TRAIN_CSV,
        transform=transforms.Compose([
            MaskToTensor(),
        ]),
        training=True,
        noise_std=NOISE_STD,
        flip_prob=FLIP_PROB,
        input_size=AE_INPUT_SIZE,
    )

    test_data = ShapeAEDataset(
        csv_file=VAL_CSV,
        transform=transforms.Compose([
            MaskToTensor(),
        ]),
        training=False,
        noise_std=0.0,
        flip_prob=0.0,
        input_size=AE_INPUT_SIZE,
    )

    print(device)

    model = ShapeAE(
        code_dim=CODE_DIM,
        base_channels=BASE_CHANNELS,
        input_size=AE_INPUT_SIZE,
    ).to(device)

    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    print('data loaded')

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Paper-faithful choice (Sec. II-B, III-B): cross-entropy on label-map reconstruction.
    recon_criterion = nn.BCEWithLogitsLoss()
    # Uncomment the line below (and comment out the one above) to compare with TverskyLoss.
    # recon_criterion = TverskyLoss(sigmoid=True)

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    df_train = pd.DataFrame(columns=['epoch', 'loss', 'IOU', 'precision'])
    df_test = pd.DataFrame(columns=['epoch', 'loss', 'IOU', 'precision'])

    for t in range(EPOCHS):
        train(train_dataloader, model, optimizer, t, recon_criterion, df_train, device)
        test(test_dataloader, model, t, recon_criterion, df_test, device)

        torch.save(model.state_dict(), LATEST_WEIGHT_PATH)
        print(f"[per-epoch] saved AE weight to {LATEST_WEIGHT_PATH}")

        df_train.to_csv(TRAIN_LOG_CSV, index=False)
        df_test.to_csv(TEST_LOG_CSV, index=False)

        sys.stdout.flush()

    print("Done!")
    print(f'Best AE weight (highest val IoU = {global_best_iou:.5f}) saved at {BEST_WEIGHT_PATH}')


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
