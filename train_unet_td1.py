import os
import sys
import math
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
from monai.losses import TverskyLoss
from monai.inferers import SlidingWindowInferer
from datetime import date
from datasets_causality import CausalityDataset, ToTensor, VariableSpatialFix, RandomCrop
import argparse
import itertools
import tqdm
from basic_unet_disentagled import BasicUNet

# Local Complexity (LC) tracking during training
from compute_local_complexity_unet import UNetLocalComplexity
from eval_step_1 import normalise_hu as lc_normalise_hu, normalise_one_one as lc_normalise_one_one

# Global variables for tracking progress
global_loss = np.inf
epoch_since_loss_didnt_improve = 0

# =====================================================================
#  Local Complexity (LC) tracking configuration
# =====================================================================
LC_TRAIN_SCANS_CSV = './ids/only_copd_1.25mm.csv'
LC_VAL_SCANS_CSV = './ids/only_ute_1.25mm.csv'
LC_NUM_TRAIN_SCANS = 110
LC_INTERVAL = 400                 # Compute LC every N optimizer steps
LC_WEIGHT_START_STEP = 2000       # Only start tracking best-LC weight after this many steps
LC_RADIUS = 0.001
LC_N_HULL = 10
LC_HULL_SEED = 42
LC_HULL_BATCH_SIZE = 1
LC_MUL_FACTOR = 16                # 2 ** num_of_double_stride_conv (=4)
LC_PADVAL = 0.5
LC_LOG_DIR = './log/local_complexity_monitoring_during_training_td1_bunet'
LC_TRAIN_CSV_PATH = os.path.join(LC_LOG_DIR, 'train_lc.csv')
LC_TEST_CSV_PATH = os.path.join(LC_LOG_DIR, 'test_lc.csv')
LC_WEIGHT_SAVE_PATH = './save_models/lc_monitoring_td1_bunet/best_unet_td1_lowest_lc.pth'
PER_EPOCH_WEIGHT_DIR = './save_models/lc_monitoring_td1_bunet'

# Module-level LC tracking state (persists across epochs within a run)
global_opt_step = 0
min_train_avg_total_lc = np.inf
min_test_avg_total_lc = np.inf


def _prepare_lc_input(path, scan_type, device):
    """Replicate the preprocessing used in compute_local_complexity_unet.py."""
    arr = np.load(path).astype(np.float32)
    img = arr[:, :, :, 0]

    if scan_type == "UTE":
        img = lc_normalise_one_one(img)
    else:
        img = lc_normalise_one_one(lc_normalise_hu(img))

    h, w, d = img.shape
    new_h = LC_MUL_FACTOR * math.ceil(h / LC_MUL_FACTOR)
    new_w = LC_MUL_FACTOR * math.ceil(w / LC_MUL_FACTOR)
    new_d = LC_MUL_FACTOR * math.ceil(d / LC_MUL_FACTOR)
    img = np.pad(
        img,
        ((0, new_h - h), (0, new_w - w), (0, new_d - d)),
        'constant',
        constant_values=LC_PADVAL,
    )
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    return tensor.to(device)


def _compute_avg_lc(lc_helper, paths, scan_type, device):
    """Compute LC on every scan in `paths` and return the average total_lc and
    the average per-layer LC across all scans.

    Returns
    -------
    avg_total_lc : float
    avg_per_layer_lc : dict {layer_name: float}
    """
    layer_names = lc_helper.layer_names
    total_sum = 0.0
    per_layer_sum = {name: 0.0 for name in layer_names}
    n = 0  # number of scans

    for p in tqdm.tqdm(paths, desc="Computing LC..."):
        tensor = _prepare_lc_input(p, scan_type, device)
        r = lc_helper.compute(
            tensor,
            r=LC_RADIUS,
            n_hull=LC_N_HULL,
            seed=LC_HULL_SEED,
            hull_batch_size=LC_HULL_BATCH_SIZE,
        )
        total_sum += float(r['total_lc'])
        for name in layer_names:
            per_layer_sum[name] += float(r['per_layer_lc'][name])
        n += 1

        del tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_total_lc = total_sum / n
    avg_per_layer_lc = {name: per_layer_sum[name] / n for name in layer_names}
    return avg_total_lc, avg_per_layer_lc


def _append_lc_row(csv_path, opt_step, epoch, avg_total_lc, avg_per_layer_lc):
    """Append a single row (one optimization step) to the LC CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    row = {
        'opt_step': int(opt_step),
        'epoch': int(epoch),
        'avg_total_lc': avg_total_lc,
    }
    for name, val in avg_per_layer_lc.items():
        row[f"avg_lc_{name}"] = val

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(csv_path, index=False)


def _run_lc_eval_and_maybe_save(model, lc_helper, lc_train_paths, lc_val_paths,
                                device, opt_step, epoch):
    """Compute LC on the fixed train & validation scan subsets, log averages to
    CSV, and save the current weight as the lowest-LC checkpoint when both
    average train and test total_lc beat their historical minima (after the
    warm-up period)."""
    global min_train_avg_total_lc, min_test_avg_total_lc

    print(f"\n[LC] step {opt_step}: computing local complexity "
          f"on {len(lc_train_paths)} train scans and {len(lc_val_paths)} val scans ...")

    train_avg_total, train_avg_layers = _compute_avg_lc(lc_helper, lc_train_paths, "CT", device)
    val_avg_total, val_avg_layers = _compute_avg_lc(lc_helper, lc_val_paths, "UTE", device)

    _append_lc_row(LC_TRAIN_CSV_PATH, opt_step, epoch, train_avg_total, train_avg_layers)
    _append_lc_row(LC_TEST_CSV_PATH, opt_step, epoch, val_avg_total, val_avg_layers)

    print(f"[LC] step {opt_step}: avg train total_lc={train_avg_total:.4f}, "
          f"avg test total_lc={val_avg_total:.4f}")

    if opt_step > LC_WEIGHT_START_STEP:
        if train_avg_total < min_train_avg_total_lc and val_avg_total < min_test_avg_total_lc:
            print(f"[LC] step {opt_step}: BOTH train ({train_avg_total:.4f} < {min_train_avg_total_lc:.4f}) "
                  f"and test ({val_avg_total:.4f} < {min_test_avg_total_lc:.4f}) improved. "
                  f"Saving LC-best weight.")
            min_train_avg_total_lc = train_avg_total
            min_test_avg_total_lc = val_avg_total
            os.makedirs(os.path.dirname(LC_WEIGHT_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), LC_WEIGHT_SAVE_PATH)

def calculate_iou_precision(pred_masks, true_masks, threshold=0.5):
    pred_masks_binary = (pred_masks > threshold).astype(np.uint8)
    true_masks_binary = (true_masks > threshold).astype(np.uint8)
    
    intersection = np.sum(np.logical_and(pred_masks_binary, true_masks_binary))
    union = np.sum(np.logical_or(pred_masks_binary, true_masks_binary))
    
    iou = intersection / union if union > 0 else 0.0
    precision = intersection / (np.sum(pred_masks_binary) + 1e-6)
    
    return [iou, precision]

def train(dataloader, model, loss_fn, optimizer, epoch, net, df, scaler=None,
          lc_helper=None, lc_train_paths=None, lc_val_paths=None, device=None):
    """
    Train one epoch.

    LC monitoring args:
        lc_helper       => UNetLocalComplexity instance for computing LC every LC_INTERVAL steps
        lc_train_paths  => list of training scan file paths for LC evaluation
        lc_val_paths    => list of validation scan file paths for LC evaluation
        device          => torch device (only used for LC eval; falls back to global `device`)
    """
    print(f"\n\n========================== Training Epoch {epoch} ===================\n\n")
    model.train()
    
    
    train_metrics = {"running_loss": 0, "IOU": 0, "precision": 0}
    idx = 0

    for batch, (X, y, _fname) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        

        optimizer.zero_grad()
        
        

        pred = model(X)
        loss = loss_fn(pred, y)
        
        loss.backward()
        optimizer.step()

        # ------------------------------------------------------------------
        # Local Complexity (LC) tracking: every LC_INTERVAL optimization steps
        # compute LC on the fixed train/val scan subsets and log to CSVs.
        # Save the best-LC weight once opt_step > LC_WEIGHT_START_STEP and both
        # train & test LC sums beat their historical minima simultaneously.
        # ------------------------------------------------------------------
        global global_opt_step
        global_opt_step += 1
        if (lc_helper is not None and lc_train_paths is not None and lc_val_paths is not None
                and global_opt_step % LC_INTERVAL == 0):
            was_training = model.training
            _run_lc_eval_and_maybe_save(
                model, lc_helper, lc_train_paths, lc_val_paths,
                device, global_opt_step, epoch,
            )
            if was_training:
                model.train()

        # Metrics
        pred_np = torch.sigmoid(pred).squeeze().detach().cpu().numpy()
        label_np = y.squeeze().detach().cpu().numpy()

        iou, precision = calculate_iou_precision(pred_np, label_np)

        train_metrics['IOU'] += iou
        train_metrics['precision'] += precision
        train_metrics["running_loss"] += loss.item()
        
        idx += 1 
        print(f"\r Epoch {epoch} Batch {idx}/{len(dataloader)} loss: {loss.item():.4f} ({train_metrics['running_loss']/idx:1.4f}) \
            Precision: {precision:.4f} ({train_metrics['precision']/idx:.5f}) \
            IOU: {iou:.4f} ({train_metrics['IOU']/idx:.5f})", end="")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    df.at[epoch, 'epoch'] = epoch
    df.at[epoch, 'loss'] = train_metrics["running_loss"] / idx
    df.at[epoch, 'IOU'] = train_metrics['IOU'] / idx
    df.at[epoch, 'precision'] = train_metrics['precision'] / idx

    print(f"\n\n Epoch {epoch} Training Ended \n")
    print('Current Learning rate: ', optimizer.param_groups[0]['lr'])
    print(f"Epoch {epoch} train loss: {train_metrics['running_loss']/idx:>7f} \
        Train Precision: {train_metrics['precision'] / idx:.5f} Train IOU: {train_metrics['IOU'] / idx:.5f} ")

def test(dataloader, model, loss_fn, epoch, df, scheduler=None):
    print(f"\n\n\n ======================= Validation Epoch {epoch} ====================\n\n\n")
    model.eval()
    test_metrics = {"running_loss": 0, "IOU": 0, "precision": 0}
    idx = 0
    
    with torch.no_grad():
        for X, y, _fname in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y).item()

            pred_np = torch.sigmoid(pred).squeeze().detach().cpu().numpy()
            label_np = y.squeeze().detach().cpu().numpy()

            iou, precision = calculate_iou_precision(pred_np, label_np)

            test_metrics['IOU'] += iou
            test_metrics['precision'] += precision
            test_metrics["running_loss"] += loss
            
            idx += 1
            print(f"\r Epoch {epoch} Batch {idx}/{len(dataloader)} loss: {loss:.4f} ({test_metrics['running_loss']/idx:.5f}) \
                Precision: {precision:.4f} ({test_metrics['precision']/idx:.5f}) \
                IOU: {iou:.4f} ({test_metrics['IOU']/idx:.5f})", end="")

        df.at[epoch, 'epoch'] = epoch
        df.at[epoch, 'loss'] = test_metrics["running_loss"] / idx
        df.at[epoch, 'IOU'] = test_metrics['IOU'] / idx
        df.at[epoch, 'precision'] = test_metrics['precision'] / idx

        print(f"\n\n Epoch {epoch} Validation Ended \n")
        val_loss = test_metrics["running_loss"] / idx
        print(f"Epoch {epoch} loss: {val_loss:>7f} \
             Test Precision: {test_metrics['precision']/idx:>7f} Test IOU: {test_metrics['IOU']/idx:>7f} ")

        if scheduler:
            scheduler.step(val_loss)

        global global_loss, epoch_since_loss_didnt_improve
        if global_loss >= val_loss:
            print(f"\nLoss improved from {global_loss} to {val_loss}. Saving model...\n")
            global_loss = val_loss
            epoch_since_loss_didnt_improve = 0
            torch.save(model.state_dict(), './save_models/best_unet_td1_lc_monitoring.pth')
            print("Model saved successfully")
        else:
            epoch_since_loss_didnt_improve += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, default="bunet", help="network")
    args = parser.parse_args()
    net = args.net
    print(f"Network: {net}")

    # Dataset setup
    training_data = CausalityDataset(csv_file='./ids/only_copd_1.25mm.csv',
                                    transform=transforms.Compose([
                                        VariableSpatialFix(num_of_double_stride_conv=4),
                                        ToTensor()
                                    ]),
                                    training=True)

    test_data = CausalityDataset(csv_file='./ids/only_ute_1.25mm.csv',
                                transform=transforms.Compose([
                                    VariableSpatialFix(num_of_double_stride_conv=4),
                                    ToTensor()
                                ]),
                                training=False)

    batch_size = 1
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    print('Data loaded')

    # Model setup
    model = BasicUNet()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    
    loss_fn = TverskyLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 200
    df_train = pd.DataFrame(columns=['epoch', 'loss', 'IOU', 'precision'])
    df_test = pd.DataFrame(columns=['epoch', 'loss', 'IOU', 'precision'])

    if not os.path.exists('./save_models'):
        os.makedirs('./save_models')
    if not os.path.exists('./log'):
        os.makedirs('./log')

    # ---- Local Complexity (LC) evaluation setup ----
    os.makedirs(LC_LOG_DIR, exist_ok=True)
    os.makedirs(PER_EPOCH_WEIGHT_DIR, exist_ok=True)
    lc_train_paths = pd.read_csv(LC_TRAIN_SCANS_CSV)['filepaths'].tolist()[:LC_NUM_TRAIN_SCANS]
    lc_val_paths = pd.read_csv(LC_VAL_SCANS_CSV)['filepaths'].tolist()
    # UNetLocalComplexity accesses BasicUNet's `norm_*` attributes by name; if
    # the model is wrapped in DataParallel they live on `.module`.
    lc_target_model = model.module if isinstance(model, nn.DataParallel) else model
    lc_helper = UNetLocalComplexity(lc_target_model, device=device, target_layers="all", hook_target="norm")
    print(f"[LC] Tracking LC on {len(lc_train_paths)} train scans and {len(lc_val_paths)} val scans "
          f"every {LC_INTERVAL} opt steps; saving LC-best weight after step {LC_WEIGHT_START_STEP}.")

    for t in range(epochs):
        train(train_dataloader, model, loss_fn, optimizer, t, net, df_train,
              lc_helper=lc_helper, lc_train_paths=lc_train_paths, lc_val_paths=lc_val_paths,
              device=device)
        test(test_dataloader, model, loss_fn, t, df_test)

        # ---- Per-epoch weight snapshot (always saved, regardless of loss/LC) ----
        per_epoch_path = os.path.join(PER_EPOCH_WEIGHT_DIR, 'latest_epoch_unet_td1.pth')
        torch.save(model.state_dict(), per_epoch_path)
        print(f"[per-epoch] saved weight to {per_epoch_path}")

        df_train.to_csv(f'./log/train_metrics_{net}_td1_lc_monitoring.csv', index=False)
        df_test.to_csv(f'./log/test_metrics_{net}_td1_lc_monitoring.csv', index=False)
        
        sys.stdout.flush()

        if epoch_since_loss_didnt_improve > 100:
            print(f"\nEarly stopping at epoch {t}")
            break

    print("Training Done!")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
