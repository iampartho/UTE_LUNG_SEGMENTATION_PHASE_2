"""
Stage-2 of the ACNN approach: train the segmenter with a shape regularisation
loss computed by the *frozen* encoder of the autoencoder pretrained in
``train_AE.py``.

This script intentionally preserves the structure of ``causality_train.py``
(GIN-based augmentation, Local Complexity (LC) monitoring, identical logging
conventions, identical model and optimiser setup) and adds a single new term
to the training loss:

    L_he = || f(sigmoid(pred)) - f(y) ||^2  (Eq. 1 of Oktay et al., 2018)

where ``f`` is the frozen encoder of the AE. ``f(y)`` is computed under
``torch.no_grad()`` so no autograd graph is retained for the GT branch
(the source of the user's "no autocast / no extra graph" constraint).

Note: autocast / mixed precision is intentionally NOT used here, matching
the user's instruction. If GPU memory becomes an issue we can re-enable
autocast on the segmenter forward and/or wrap the AE encoder forward in
autocast as a follow-up.
"""
import os
import sys
import shutil
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*use_reentrant parameter should be passed explicitly.*",
    category=UserWarning,
)
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import SimpleITK as sitk
import tqdm

from monai.losses import TverskyLoss

from datasets_causality import CausalityDataset, ToTensor, VariableSpatialFix, RandomCrop
from basic_unet_disentangled import BasicUNet
from gin_with_log_capability import CausalityAugmentation3D

from compute_local_complexity_unet import UNetLocalComplexity, TARGET_SHAPE as LC_TARGET_SHAPE
from eval_step_1 import normalise_hu as lc_normalise_hu, normalise_one_one as lc_normalise_one_one

from model_AE import ShapeAE
from model_vjepa import VJEPAEncoder, gaussian_blur3d, count_parameters


# ---------------------------------------------------------------------------
# Globals (mirroring causality_train.py)
# ---------------------------------------------------------------------------
global_loss = np.inf
global_train_loss = np.inf
epoch_since_loss_didnt_improve = 0
LAMBDA_DIV = 1

# ---------------------------------------------------------------------------
# ACNN-specific configuration
# ---------------------------------------------------------------------------
# Path to the AE checkpoint produced by train_AE.py (Stage-1).
AE_CHECKPOINT = './save_models/best_shape_ae.pth'
# Must match what was used to train the AE.
AE_CODE_DIM = 64
AE_BASE_CHANNELS = 16
# Canonical fixed AE input size. The Stage-1 AE was trained on masks resized
# to this exact size (paper Sec. II-B uses an FC bottleneck which forces a
# fixed input). At Stage-2, the segmenter's prediction and the GT mask are
# resized to this size on the fly (via F.interpolate) before being passed
# through the frozen encoder. This MUST match AE_INPUT_SIZE in train_AE.py.
AE_INPUT_SIZE = (96, 96, 96)

# Weight of the latent-space shape regularisation in the segmenter loss
# (paper's lambda_1 in Eq. 1). Tune by sweeping; 0.5 is a reasonable start.
LAMBDA_HE = 0.5


# ---------------------------------------------------------------------------
# V-JEPA anatomy-encoder configuration (alternative to the ACNN ShapeAE).
#
# When USE_VJEPA_ENCODER is True, the latent anatomy-consistency loss (the
# drop-in for L_he) is computed with the frozen V-JEPA context encoder
# pretrained by train_vjepa.py instead of the ACNN ShapeAE encoder. Everything
# else in this script stays identical. The constraint is:
#
#     L_anat = || g(blur(sigmoid(pred))) - g(blur(y)) ||^2
#
# where g(.) is the mean-pooled token embedding of the frozen V-JEPA encoder and
# blur(.) is the same Gaussian smoothing used during V-JEPA pretraining (so the
# encoder sees a matching input distribution). The GT branch is under no_grad.
# ---------------------------------------------------------------------------
USE_VJEPA_ENCODER = True
# Checkpoint produced by train_vjepa.py (contains 'config', 'encoder',
# 'target_encoder').
VJEPA_CHECKPOINT = './save_models/best_vjepa.pth'
# Load the EMA target encoder (often the better downstream representation) vs.
# the context encoder. Set False to use the context encoder.
VJEPA_USE_TARGET_ENCODER = False
# Must match what train_vjepa.py used (read back from the checkpoint config, but
# kept here as the resize target on the segmenter side).
VJEPA_INPUT_SIZE = (96, 96, 96)
# Gaussian sigma used to soften masks/predictions before the encoder. MUST match
# GAUSSIAN_SIGMA in train_vjepa.py.
VJEPA_SIGMA = 1.0
VJEPA_BLUR_KERNEL = 5


# =====================================================================
# Local Complexity (LC) tracking configuration -- IDENTICAL to
# causality_train.py except that the run name is suffixed with _ACNN
# so log/save artefacts do not collide between the two runs.
# =====================================================================
COMPUTE_LOCAL_COMPLEXITY_DURING_TRAINING = True

LC_TRAIN_SCANS_CSV = './ids/only_copd_1.25mm.csv'
LC_VAL_SCANS_CSV = './ids/only_ute_1.25mm.csv'
LC_NUM_TRAIN_SCANS = 110
LC_INTERVAL = 400
LC_WEIGHT_START_STEP = 2000
LC_RADIUS = 0.001
LC_N_HULL = 10
LC_HULL_SEED = 42
LC_HULL_BATCH_SIZE = 1
# All LC scans are forced to LC_TARGET_SHAPE inside UNetLocalComplexity.compute()
# (centre-crop / zero-pad) so the hull-direction cache holds a single entry.
LC_LOG_DIR = './log/local_complexity_monitoring_during_training_70_RE_80_GIN_instance_norm_ACNN_vjepa'
LC_TRAIN_CSV_PATH = os.path.join(LC_LOG_DIR, 'train_lc.csv')
LC_TEST_CSV_PATH = os.path.join(LC_LOG_DIR, 'test_lc.csv')
LC_WEIGHT_SAVE_PATH = './save_models/lc_monitoring_70_RE_80_GIN_instance_norm_ACNN_vjepa/best_bunet_causality_lowest_lc_70_RE_80_GIN_instance_norm_ACNN_vjepa.pth'
PER_EPOCH_WEIGHT_DIR = './save_models/lc_monitoring_70_RE_80_GIN_instance_norm_ACNN_vjepa'

global_opt_step = 0
min_train_avg_total_lc = np.inf
min_test_avg_total_lc = np.inf


def _prepare_lc_input(path, scan_type, device):
    """Load and normalise a scan for LC; shape is fixed inside compute()."""
    arr = np.load(path).astype(np.float32)
    img = arr[:, :, :, 0]

    if scan_type == "UTE":
        img = lc_normalise_one_one(img)
    else:
        img = lc_normalise_one_one(lc_normalise_hu(img))

    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    return tensor.to(device)


def _compute_avg_lc(lc_helper, paths, scan_type, device):
    layer_names = lc_helper.layer_names
    total_sum = 0.0
    per_layer_sum = {name: 0.0 for name in layer_names}
    n = 0

    for p in tqdm.tqdm(paths, desc="Computing LC..."):
        tensor = _prepare_lc_input(p, scan_type, device)
        r = lc_helper.compute(
            tensor,
            r=LC_RADIUS,
            n_hull=LC_N_HULL,
            seed=LC_HULL_SEED,
            hull_batch_size=LC_HULL_BATCH_SIZE,
            target_shape=LC_TARGET_SHAPE,
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


def log_augmentation_weights(csv_path, filename, loss_value, params):
    row_data = {'filename': filename}
    for i, p in enumerate(params):
        row_data[f'kernel_size_{i}'] = p['k']
        row_data[f'kernel_{i}'] = p['w'].cpu().numpy().flatten().tolist()

    row_data['loss_value'] = float(loss_value)

    columns = ['filename']
    for i in range(len(params)):
        columns.append(f'kernel_size_{i}')
        columns.append(f'kernel_{i}')
    columns.append('loss_value')

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(columns=columns)

    if filename in df['filename'].values:
        existing_loss = df.loc[df['filename'] == filename, 'loss_value'].values[0]
        if loss_value < existing_loss:
            idx = df.index[df['filename'] == filename].tolist()[0]
            for col, val in row_data.items():
                if isinstance(val, list):
                    df.at[idx, col] = str(val)
                else:
                    df.at[idx, col] = val
            df.to_csv(csv_path, index=False)
    else:
        row_ready = {}
        for k, v in row_data.items():
            row_ready[k] = str(v) if isinstance(v, list) else v
        new_row_df = pd.DataFrame([row_ready])
        df = pd.concat([df, new_row_df], ignore_index=True)
        df.to_csv(csv_path, index=False)


def save_nifty(array, spacing, save_path):
    out_img = sitk.GetImageFromArray(array.astype(np.float32))
    out_img.SetSpacing(spacing)
    sitk.WriteImage(out_img, save_path)
    print(f"Saved: {save_path}")


def calculate_iou_precision(pred_masks, true_masks, threshold=0.5):
    pred_mask = (pred_masks > threshold).astype(np.uint8)
    true_mask = (true_masks > threshold).astype(np.uint8)

    intersection = np.sum(np.logical_and(pred_mask, true_mask))
    union = np.sum(np.logical_or(pred_mask, true_mask))

    iou = intersection / union if union > 0 else 0.0
    precision = intersection / (np.sum(pred_mask) + 1e-6)

    return [iou, precision]


class BinarySoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(BinarySoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_score.mean()


def shape_regularisation_loss(pred_logits, gt_mask, ae_model, ae_input_size):
    """L_he from Eq. 1 of Oktay et al., 2018.

    Computes the squared-L2 distance between the encoder code of
    sigmoid(pred_logits) and the encoder code of the ground-truth mask.
    Both tensors are resized to ``ae_input_size`` before passing through the
    AE because the paper's FC bottleneck is fixed-size. The trilinear resize
    is differentiable so gradients flow from L_he back into the segmenter.

    The GT branch is wrapped in ``torch.no_grad()`` so we never build an
    autograd graph for it (saves memory; matches the user's "no graph for
    the AE on the target side" requirement).

    Args:
        pred_logits: segmenter output (B, 1, H, W, D), pre-sigmoid logits,
            full (variable) spatial size.
        gt_mask: ground-truth binary mask (B, 1, H, W, D), already 0/1,
            full (variable) spatial size.
        ae_model: ShapeAE with frozen encoder, set to eval() mode.
        ae_input_size: tuple (H, W, D) the AE was trained on.

    Returns:
        torch scalar; mean of squared-distance over all latent elements
        (averaged over batch + 64 code dims). Multiply by LAMBDA_HE outside
        this function.
    """
    prob_pred = torch.sigmoid(pred_logits)
    prob_pred_resized = F.interpolate(
        prob_pred,
        size=tuple(ae_input_size),
        mode='trilinear',
        align_corners=False,
    )
    z_pred = ae_model.encode_only(prob_pred_resized)

    with torch.no_grad():
        gt_resized = F.interpolate(
            gt_mask,
            size=tuple(ae_input_size),
            mode='trilinear',
            align_corners=False,
        )
        z_gt = ae_model.encode_only(gt_resized)

    return F.mse_loss(z_pred, z_gt)


def vjepa_shape_regularisation_loss(pred_logits, gt_mask, encoder, input_size, sigma, kernel_size):
    """Anatomy-consistency loss using the frozen V-JEPA encoder (token-level).

    Drop-in analogue of ``shape_regularisation_loss`` (the ACNN L_he), using
    the frozen V-JEPA encoder as the shape prior.

    Token-level design rationale
    ----------------------------
    The encoder produces one embedding vector per 3D patch (e.g. 216 tokens for
    a 96^3 volume with 16^3 patches).  The previous design called
    ``encoder.embed`` which mean-pools all 216 vectors before comparing, so
    errors at different spatial positions can cancel each other out in the mean
    — a large over-prediction in one boundary patch can be offset by a large
    under-prediction in a neighbouring patch, making the combined loss small
    even though both positions are wrong.

    This version calls ``encoder.embed_tokens`` to keep the full
    (B, N, embed_dim) tensor and computes L1 token-by-token before reducing.
    By Jensen's inequality E[|X|] >= |E[X]|, the token-level loss is always at
    least as large as the mean-pool loss and strictly larger whenever errors
    cancel — providing a richer, position-specific gradient that directly
    penalises localised boundary mismatches (the dominant failure mode on UTE).

    Both branches are resized to the encoder's fixed ``input_size`` and Gaussian
    blurred with the same sigma used during V-JEPA pretraining, so the encoder
    sees a matching (soft) input distribution. The trilinear resize and the
    blur are differentiable, so gradients flow from this loss into the segmenter
    through the prediction branch only. The GT branch runs under
    ``torch.no_grad`` so no autograd graph is built for it.

    Args:
        pred_logits:  segmenter output (B, 1, H, W, D), pre-sigmoid logits.
        gt_mask:      ground-truth binary mask (B, 1, H, W, D), values 0/1.
        encoder:      frozen VJEPAEncoder in eval() mode.
        input_size:   tuple (H, W, D) the encoder was pretrained on.
        sigma:        Gaussian blur sigma (must match train_vjepa.py GAUSSIAN_SIGMA).
        kernel_size:  Gaussian blur kernel size.

    Returns:
        Scalar tensor; mean L1 distance over all tokens and all feature
        dimensions.  Multiply by LAMBDA_HE outside this function.
    """
    prob_pred = torch.sigmoid(pred_logits)
    prob_pred_resized = F.interpolate(
        prob_pred, size=tuple(input_size), mode='trilinear', align_corners=False,
    )
    prob_pred_resized = gaussian_blur3d(prob_pred_resized, sigma=sigma, kernel_size=kernel_size)
    # (B, N, embed_dim) — one vector per spatial patch, gradients flow back.
    tokens_pred = encoder.embed_tokens(prob_pred_resized)

    with torch.no_grad():
        gt_resized = F.interpolate(
            gt_mask, size=tuple(input_size), mode='trilinear', align_corners=False,
        )
        gt_resized = gaussian_blur3d(gt_resized, sigma=sigma, kernel_size=kernel_size)
        # (B, N, embed_dim) — same spatial layout; no gradient needed.
        tokens_gt = encoder.embed_tokens(gt_resized)

    # L1 over every (batch, token, feature) element, then averaged.
    # reduction='mean' divides by B * N * embed_dim, which is equivalent to
    # averaging the per-token L1 norms over the batch and over positions.
    return F.l1_loss(tokens_pred, tokens_gt, reduction='mean')


def train(dataloader, model, ae_model, optimizer, epoch, bce_criterion, dice_criterion,
          augmentor, df, scaler=None,
          lc_helper=None, lc_train_paths=None, lc_val_paths=None, device=None):
    """Stage-2 ACNN training loop.

    Identical to the ``train`` in causality_train.py except that the per-batch
    loss includes ``LAMBDA_HE * L_he`` (the latent-space shape prior).
    ``ae_model`` is expected to have its encoder frozen and to be in eval()
    mode. autocast is intentionally not used.
    """
    print(f"\n\n========================== Training Epoch {epoch}===================\n\n")
    size = len(dataloader.dataset)
    model.train()
    augmentor.eval()
    ae_model.eval()

    log_csv_path = './log/augmentation_weights_log_td1_lc_monitor_70_RE_80_GIN_instance_norm_ACNN_vjepa.csv'
    if not os.path.exists('./log'):
        os.makedirs('./log')

    train_metrics = {"loss": 0, "loss_seg": 0, "loss_he": 0, "IOU": 0, "precision": 0}
    idx = 0

    for batch, (X, y, fname) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        use_gin = np.random.rand() < 0.8
        gin_params = None

        with torch.no_grad():
            if use_gin:
                aug_view1 = augmentor(X)
                gin_params = augmentor.get_gin_params()
            else:
                aug_view1 = X

        pred_1 = model(aug_view1)

        loss_dice_1 = dice_criterion(pred_1, y)
        

        if isinstance(ae_model, VJEPAEncoder):
            loss_he = vjepa_shape_regularisation_loss(
                pred_1, y, ae_model, VJEPA_INPUT_SIZE, VJEPA_SIGMA, VJEPA_BLUR_KERNEL,
            )
        else:
            loss_he = shape_regularisation_loss(pred_1, y, ae_model, AE_INPUT_SIZE)

        total_loss = loss_dice_1 + LAMBDA_HE * loss_he

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if use_gin and gin_params is not None:
            f_name = fname[0] if isinstance(fname, (tuple, list)) else fname
            log_augmentation_weights(log_csv_path, f_name, total_loss.item(), gin_params)

        # ------------------------------------------------------------------
        # Local Complexity (LC) tracking: every LC_INTERVAL optimization steps
        # compute LC on the fixed train/val scan subsets and log to CSVs.
        # ------------------------------------------------------------------
        global global_opt_step
        global_opt_step += 1
        if (COMPUTE_LOCAL_COMPLEXITY_DURING_TRAINING
                and lc_helper is not None and lc_train_paths is not None and lc_val_paths is not None
                and global_opt_step % LC_INTERVAL == 0):
            was_training = model.training
            _run_lc_eval_and_maybe_save(
                model, lc_helper, lc_train_paths, lc_val_paths,
                device, global_opt_step, epoch,
            )
            if was_training:
                model.train()
                augmentor.eval()
                ae_model.eval()

        pred_np = torch.sigmoid(pred_1).squeeze().detach().cpu().numpy()
        label_np = y.squeeze().detach().cpu().numpy()

        pred_np[pred_np > 0.5] = 1
        pred_np[pred_np <= 0.5] = 0

        metrics = calculate_iou_precision(pred_np, label_np)

        train_metrics['IOU'] += metrics[0]
        train_metrics['precision'] += metrics[1]
        train_metrics["loss"] += total_loss.item()
        train_metrics["loss_seg"] += loss_dice_1.item()
        train_metrics["loss_he"] += loss_he.item()

        idx = idx + 1
        print(f"\n Epoch {epoch} Batch {idx}/{len(dataloader)} loss: {total_loss.item():.4f} ({train_metrics['loss']/idx:1.4f}) \
            seg: {loss_dice_1.item():.4f} ({train_metrics['loss_seg']/idx:.4f}) \
            L_he: {loss_he.item():.6f} ({train_metrics['loss_he']/idx:.6f}) \
            Precision: {metrics[1]:.4f} ({train_metrics['precision']/idx:.5f}) \
            IOU: {metrics[0]:.4f} ({train_metrics['IOU']/idx:.5f})")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df.at[epoch, 'epoch'] = epoch
    df.at[epoch, 'loss'] = train_metrics["loss"] / idx
    df.at[epoch, 'loss_seg'] = train_metrics["loss_seg"] / idx
    df.at[epoch, 'loss_he'] = train_metrics["loss_he"] / idx
    df.at[epoch, 'IOU'] = train_metrics['IOU'] / idx
    df.at[epoch, 'precision'] = train_metrics['precision'] / idx

    print(f"\n\n\n\n Epoch {epoch} Training Ended \n")
    print('\n\n Current Learning rate is : ', optimizer.param_groups[0]['lr'], '\n\n\n')
    print(f"Epoch {epoch} train loss: {train_metrics['loss']/idx:>7f} \
        seg: {train_metrics['loss_seg']/idx:>7f} \
        L_he: {train_metrics['loss_he']/idx:>7f} \
        Precision: {train_metrics['precision']/idx:>7f} \
        IOU: {train_metrics['IOU']/idx:>7f}")
    print("\n\n\n\n\n")


def test(dataloader, model, epoch, bce_criterion, dice_criterion, df, scheduler=None):
    """Validation loop. Identical to causality_train.py.test (the AE is not
    used at validation time -- L_he is a training-only regulariser)."""
    print(f"\n\n\n ======================= Validation Epoch {epoch} ====================\n\n\n")
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    print("Using Sliding Window Inference with ROI size (192, 192, 96)")

    model.eval()
    test_loss, correct = 0, 0
    idx = 0
    test_metrics = {"running_loss": 0, "IOU": 0, "precision": 0}

    log_csv_path = './log/augmentation_weights_log_td1_lc_monitor_70_RE_80_GIN_instance_norm_ACNN_vjepa.csv'
    save_on_best_test_log_csv_path = './log/augmentation_weights_log_td1_lc_monitor_70_RE_80_GIN_instance_norm_ACNN_vjepa_saved_on_best_test.csv'

    with torch.no_grad():
        for X, y, _ in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss = dice_criterion(pred, y).item()

            pred_np = torch.sigmoid(pred).squeeze().detach().cpu().numpy()
            label_np = y.squeeze().detach().cpu().numpy()

            metrics = calculate_iou_precision(pred_np, label_np)

            test_metrics['IOU'] += metrics[0]
            test_metrics['precision'] += metrics[1]
            test_metrics["running_loss"] += test_loss

            idx = idx + 1

            print(f"Epoch {epoch} Batch {idx}/{len(dataloader)} loss: {test_loss:.4f} ({test_metrics['running_loss']/idx:.5f}) \
                Precision: {metrics[1]:.4f} ({test_metrics['precision']/idx:.5f}) \
                IOU: {metrics[0]:.4f} ({test_metrics['IOU']/idx:.5f})")

        df.at[epoch, 'epoch'] = epoch
        df.at[epoch, 'loss'] = test_metrics["running_loss"] / idx
        df.at[epoch, 'IOU'] = test_metrics['IOU'] / idx
        df.at[epoch, 'precision'] = test_metrics['precision'] / idx

        print(f"\n\n\n\n Epoch {epoch} Validation Ended \n")
        print(f"Epoch {epoch} loss: {test_metrics['running_loss']/idx:>7f} \
             Test  Precision: {test_metrics['precision']/idx:>7f} Test  IOU: {test_metrics['IOU']/idx:>7f} ")
        print("\n\n\n\n\n")

        val_loss = test_metrics["running_loss"] / idx
        if scheduler:
            scheduler.step(val_loss)

        global global_loss, epoch_since_loss_didnt_improve
        if global_loss >= val_loss:
            print(f"\nLoss improved from {global_loss} to {val_loss} . Saving the current weight\n\n\n\n\n")
            global_loss = val_loss
            epoch_since_loss_didnt_improve = 0
            torch.save(model.state_dict(), './save_models/' + 'best_bunet_td1_lc_monitor_70_RE_80_GIN_instance_norm_ACNN_vjepa.pth')
            print("Model saved successfully")
            if os.path.exists(log_csv_path):
                shutil.copy(log_csv_path, save_on_best_test_log_csv_path)
        else:
            epoch_since_loss_didnt_improve += 1


def main():

    net = "unet"
    print(net)
    crop_size = (256, 128, 256)
    training_data = CausalityDataset(
        csv_file='./ids/only_copd_1.25mm.csv',
        transform=transforms.Compose([
            VariableSpatialFix(num_of_double_stride_conv=4),
            ToTensor(),
        ]),
        training=True,
    )

    test_data = CausalityDataset(
        csv_file='./ids/only_ute_1.25mm.csv',
        transform=transforms.Compose([
            VariableSpatialFix(num_of_double_stride_conv=4),
            ToTensor(),
        ]),
        training=False,
    )

    print(device)

    print("bunet")
    batch_size = 1
    gen = BasicUNet()
    gen = gen.to(device)

    model_checkpoint = ''
    if model_checkpoint:
        gen.load_state_dict(torch.load(model_checkpoint))
        print("Model Loaded successfully and Normalization layers frozen")

    # ------------------------------------------------------------------
    # Load the Stage-1 anatomy encoder and freeze it. Either the ACNN ShapeAE
    # (train_AE.py) or the V-JEPA encoder (train_vjepa.py), selected by
    # USE_VJEPA_ENCODER. In both cases the frozen encoder provides the latent
    # space for the anatomy-consistency loss (the drop-in for L_he).
    # ------------------------------------------------------------------
    if USE_VJEPA_ENCODER:
        if not os.path.exists(VJEPA_CHECKPOINT):
            raise FileNotFoundError(
                f"V-JEPA checkpoint not found at {VJEPA_CHECKPOINT}. "
                f"Run train_vjepa.py first to produce it.")
        ckpt = torch.load(VJEPA_CHECKPOINT, map_location=device)
        cfg = ckpt['config']
        # Rebuild a bare encoder (the checkpoint config carries the encoder
        # keys; predictor-only keys are ignored by VJEPAEncoder).
        ae_model = VJEPAEncoder(
            in_chans=cfg['in_chans'],
            input_size=cfg['input_size'],
            patch_size=cfg['patch_size'],
            embed_dim=cfg['embed_dim'],
            depth=cfg['depth'],
            num_heads=cfg['num_heads'],
            mlp_ratio=cfg['mlp_ratio'],
        ).to(device)
        which = 'target_encoder' if VJEPA_USE_TARGET_ENCODER else 'encoder'
        ae_model.load_state_dict(ckpt[which])
        ae_model.freeze()
        ae_model.eval()
        print(f"[V-JEPA] loaded frozen '{which}' from {VJEPA_CHECKPOINT} "
              f"({count_parameters(ae_model)/1e6:.2f}M params); "
              f"L_anat weighted by lambda_he={LAMBDA_HE}")
    else:
        ae_model = ShapeAE(
            code_dim=AE_CODE_DIM,
            base_channels=AE_BASE_CHANNELS,
            input_size=AE_INPUT_SIZE,
        ).to(device)
        if not os.path.exists(AE_CHECKPOINT):
            raise FileNotFoundError(
                f"Stage-1 AE checkpoint not found at {AE_CHECKPOINT}. "
                f"Run train_AE.py first to produce it.")
        ae_model.load_state_dict(torch.load(AE_CHECKPOINT, map_location=device))
        ae_model.freeze_encoder()
        ae_model.eval()
        print(f"[ACNN] loaded frozen AE encoder from {AE_CHECKPOINT}; "
              f"L_he weighted by lambda_he={LAMBDA_HE}")

    scaler = None  # autocast intentionally disabled per user request

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    print('data loaded')

    optimizer = torch.optim.Adam(gen.parameters(), lr=1e-4)
    scheduler = None

    print('training data')

    epochs = 5000
    df_train = pd.DataFrame(columns=['epoch', 'loss', 'loss_seg', 'loss_he', 'IOU', 'precision'])
    df_test = pd.DataFrame(columns=['epoch', 'loss', 'IOU', 'precision'])

    bce_criterion = nn.BCEWithLogitsLoss()
    dice_criterion = TverskyLoss(sigmoid=True)
    augmentor = CausalityAugmentation3D(in_channels=1).to(device)

    if COMPUTE_LOCAL_COMPLEXITY_DURING_TRAINING:
        os.makedirs(LC_LOG_DIR, exist_ok=True)
        os.makedirs(PER_EPOCH_WEIGHT_DIR, exist_ok=True)
        lc_train_paths = pd.read_csv(LC_TRAIN_SCANS_CSV)['filepaths'].tolist()[:LC_NUM_TRAIN_SCANS]
        lc_val_paths = pd.read_csv(LC_VAL_SCANS_CSV)['filepaths'].tolist()
        lc_helper = UNetLocalComplexity(gen, device=device, target_layers="all", hook_target="norm")
        print(f"[LC] Tracking LC on {len(lc_train_paths)} train scans and {len(lc_val_paths)} val scans "
              f"every {LC_INTERVAL} opt steps (target shape {LC_TARGET_SHAPE}); "
              f"saving LC-best weight after step {LC_WEIGHT_START_STEP}.")
    else:
        lc_helper = lc_train_paths = lc_val_paths = None
        print("[LC] Disabled (COMPUTE_LOCAL_COMPLEXITY_DURING_TRAINING=False).")

    for t in range(epochs):
        train(train_dataloader, gen, ae_model, optimizer, t, bce_criterion, dice_criterion,
              augmentor, df_train, scaler,
              lc_helper=lc_helper, lc_train_paths=lc_train_paths, lc_val_paths=lc_val_paths, device=device)

        test(test_dataloader, gen, t, bce_criterion, dice_criterion, df_test, scheduler)

        if COMPUTE_LOCAL_COMPLEXITY_DURING_TRAINING:
            per_epoch_path = os.path.join(PER_EPOCH_WEIGHT_DIR,
                                          'latest_epoch_bunet_causality_70_RE_80_GIN_instance_norm_ACNN_vjepa.pth')
            torch.save(gen.state_dict(), per_epoch_path)
            print(f"[per-epoch] saved weight to {per_epoch_path}")

        df_train.to_csv('./log/train_metrics_' + net + '_td1_lc_monitor_70_RE_80_GIN_instance_norm_ACNN_vjepa.csv', index=False)
        df_test.to_csv('./log/test_metrics_' + net + '_td1_lc_monitor_70_RE_80_GIN_instance_norm_ACNN_vjepa.csv', index=False)

        sys.stdout.flush()

    print("Done!")
    print('Saved Pytorch model to model.pth')


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main()
