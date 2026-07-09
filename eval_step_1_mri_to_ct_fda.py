"""
eval_step_1_mri_to_ct_fda.py
=============================

Two-stage MRI lung-segmentation pipeline using target-aware Fourier Domain
Adaptation (FDA) for domain translation instead of a CycleGAN generator.

Stage 1 — Translation  (MRI → FDA-translated "CT-like" image)
    Pick one random real CT volume from CT_CSV.  For each test MRI:
      1. Normalise both MRI and CT to [-1, 1].
      2. Resize the MRI to match the CT shape (crop / zero-pad, no resampling).
      3. Apply ``fda_mri_to_ct``:
           - Keep the MRI's Fourier *phase* (anatomy / edges stay MRI-like)
           - Overwrite the MRI's *low-frequency amplitude* cube with the CT's
             (transfers the CT's overall brightness / contrast "look")
           beta = 0.01  →  only the very lowest frequency content is swapped,
           leaving anatomy largely intact.
      4. Renormalise the FDA output back to [-1, 1] before feeding the UNet.

Stage 2 — Segmentation  (FDA image → lung mask)
    Apply VariableSpatialFix + ToTensor (identical to eval_step_1.py), then
    run the trained BasicUNet segmentation model exactly as in eval_step_1.py.

Outputs written to OUTPUT_DIR for each input file:
    *_pred.nii.gz   — binary lung segmentation mask
    *_gt.nii.gz     — ground-truth lung mask (for Dice / evaluation)
    *_fda_ct.nii.gz — FDA-translated image (MRI phase + CT amplitude), for QC

This script is intentionally kept separate from eval_step_1.py and
eval_step_1_mri_to_ct.py so neither existing evaluation path is altered.
"""

import os
import random

import torch
import pandas as pd
import numpy as np
from torchvision import transforms
import tqdm

from basic_unet_disentangled_no_normalising import BasicUNet

# Normalisation helpers and transform classes shared across the codebase.
from eval_step_1 import (
    normalise_hu,
    normalise_one_one,
    normalise_zero_one,
    VariableSpatialFix,
    ToTensor,
    generate_nifty,
)


# ======================================================================
#  CONFIG  — edit then run; no argparse by design
# ======================================================================

TEST_CSV = "./ids/only_ute_1.25mm.csv"    # UTE / MRI test cases
CT_CSV   = "./ids/only_copd_1.25mm.csv"   # pool from which the random CT is drawn

# Trained segmentation model checkpoint (same as eval_step_1.py).
SEG_CHECKPOINT = (
    "./save_models/current_best_may_2026/"
    "best_bunet_causality_paper_ct_train_UTE_test_w_tversky_wo_kl_only_gin_roughness_enforced.pth"
)

OUTPUT_DIR = "./prediction/mri_to_ct_fda"
SPACING    = (1.25, 1.25, 1.25)

# FDA "beta": half-width of the centred low-frequency amplitude cube that is
# copied from the CT onto the MRI, expressed as a fraction of each spatial
# dimension.  0.01 copies only the very lowest frequencies (DC + neighbour).
FDA_BETA = 0.01

# Seed for reproducible random CT selection.
SEED = 42


# ======================================================================
#  Model loading
# ======================================================================

def load_seg_model(checkpoint_path: str, device: torch.device) -> BasicUNet:
    """Load the lung-segmentation BasicUNet, tolerating legacy key mismatches."""
    model = BasicUNet().to(device)
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=device)
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            print(
                f"[seg model] missing keys ({len(result.missing_keys)}): "
                f"{result.missing_keys[:8]}"
                f"{'...' if len(result.missing_keys) > 8 else ''}"
            )
        if result.unexpected_keys:
            print(
                f"[seg model] ignored unexpected keys ({len(result.unexpected_keys)}): "
                f"{result.unexpected_keys[:8]}"
                f"{'...' if len(result.unexpected_keys) > 8 else ''}"
            )
    model.eval()
    return model


# ======================================================================
#  FDA helpers  (ported from visualize_gin_vs_fda.py)
# ======================================================================

def resize_to(volume: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resize ``volume`` to ``target_shape`` by centre-cropping or zero-padding.

    No trilinear resampling — only trimming and zero-padding — so real voxel
    values are never interpolated.  Required by ``fda_mri_to_ct`` which needs
    both arrays to have identical shapes so their FFTs line up.
    """
    result = volume
    for axis_index in range(result.ndim):
        current_len = result.shape[axis_index]
        target_len  = target_shape[axis_index]

        if current_len == target_len:
            continue

        if current_len > target_len:
            extra       = current_len - target_len
            start       = extra // 2
            slicer      = [slice(None)] * result.ndim
            slicer[axis_index] = slice(start, start + target_len)
            result = result[tuple(slicer)]
        else:
            to_add  = target_len - current_len
            pad_l   = to_add // 2
            pad_r   = to_add - pad_l
            padding = [(0, 0)] * result.ndim
            padding[axis_index] = (pad_l, pad_r)
            result = np.pad(result, padding, mode="constant", constant_values=0)

    assert result.shape == tuple(target_shape), (
        f"resize_to failed: got {result.shape}, expected {target_shape}"
    )
    return result.astype(np.float32)


def fda_mri_to_ct(mri_volume: np.ndarray, ct_volume: np.ndarray, beta: float) -> np.ndarray:
    """Target-aware FDA: MRI phase + CT low-frequency amplitude.

    Keeps the MRI's Fourier phase (which carries anatomy / edges) and replaces
    the MRI's low-frequency amplitude with the CT's, giving the MRI the CT's
    overall brightness / contrast "look" without distorting its structure.

    Algorithm (Yang & Soatto, 2020, extended to 3D):
      1. FFT both volumes.
      2. Shift DC to the centre so the low-frequency region is a centred cube.
      3. Overwrite the MRI's low-frequency amplitude (half-width = beta * dim)
         with the CT's low-frequency amplitude.
      4. Un-shift, recombine the modified amplitude with the MRI phase, invert.

    Both volumes must share the same shape (call ``resize_to`` first).
    """
    assert mri_volume.shape == ct_volume.shape, (
        f"FDA needs matching shapes; got {mri_volume.shape} vs {ct_volume.shape}"
    )

    fft_mri = np.fft.fftn(mri_volume)
    fft_ct  = np.fft.fftn(ct_volume)

    amp_mri, pha_mri = np.abs(fft_mri), np.angle(fft_mri)
    amp_ct           = np.abs(fft_ct)

    amp_mri_sh = np.fft.fftshift(amp_mri)
    amp_ct_sh  = np.fft.fftshift(amp_ct)

    d, h, w = mri_volume.shape
    cz, cy, cx = d // 2, h // 2, w // 2
    bz = max(1, int(beta * d))
    by = max(1, int(beta * h))
    bx = max(1, int(beta * w))

    amp_mri_sh[cz - bz:cz + bz + 1,
               cy - by:cy + by + 1,
               cx - bx:cx + bx + 1] = amp_ct_sh[cz - bz:cz + bz + 1,
                                                  cy - by:cy + by + 1,
                                                  cx - bx:cx + bx + 1]

    amp_mri_new = np.fft.ifftshift(amp_mri_sh)
    fft_new     = amp_mri_new * np.exp(1j * pha_mri)
    return np.real(np.fft.ifftn(fft_new)).astype(np.float32)


# ======================================================================
#  Main
# ======================================================================

if __name__ == "__main__":

    np.random.seed(SEED)
    random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[config] device         = {device}")
    print(f"[config] test csv       = {TEST_CSV}")
    print(f"[config] ct csv         = {CT_CSV}")
    print(f"[config] seg checkpoint = {SEG_CHECKPOINT}")
    print(f"[config] output dir     = {OUTPUT_DIR}")
    print(f"[config] fda beta       = {FDA_BETA}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Pick one random CT volume to serve as the FDA amplitude source.
    # ------------------------------------------------------------------
    ct_files = pd.read_csv(CT_CSV)["filepaths"].values
    ct_path  = np.random.choice(ct_files)
    print(f"\n[FDA] Using random CT for amplitude transfer: {ct_path}")

    ct_arr  = np.load(ct_path)
    ct_norm = normalise_one_one(normalise_hu(ct_arr[:, :, :, 0])).astype(np.float32)
    print(f"[FDA] CT shape: {ct_norm.shape}")

    # ------------------------------------------------------------------
    # 2. Load segmentation model.
    # ------------------------------------------------------------------
    print("\n[model] Loading lung-segmentation model ...")
    seg_model = load_seg_model(SEG_CHECKPOINT, device)

    # ------------------------------------------------------------------
    # Transforms (identical to eval_step_1.py)
    # ------------------------------------------------------------------
    transform = transforms.Compose([
        VariableSpatialFix(num_of_double_stride_conv=4, padval=0.5),
        ToTensor(),
    ])

    # ------------------------------------------------------------------
    # 3. Inference loop
    # ------------------------------------------------------------------
    test_df   = pd.read_csv(TEST_CSV)
    all_files = test_df["filepaths"].values

    print(f"\nRunning two-stage (FDA + seg) inference on {len(all_files)} files ...\n")

    for idx, each_file in tqdm.tqdm(enumerate(all_files), total=len(all_files)):

        arr    = np.load(each_file)
        img_np = arr[:, :, :, 0]   # raw MRI intensity
        img_gt = arr[:, :, :, 1]   # ground-truth lung mask

        # --- Stage 1: FDA MRI → CT-like image -------------------------------
        # normalise_one_one matches the same convention used for the CT above.
        mri_norm = normalise_one_one(img_np).astype(np.float32)
        gt_norm  = normalise_zero_one(img_gt)

        # FDA requires both arrays to have the same shape.
        mri_resized = resize_to(mri_norm, ct_norm.shape)

        # MRI phase + CT low-frequency amplitude → "MRI wearing CT's look"
        fda_np = fda_mri_to_ct(mri_resized, ct_norm, FDA_BETA)

        # Renormalise the FDA output to [-1, 1] to match what the segmentation
        # model expects (trained on normalised CT images in that range).
        fda_np = normalise_one_one(fda_np)

        # --- Stage 2: pad, tensorise, segment the FDA-translated image -------
        fda_tensor, gt_tensor = transform([fda_np, gt_norm])
        fda_tensor = fda_tensor.unsqueeze(0).to(device)   # [1, 1, D, H, W]

        with torch.no_grad():
            pred = seg_model(fda_tensor)
            pred = torch.sigmoid(pred)
            pred[pred >  0.5] = 1
            pred[pred <= 0.5] = 0

        pred_np = pred.detach().squeeze().cpu().numpy().astype(np.uint8)

        # --- Save outputs ---------------------------------------------------
        filename = f"{'_'.join(each_file.split('/')[-4:])}".split(".npy")[0]

        generate_nifty(pred_np,
                       SPACING,
                       f"{OUTPUT_DIR}/{filename}_pred.nii.gz")

        generate_nifty(gt_tensor.squeeze().cpu().numpy(),
                       SPACING,
                       f"{OUTPUT_DIR}/{filename}_gt.nii.gz")

        # Save the FDA-translated image for visual QC in Slicer.
        generate_nifty(fda_np,
                       SPACING,
                       f"{OUTPUT_DIR}/{filename}_fda_ct.nii.gz")

    print(f"\nDone.  Results written to {OUTPUT_DIR}")
    print("  *_pred.nii.gz  — segmentation prediction")
    print("  *_gt.nii.gz    — ground-truth mask")
    print("  *_fda_ct.nii.gz — FDA-translated image (MRI phase + CT amplitude, for QC)")
