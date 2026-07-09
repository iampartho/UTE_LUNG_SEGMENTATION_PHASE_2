"""
eval_step_1_frequency_filter.py
================================

A copy of the evaluation flow in ``eval_step_1.py`` whose ONLY functional
addition is a *spatial-frequency filter* applied to every UTE MRI volume
**before** it is fed to the segmentation network.

Why this script exists
-----------------------
We want to empirically test the hypothesis (inspired by Balestriero & LeCun,
2402.11337) that the segmenter generalises better when it is shown the
"structure / edge" content of the image (high spatial frequency) versus the
"smooth appearance" content (low spatial frequency).  Instead of doing a true
PCA subspace projection (ill-defined on variable-size, unregistered 3D
volumes) we approximate the paper's top/bottom subspace split with a
frequency-domain filter:

    * low-pass   -> keeps the smooth, low-frequency appearance (blurry image,
                    paper's "top variance" subspace).
    * high-pass  -> keeps edges / outlines (paper's "bottom variance" subspace).
    * band-pass  -> keeps a mid-frequency band (edges without the noisiest
                    highest frequencies).

The script ALSO writes the filtered volume to disk as ``.nii.gz`` so you can
open it in Slicer next to the original and judge visually whether the filter
is making the image noisier (high-pass amplifying noise) or smoother
(low-pass), and reason about the segmentation result accordingly.

How to use
----------
Edit the CONFIG block below.  The single most important knob is
``FILTER_TYPE``.  No argparse is used on purpose - just change the variable
and re-run.
"""

import os

import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk
from torchvision import transforms
import tqdm

# Same model / helpers as eval_step_1.py so behaviour matches the baseline.
from basic_unet_disentangled import BasicUNet
from eval_step_1 import (
    normalise_hu,
    normalise_zero_one,
    normalise_one_one,
    VariableSpatialFix,
    ToTensor,
    generate_nifty,
)


# =====================================================================
#  CONFIG  -- edit these variables, then run.  (No argparse by design.)
# =====================================================================

# --- Which filter to apply to the input MRI before segmentation. ---
# One of: "none", "lowpass", "highpass", "bandpass".
#   "none"     -> sanity baseline, identical to eval_step_1.py.
#   "lowpass"  -> keep frequencies BELOW LOW_CUTOFF  (smooth / blurry image).
#   "highpass" -> keep frequencies ABOVE LOW_CUTOFF  (edges / outlines).
#   "bandpass" -> keep frequencies between LOW_CUTOFF and HIGH_CUTOFF.
FILTER_TYPE = "bandpass"

# --- Filter cutoffs, expressed as a fraction of the Nyquist frequency. ---
# Nyquist frequency is half of the sampling frequency. Sampling frequency is 1/voxel spacing.
# Range is (0, 1]; 1.0 == Nyquist (the highest representable frequency).
#   * For "lowpass":  only LOW_CUTOFF is used (sigma of the Gaussian).
#   * For "highpass": only LOW_CUTOFF is used (everything above it passes).
#   * For "bandpass": BOTH are used and we require HIGH_CUTOFF > LOW_CUTOFF.
# Smaller LOW_CUTOFF  -> more aggressive (high-pass keeps only the sharpest edges).
# Larger  LOW_CUTOFF  -> gentler        (low-pass keeps more detail).
LOW_CUTOFF = 0.20
HIGH_CUTOFF = 0.40

# --- Shape of the filter transition. ---
#   "gaussian" -> smooth roll-off (recommended; avoids ringing artefacts).
#   "ideal"    -> hard brick-wall cut (sharper split but introduces ringing).
FILTER_SHAPE = "gaussian"

# After filtering, re-stretch the volume back to the [-1, 1] range the network
# was trained on.  Strongly recommended for high/band-pass (those outputs are
# ~zero-mean and would otherwise sit in a tiny intensity range the net never
# saw).  Set False only if you want to inspect the raw filter response.
# NOTE: ignored when GIN_STYLE_BLEND is True (the blend handles magnitude itself).
RENORMALISE_AFTER_FILTER = True

# --- GIN-style post-processing of the filtered volume. ---
# Your GIN augmentor never feeds a "pure" transformed image to the network: it
# (1) blends the transform back with the original input, and (2) rescales the
# blend so its Frobenius norm equals the original's.  A purely filtered volume
# therefore looks out-of-distribution (missing low-freq content + wrong energy).
# Setting this True replicates GIN's post-processing so the filtered input lands
# back on the manifold the network was trained on:
#       mixed = alpha * filtered + (1 - alpha) * original
#       mixed = mixed * ||original||_F / ||mixed||_F        (Frobenius match)
# Caveat: blending re-injects the full-spectrum original, so this tests
# "frequency emphasis while in-distribution", not a pure subspace projection.
GIN_STYLE_BLEND = True

# Blend weight on the FILTERED volume (GIN uses a random alpha; we fix it).
#   alpha = 0.5 -> equal mix of filtered and original (as requested).
#   alpha = 1.0 -> pure filtered (then only frob-matched, no original re-injected).
BLEND_ALPHA = 0.5

# --- Data / model paths (mirrors eval_step_1.py). ---
INPUT_CSV = "./ids/only_ute_1.25mm.csv"
MODEL_CHECKPOINT = (
    "./save_models/best_bunet_causality_paper_ct_train_UTE_test_w_tversky_wo_kl_only_gin_roughness_enforced.pth"
)

# Output folder.  Filtered inputs, predictions and GTs all land here.  The
# folder name embeds the filter setting so different runs don't overwrite
# each other.
OUTPUT_DIR = f"./prediction/freq_filter_{FILTER_TYPE}_{LOW_CUTOFF}_{HIGH_CUTOFF}"

# Voxel spacing written into every .nii.gz (purely for correct display scale).
SPACING = (1.25, 1.25, 1.25)

# Limit how many scans to process (None = all rows in the CSV).
MAX_SCANS = None

# Pad input so every spatial dim is a multiple of 2**4 (= 16), matching the
# 4 down-sampling stages of BasicUNet.  Same as eval_step_1.py.
NUM_DOUBLE_STRIDE_CONV = 4
PAD_VALUE = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
#  Frequency-domain filtering
# =====================================================================

def _radial_frequency_grid(shape):
    """Return a 3D array of normalised radial frequency magnitudes.

    For each axis we use ``np.fft.fftfreq`` (cycles/sample, in [-0.5, 0.5)) and
    divide by 0.5 so that a value of 1.0 corresponds exactly to the Nyquist
    frequency.  The returned grid is fft-shifted (DC at the centre) so it lines
    up with the fft-shifted spectrum we build below.

    Parameters
    ----------
    shape : tuple(int, int, int) -- (D, H, W) of the volume.

    Returns
    -------
    rho : np.ndarray, same shape, where rho == 0 at DC and grows outward.
          A voxel at the cube face along one axis has rho ~ 1.0 (Nyquist);
          the cube corners can exceed 1.0 (up to sqrt(3)), which is expected.
    """
    axes = []
    for n in shape:
        f = np.fft.fftshift(np.fft.fftfreq(n))  # [-0.5, 0.5), DC at centre
        axes.append(f / 0.5)                    # normalise: Nyquist -> 1.0
    fz, fy, fx = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    return np.sqrt(fz ** 2 + fy ** 2 + fx ** 2)


def _lowpass_mask(shape, cutoff, shape_kind):
    """Build a low-pass transfer function H(rho) in [0, 1].

    cutoff     : fraction of Nyquist where the pass-band ends.
    shape_kind : "gaussian" (smooth) or "ideal" (hard cut).
    """
    rho = _radial_frequency_grid(shape)
    if shape_kind == "ideal":
        return (rho <= cutoff).astype(np.float32)
    # Gaussian: sigma == cutoff so that H(cutoff) ~ 0.61 (a natural knee point).
    return np.exp(-(rho ** 2) / (2.0 * (cutoff ** 2))).astype(np.float32)


def build_filter_mask(shape, filter_type, low_cutoff, high_cutoff, shape_kind):
    """Return the multiplicative transfer function for the chosen filter.

    The mask is fft-shifted (DC at the centre) to match ``apply_frequency_filter``.
    """
    if filter_type == "none":
        return np.ones(shape, dtype=np.float32)

    if filter_type == "lowpass":
        return _lowpass_mask(shape, low_cutoff, shape_kind)

    if filter_type == "highpass":
        # high-pass = 1 - low-pass(low_cutoff)
        return 1.0 - _lowpass_mask(shape, low_cutoff, shape_kind)

    if filter_type == "bandpass":
        assert high_cutoff > low_cutoff, "bandpass needs HIGH_CUTOFF > LOW_CUTOFF"
        # band-pass = low-pass(high) - low-pass(low) -> passes the [low, high] band.
        lp_high = _lowpass_mask(shape, high_cutoff, shape_kind)
        lp_low = _lowpass_mask(shape, low_cutoff, shape_kind)
        return np.clip(lp_high - lp_low, 0.0, 1.0)

    raise ValueError(f"Unknown FILTER_TYPE: {filter_type!r}")


def apply_frequency_filter(img, filter_type, low_cutoff, high_cutoff, shape_kind):
    """Apply the selected spatial-frequency filter to a 3D volume.

    Steps:
      1. FFT the volume and shift DC to the centre.
      2. Multiply by the transfer function (low/high/band-pass).
      3. Inverse-shift and inverse-FFT; keep the real part.

    Parameters
    ----------
    img : np.ndarray (D, H, W), real-valued.

    Returns
    -------
    filtered : np.ndarray (D, H, W), real-valued (same dtype family as input).
    """
    if filter_type == "none":
        return img.astype(np.float32)

    spectrum = np.fft.fftshift(np.fft.fftn(img))
    mask = build_filter_mask(
        img.shape, filter_type, low_cutoff, high_cutoff, shape_kind
    )
    filtered_spectrum = spectrum * mask
    filtered = np.fft.ifftn(np.fft.ifftshift(filtered_spectrum))
    return np.real(filtered).astype(np.float32)


# =====================================================================
#  GIN-style post-processing (blend with original + Frobenius match)
# =====================================================================

def _frobenius_match(volume, reference):
    """Scale ``volume`` so its Frobenius norm equals ``reference``'s.

    The Frobenius norm of a volume is just sqrt(sum of squared voxels) -- a
    single number measuring its total energy.  GIN forces its output to have
    the same total energy as the original input; we mirror that exactly here.
    Done in float64 for numerical stability, then cast back to float32.
    """
    ref_norm = float(np.sqrt(np.sum(reference.astype(np.float64) ** 2)))
    vol_norm = float(np.sqrt(np.sum(volume.astype(np.float64) ** 2)))
    if vol_norm < 1e-8:                       # avoid divide-by-zero on a ~empty volume
        return volume.astype(np.float32)
    return (volume * (ref_norm / vol_norm)).astype(np.float32)


def gin_style_blend(filtered, original, alpha):
    """Replicate GIN's post-processing on a filtered volume.

    Steps (mirrors gin_with_log_capability.py forward()):
      1. mixed = alpha * filtered + (1 - alpha) * original   (re-inject original).
      2. mixed = mixed * ||original||_F / ||mixed||_F        (Frobenius match).

    Parameters
    ----------
    filtered : np.ndarray (D, H, W) -- the frequency-filtered volume.
    original : np.ndarray (D, H, W) -- the normalised input the filter ran on.
    alpha    : float in [0, 1] -- weight on the filtered volume (0.5 = equal mix).
    """
    mixed = alpha * filtered + (1.0 - alpha) * original
    return _frobenius_match(mixed, original)


# =====================================================================
#  Segmentation metrics
# =====================================================================

def iou_and_dice(pred, gt, threshold=0.5):
    """Return (IoU, Dice) between a predicted and a ground-truth binary mask."""
    pred_bin = (pred > threshold)
    gt_bin = (gt > threshold)

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum()

    iou = intersection / union if union > 0 else 0.0
    dice = (2.0 * intersection) / denom if denom > 0 else 0.0
    return float(iou), float(dice)


# =====================================================================
#  Main
# =====================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[config] FILTER_TYPE={FILTER_TYPE}  shape={FILTER_SHAPE}  "
          f"low_cutoff={LOW_CUTOFF}  high_cutoff={HIGH_CUTOFF}")
    if GIN_STYLE_BLEND:
        print(f"[config] post-process=GIN_STYLE_BLEND  alpha={BLEND_ALPHA} "
              f"(blend + Frobenius match)")
    else:
        print(f"[config] post-process=renormalise={RENORMALISE_AFTER_FILTER}")
    print(f"[config] device={DEVICE}")
    print(f"[config] writing outputs to {OUTPUT_DIR}")

    # ---- model (identical load logic to eval_step_1.py) ----
    model = BasicUNet().to(DEVICE)
    if MODEL_CHECKPOINT:
        state_dict = torch.load(MODEL_CHECKPOINT, map_location=DEVICE.type)
        load_result = model.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys:
            print(f"[load_state_dict] missing keys ({len(load_result.missing_keys)}): "
                  f"{load_result.missing_keys[:8]}"
                  f"{' ...' if len(load_result.missing_keys) > 8 else ''}")
        if load_result.unexpected_keys:
            print(f"[load_state_dict] ignored unexpected keys "
                  f"({len(load_result.unexpected_keys)}): "
                  f"{load_result.unexpected_keys[:8]}"
                  f"{' ...' if len(load_result.unexpected_keys) > 8 else ''}")
    model.eval()

    # ---- pad + to-tensor transform (same as eval_step_1.py) ----
    transform = transforms.Compose([
        VariableSpatialFix(num_of_double_stride_conv=NUM_DOUBLE_STRIDE_CONV,
                           padval=PAD_VALUE),
        ToTensor(),
    ])

    all_files = pd.read_csv(INPUT_CSV)["filepaths"].values
    if MAX_SCANS is not None:
        all_files = all_files[:MAX_SCANS]

    # Per-scan metric rows for a CSV summary at the end.
    metric_rows = []

    for idx, each_file in tqdm.tqdm(list(enumerate(all_files))):
        arr = np.load(each_file)
        img_np = arr[:, :, :, 0]
        gt_np = normalise_zero_one(arr[:, :, :, 1])

        # UTE MRI is normalised straight to [-1, 1] (no HU clamp). This matches
        # the UTE branch used elsewhere in the codebase.
        img_np = normalise_one_one(img_np)

        # ---- STEP 1: filter the (normalised) volume. ----
        # We filter BEFORE padding so the constant pad border can't create a
        # fake high-frequency edge that contaminates the spectrum.
        filtered_np = apply_frequency_filter(
            img_np, FILTER_TYPE, LOW_CUTOFF, HIGH_CUTOFF, FILTER_SHAPE
        )

        # ---- STEP 2: post-process so the input is in-distribution. ----
        # Preferred: GIN-style blend (re-inject original + Frobenius match), which
        # is what the network actually saw during training.  Fallback: a plain
        # min-max re-stretch to [-1, 1].  "none" filter skips both.
        if FILTER_TYPE == "none":
            model_input_np = filtered_np
        elif GIN_STYLE_BLEND:
            model_input_np = gin_style_blend(filtered_np, img_np, BLEND_ALPHA)
        elif RENORMALISE_AFTER_FILTER:
            model_input_np = normalise_one_one(filtered_np)
        else:
            model_input_np = filtered_np

        h, w, d = model_input_np.shape

        # A filesystem-safe name built from the last path components.
        filename = "_".join(each_file.split("/")[-4:]).split(".npy")[0]

        # ---- save original / pure-filter / model-input for visual comparison ----
        # *_input_original  : the unfiltered normalised MRI.
        # *_filter_only     : the raw frequency-filter output (pre blend/renorm).
        # *_model_input     : what the network actually receives (post blend).
        generate_nifty(img_np.astype(np.float32), SPACING,
                       f"{OUTPUT_DIR}/{filename}_input_original.nii.gz")
        generate_nifty(filtered_np.astype(np.float32), SPACING,
                       f"{OUTPUT_DIR}/{filename}_filter_only_{FILTER_TYPE}.nii.gz")
        generate_nifty(model_input_np.astype(np.float32), SPACING,
                       f"{OUTPUT_DIR}/{filename}_model_input_{FILTER_TYPE}.nii.gz")

        # ---- run the segmenter on the post-processed input ----
        img_tensor = transform(model_input_np).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = torch.sigmoid(model(img_tensor))
            pred[pred > 0.5] = 1
            pred[pred <= 0.5] = 0

        # Crop the padding away before saving / scoring so shapes line up with GT.
        pred_np = pred.detach().squeeze().cpu().numpy().astype(np.uint8)[:h, :w, :d]

        generate_nifty(pred_np, SPACING, f"{OUTPUT_DIR}/{filename}_pred.nii.gz")
        generate_nifty(gt_np.astype(np.uint8), SPACING,
                       f"{OUTPUT_DIR}/{filename}_gt.nii.gz")

        iou, dice = iou_and_dice(pred_np, gt_np)
        metric_rows.append({"filename": filename, "iou": iou, "dice": dice})
        tqdm.tqdm.write(f"  {filename}: IoU={iou:.4f}  Dice={dice:.4f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- write + print a metric summary ----
    if metric_rows:
        df = pd.DataFrame(metric_rows)
        summary_csv = f"{OUTPUT_DIR}/_metrics_{FILTER_TYPE}.csv"
        df.to_csv(summary_csv, index=False)
        print(f"\n[summary] filter={FILTER_TYPE}  "
              f"mean IoU={df['iou'].mean():.4f}  mean Dice={df['dice'].mean():.4f}  "
              f"(n={len(df)})")
        print(f"[summary] per-scan metrics written to {summary_csv}")


if __name__ == "__main__":
    main()
