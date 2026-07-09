"""
temp_2.py — Generate GIN-augmented copies of every .npy scan in a directory.

For each input .npy:
  1. Load (handles 4D npy of shape (D, H, W, [img, gt]) by taking channel 0).
  2. Normalise to [-1, 1] (HU clipping first if SCAN_TYPE == "CT").
  3. Run GIN3D from `gin_with_log_capability.py` ``N_INSTANCES`` times.
  4. Tag each run with a per-layer roughness label, e.g. "L-L-H-H", using the
     same FFT-based criterion as `kernel_combination_distribution_for_gin.py`
     (L = low roughness, < threshold; H = high, >= threshold).
  5. Save augmented images as
        <stem>_[L-H-L-H].nii.gz
     Duplicate combos within the same scan get an underscore suffix
     (`_2`, `_3`, …) so nothing is overwritten.

Note: GIN3D inside `gin_with_log_capability.py` actively biases its random
kernels toward high (~65%) or low (~35%) roughness per layer, so the 8 runs
are *not* guaranteed to hit 8 distinct combos out of the 16 possible — that
is by design (matches the trained distribution).  If you want exactly one of
each combo, switch to the unbiased GIN3D in `causality_paper_augmentation_2.py`.
"""

import glob
import os

import SimpleITK as sitk
import numpy as np
import torch
from scipy.fftpack import fftn

from gin_with_log_capability import GIN3D


# =====================================================================
#  CONFIG  — edit these before running
# =====================================================================

INPUT_DIR   = "/Shared/lss_segerard/parthghosh/data/COPDgene_CT_1.25mm_numpy"
OUTPUT_DIR  = "./gin_augmented"

SCAN_TYPE   = "CT"               # "UTE" or "CT"  (CT applies HU clipping first)
N_INSTANCES = 8

# GIN3D architecture — must match what was used at training time.
GIN_KW = dict(
    in_channel=1,
    out_channel=1,
    interm_channel=4,
    scale_pool=[1, 2, 3],
    n_layer=4,
    out_norm="frob",
)

# Per-layer roughness threshold (matches kernel_combination_distribution_for_gin.py).
ROUGHNESS_THRESHOLD = 0.5

# Output NIfTI metadata (npy files have no header, so we set this manually).
OUTPUT_SPACING_MM = (1.0, 1.0, 1.0)   # SimpleITK order: (x, y, z)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0                              # set to None for non-deterministic runs


# =====================================================================
#  Image utilities
# =====================================================================

def normalise_zero_one(x):
    mn, mx = float(x.min()), float(x.max())
    if mx > mn:
        return (x - mn) / (mx - mn)
    return x * 0.0


def normalise_one_one(x):
    return normalise_zero_one(x) * 2.0 - 1.0


def normalise_hu(x, lo=-1024.0, hi=600.0):
    return np.clip(x, lo, hi)


def load_image(path):
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 4:
        # Convention in this repo: (D, H, W, [img, gt]) — keep image only.
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(
            f"{path} has shape {arr.shape}; expected 3D or 4D-with-2-channels."
        )
    return arr


def save_nifti(arr_zyx, out_path, spacing_xyz=(1.0, 1.0, 1.0)):
    img = sitk.GetImageFromArray(arr_zyx.astype(np.float32))   # expects (z, y, x)
    img.SetSpacing(tuple(float(s) for s in spacing_xyz))
    sitk.WriteImage(img, out_path, useCompression=True)


# =====================================================================
#  Roughness classification (mirrors kernel_combination_distribution_for_gin.py)
# =====================================================================

def _channels_for_layer(layer_idx, n_layer, in_ch, out_ch, interm):
    """Number of (Out * In) channels per layer in GIN3D."""
    if layer_idx == 0:
        return interm * in_ch                # input layer
    if layer_idx == n_layer - 1:
        return out_ch * interm               # output layer
    return interm * interm                   # hidden layer


def compute_roughness(kernel_tensor, layer_idx):
    """High-frequency energy ratio of a stored kernel.

    `kernel_tensor` is the per-layer kernel cached by GIN3D — shape
    [Batch*Out, In, k, k, k].  Mean over the (Out, In) axes gives a single
    representative spatial kernel of shape (k, k, k); the 3D FFT magnitude
    is then split into DC vs. AC energy.

    Returns a scalar in [0, 1):
        ~0.0 → smooth (low frequency dominates)
        ~1.0 → rough  (high frequency dominates)
    """
    flat = kernel_tensor.detach().cpu().numpy().reshape(-1)
    channels = _channels_for_layer(
        layer_idx,
        n_layer=GIN_KW["n_layer"],
        in_ch=GIN_KW["in_channel"],
        out_ch=GIN_KW["out_channel"],
        interm=GIN_KW["interm_channel"],
    )
    n_params = flat.size
    vol = n_params / channels
    k = int(round(vol ** (1.0 / 3.0)))
    if k < 2:
        return 0.0

    w = flat.reshape(channels, k, k, k)
    w_spatial = w.mean(axis=0)
    fft_mag = np.abs(fftn(w_spatial))
    total = float(fft_mag.sum())
    dc = float(fft_mag[0, 0, 0])
    return 1.0 - dc / (total + 1e-9)


def classify_roughness(r, threshold=ROUGHNESS_THRESHOLD):
    return "L" if r < threshold else "H"


# =====================================================================
#  Main
# =====================================================================

@torch.no_grad()
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if SEED is not None:
        torch.manual_seed(SEED)
        np.random.seed(SEED)

    paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.npy")))[0:1]
    print(f"Found {len(paths)} .npy files in {INPUT_DIR}")

    gin = GIN3D(**GIN_KW).to(DEVICE)
    gin.eval()

    for i, p in enumerate(paths, 1):
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            img_np = load_image(p)
        except Exception as e:                                # noqa: BLE001
            print(f"[{i}/{len(paths)}] FAILED to load {stem}: {e}")
            continue

        if SCAN_TYPE.upper() == "CT":
            img_np = normalise_one_one(normalise_hu(img_np))
        else:
            img_np = normalise_one_one(img_np)

        x = (
            torch.from_numpy(img_np)
            .unsqueeze(0)        # (1, D, H, W)
            .unsqueeze(0)        # (1, 1, D, H, W)
            .to(DEVICE)
        )

        combo_count = {}
        for _ in range(N_INSTANCES):
            x_aug = gin(x)                                    # (1, 1, D, H, W)

            classes = [
                classify_roughness(compute_roughness(layer["w"], layer["idx"]))
                for layer in gin.get_layer_params()
            ]
            combo = "-".join(classes)

            seen = combo_count.get(combo, 0) + 1
            combo_count[combo] = seen
            suffix = "" if seen == 1 else f"_{seen}"

            out_name = f"{stem}_[{combo}]{suffix}.nii.gz"
            out_path = os.path.join(OUTPUT_DIR, out_name)

            arr_out = x_aug.squeeze(0).squeeze(0).cpu().numpy()  # (D, H, W)
            save_nifti(arr_out, out_path, OUTPUT_SPACING_MM)

        combo_summary = ", ".join(f"{k}x{v}" for k, v in combo_count.items())
        print(f"[{i}/{len(paths)}] {stem}: {N_INSTANCES} variants  ({combo_summary})")

    print(f"\nDone. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
