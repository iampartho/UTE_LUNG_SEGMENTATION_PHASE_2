"""
temp_3.py

Loads a .npy scan, builds the 6-vertex orthogonal cross-polytopal hull
(3 orthogonal directions x +/- perturbation, same method as
compute_local_complexity_unet.py), and saves:
  - the mid-coronal PNG of the original (normalised) scan
  - the mid-coronal PNG for each of the 6 hull vertices
  - optionally a NIfTI for each as well (SAVE_NIFTI flag)

Coronal axis convention: dimension 1 (H axis) of the (D, H, W) volume.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import SimpleITK as sitk
from torchvision import transforms

from eval_step_1 import (
    normalise_one_one, normalise_zero_one, normalise_hu,
    VariableSpatialFix, ToTensor,
)

# =====================================================================
#  CONFIGURATION
# =====================================================================

NPY_PATH   = "/Shared/lss_segerard/parthghosh/data/UTE_new_data_numpy/103-035/20180424/AnatCorrLungs.npy"
SCAN_TYPE  = "UTE"          # "UTE" or "CT"

OUTPUT_DIR = "./hull_slices"
SPACING    = (1.25, 1.25, 1.25)

RADIUS     = 0.005          # cross-polytope radius (same as compute_local_complexity_unet.py)
N_HULL     = 6              # must be even; gives N_HULL/2 = 3 orthogonal directions
HULL_SEED  = 42

SAVE_NIFTI = False          # also save .nii.gz for each volume

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================================
#  HELPERS
# =====================================================================

def _normalise(img: np.ndarray, scan_type: str) -> np.ndarray:
    if scan_type == "UTE":
        return normalise_one_one(img)
    else:
        return normalise_one_one(normalise_hu(img))


def save_mid_coronal_png(volume: np.ndarray, output_path: str, title: str = ""):
    """Save the mid-coronal slice of a (D, H, W) volume as a PNG."""
    mid = volume.shape[1] // 2
    sl = volume[:, mid, :]
    sl = np.rot90(sl.T, k=3)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(sl, cmap="gray", origin="lower")
    if title:
        ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close(fig)


def save_nifti(arr: np.ndarray, output_path: str, spacing=(1.25, 1.25, 1.25)):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1.0, 0.0, 0.0,
                      0.0, 1.0, 0.0,
                      0.0, 0.0, 1.0))
    sitk.WriteImage(img, output_path)


@torch.no_grad()
def get_ortho_hull_3d(x, r=0.005, n=6, seed=42):
    """
    Cross-polytopal hull vertices around a single 3D volume.

    x    : (1, C, D, H, W) tensor
    r    : radius
    n    : number of vertices (must be even; n/2 orthogonal directions)
    seed : reproducibility seed

    Returns (n, C, D, H, W) tensor -- the hull vertices.
    """
    assert n % 2 == 0, "N_HULL must be even"
    if seed is not None:
        torch.manual_seed(seed)

    flat_dim = int(np.prod(x.shape[1:]))
    n_dirs   = n // 2

    orth = torch.nn.utils.parametrizations.orthogonal(
        torch.nn.Linear(flat_dim, n_dirs).to(x.device),
        use_trivialization=False,
    )
    dirs   = orth.weight * r                             # (n_dirs, flat_dim)
    x_flat = x.reshape(1, -1)                            # (1, flat_dim)
    hull   = torch.cat([x_flat + dirs,
                        x_flat - dirs], dim=0)           # (n, flat_dim)
    return hull.reshape(n, *x.shape[1:])


# =====================================================================
#  MAIN
# =====================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    basename = os.path.splitext(os.path.basename(NPY_PATH))[0]

    # ------------------------------------------------------------------
    # 1. Load and normalise the scan
    # ------------------------------------------------------------------
    arr    = np.load(NPY_PATH).astype(np.float32)
    img_np = arr[:, :, :, 0]
    img_np = _normalise(img_np, SCAN_TYPE)

    # ------------------------------------------------------------------
    # 2. Apply the same spatial transform used during inference
    # ------------------------------------------------------------------
    tfm = transforms.Compose([
        VariableSpatialFix(num_of_double_stride_conv=4, padval=0.5),
        ToTensor(),
    ])
    scan_tensor = tfm(img_np).unsqueeze(0).to(DEVICE)  # (1, 1, D, H, W)

    print(f"Scan tensor shape : {tuple(scan_tensor.shape)}")
    print(f"Generating {N_HULL} hull vertices  (radius={RADIUS}, seed={HULL_SEED}) ...")

    # ------------------------------------------------------------------
    # 3. Save original scan mid-coronal slice
    # ------------------------------------------------------------------
    orig_np   = scan_tensor.squeeze().cpu().numpy()     # (D, H, W)
    orig_png  = os.path.join(OUTPUT_DIR, f"{basename}_original_midcoronal.png")
    save_mid_coronal_png(orig_np, orig_png, title="Original")
    print(f"  Saved original  -> {orig_png}")

    if SAVE_NIFTI:
        orig_nii = os.path.join(OUTPUT_DIR, f"{basename}_original.nii.gz")
        save_nifti(orig_np, orig_nii, SPACING)
        print(f"  Saved NIfTI     -> {orig_nii}")

    # ------------------------------------------------------------------
    # 4. Generate hull vertices and save mid-coronal slices
    # ------------------------------------------------------------------
    hull = get_ortho_hull_3d(scan_tensor, r=RADIUS, n=N_HULL, seed=HULL_SEED)
    # hull shape: (N_HULL, 1, D, H, W)

    n_dirs = N_HULL // 2
    for i in range(N_HULL):
        # Readable label: dir1+, dir1-, dir2+, dir2-, dir3+, dir3-
        dir_idx  = (i % n_dirs) + 1
        polarity = "pos" if i < n_dirs else "neg"
        tag      = f"hull_dir{dir_idx}_{polarity}"

        vol_np  = hull[i].squeeze().cpu().numpy()           # (D, H, W)
        png_out = os.path.join(OUTPUT_DIR, f"{basename}_{tag}_midcoronal.png")
        save_mid_coronal_png(vol_np, png_out, title=tag)
        print(f"  Saved hull[{i}]  {tag:20s} -> {png_out}")

        if SAVE_NIFTI:
            nii_out = os.path.join(OUTPUT_DIR, f"{basename}_{tag}.nii.gz")
            save_nifti(vol_np, nii_out, SPACING)
            print(f"  Saved NIfTI     -> {nii_out}")

    print(f"\nDone. {1 + N_HULL} images saved to: {os.path.abspath(OUTPUT_DIR)}/")


if __name__ == "__main__":
    main()
