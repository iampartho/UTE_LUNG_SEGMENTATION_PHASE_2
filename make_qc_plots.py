#!/usr/bin/env python
"""
QC plotting for MRI -> synthetic-CT lung segmentation.

For every subdirectory of `nifti_to_send/` this produces one `<folder_name>.png`
with three rows:

  Row 1 : mid axial / coronal / sagittal slice of the MRI (original UTE) image
  Row 2 : the same three slice positions of the synthetic (fake) CT image
  Row 3 : 3D surface renderings of the lung masks -
            ground truth | prediction on MRI | prediction on synthetic CT

The `original_cyclegan_validation` folder has no MRI-prediction, so its mask
row shows only two columns (ground truth | prediction on synthetic CT).
"""

import os
import numpy as np
import nibabel as nib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nifti_to_send")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qc_plots")

# candidate filenames (first existing one wins)
MRI_NAMES = ["original_ute.nii.gz"]
CT_NAMES = ["fake_ct.nii.gz"]
GT_NAMES = ["original_ute_gt.nii.gz", "ute_gt.nii.gz", "original_gt.nii.gz"]
PRED_MRI_NAMES = ["model_pred_on_ute.nii.gz"]
PRED_CT_NAMES = ["model_pred_on_fake_ct.nii.gz"]

MASK_COLORS = {
    "Ground truth": "#2ca02c",       # green
    "Pred on MRI": "#1f77b4",        # blue
    "Pred on synth CT": "#ff7f0e",   # orange
}


def find_file(folder, names):
    for n in names:
        p = os.path.join(folder, n)
        if os.path.exists(p):
            return p
    return None


def load_canonical(path):
    """Load a NIfTI, reorient to closest-canonical (RAS) so axis 0/1/2 map to
    sagittal/coronal/axial consistently. Returns (array, zooms)."""
    img = nib.as_closest_canonical(nib.load(path))
    return np.asarray(img.get_fdata(), dtype=np.float32), img.header.get_zooms()[:3]


def window(vol):
    """Robust intensity window (1st-99th percentile) for display."""
    lo, hi = np.percentile(vol, (1, 99))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def show_slice(ax, vol, axis, idx, lo, hi, title):
    """Display one 2D slice, oriented superior/anterior up."""
    if axis == 2:      # axial: fix z
        sl = vol[:, :, idx]
    elif axis == 1:    # coronal: fix y
        sl = vol[:, idx, :]
    else:              # sagittal: fix x
        sl = vol[idx, :, :]
    ax.imshow(sl.T, cmap="gray", origin="lower", vmin=lo, vmax=hi, aspect="equal")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def render_mask_3d(ax, mask, zooms, title, color):
    """Render a binary mask as a 3D surface via marching cubes."""
    ax.set_title(title, fontsize=10)
    m = mask > 0.5
    if m.sum() < 10:
        ax.text2D(0.5, 0.5, "empty mask", ha="center", va="center",
                  transform=ax.transAxes, fontsize=9)
        ax.set_axis_off()
        return

    # downsample x2 for speed; scale spacing accordingly
    m_ds = m[::2, ::2, ::2]
    spacing = tuple(float(z) * 2 for z in zooms)
    try:
        verts, faces, _, _ = measure.marching_cubes(
            m_ds.astype(np.float32), level=0.5, spacing=spacing, step_size=1
        )
    except (RuntimeError, ValueError):
        ax.text2D(0.5, 0.5, "no surface", ha="center", va="center",
                  transform=ax.transAxes, fontsize=9)
        ax.set_axis_off()
        return

    mesh = Poly3DCollection(verts[faces], alpha=1.0)
    mesh.set_facecolor(color)
    mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)

    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())
    ax.set_box_aspect((np.ptp(verts[:, 0]), np.ptp(verts[:, 1]), np.ptp(verts[:, 2])))
    ax.view_init(elev=15, azim=-70)
    ax.set_axis_off()


def process_folder(folder, name):
    mri_p = find_file(folder, MRI_NAMES)
    ct_p = find_file(folder, CT_NAMES)
    gt_p = find_file(folder, GT_NAMES)
    pred_mri_p = find_file(folder, PRED_MRI_NAMES)
    pred_ct_p = find_file(folder, PRED_CT_NAMES)

    if mri_p is None or ct_p is None:
        print(f"[skip] {name}: missing MRI or CT image")
        return

    mri, zooms = load_canonical(mri_p)
    ct, _ = load_canonical(ct_p)

    # mid slice positions (geometric center of the volume)
    centers = [s // 2 for s in mri.shape]
    planes = [(2, "Axial"), (1, "Coronal"), (0, "Sagittal")]

    mri_lo, mri_hi = window(mri)
    ct_lo, ct_hi = window(ct)

    # assemble mask column list
    mask_cols = []
    if gt_p is not None:
        mask_cols.append(("Ground truth", gt_p))
    if pred_mri_p is not None:
        mask_cols.append(("Pred on MRI", pred_mri_p))
    if pred_ct_p is not None:
        mask_cols.append(("Pred on synth CT", pred_ct_p))

    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 3)

    # Row 1: MRI slices
    for c, (axis, pname) in enumerate(planes):
        ax = fig.add_subplot(gs[0, c])
        show_slice(ax, mri, axis, centers[axis], mri_lo, mri_hi,
                   f"MRI - {pname} (idx {centers[axis]})")

    # Row 2: synthetic CT slices at same positions
    for c, (axis, pname) in enumerate(planes):
        ax = fig.add_subplot(gs[1, c])
        show_slice(ax, ct, axis, centers[axis], ct_lo, ct_hi,
                   f"Synth CT - {pname} (idx {centers[axis]})")

    # Row 3: 3D mask renderings
    for c, (label, path) in enumerate(mask_cols):
        ax = fig.add_subplot(gs[2, c], projection="3d")
        mask, mzooms = load_canonical(path)
        render_mask_3d(ax, mask, mzooms, f"3D mask - {label}",
                       MASK_COLORS.get(label, "#7f7f7f"))

    fig.suptitle(name, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok]   {name}: {len(mask_cols)} mask columns -> {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    subdirs = sorted(
        d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))
    )
    for name in subdirs:
        process_folder(os.path.join(ROOT, name), name)


if __name__ == "__main__":
    main()
