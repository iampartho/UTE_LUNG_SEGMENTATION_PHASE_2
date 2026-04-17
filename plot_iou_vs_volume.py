import os
import glob
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import argparse


def compute_iou(gt_arr, pred_arr):
    gt_bin = (gt_arr > 0).astype(np.uint8)
    pred_bin = (pred_arr > 0).astype(np.uint8)
    intersection = np.sum(gt_bin & pred_bin)
    union = np.sum(gt_bin | pred_bin)
    if union == 0:
        return 0.0
    return intersection / union


def compute_volume_ml(mask_arr, spacing):
    """Volume in millilitres (1 mL = 1000 mm^3)."""
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
    num_voxels = np.sum(mask_arr > 0)
    return (num_voxels * voxel_vol_mm3) / 1000.0


def main():
    parser = argparse.ArgumentParser(
        description="Plot IoU vs Lung Volume from prediction and ground truth NIfTI files."
    )
    parser.add_argument(
        "--pred_dir", type=str, default="./prediction/temp",
        help="Directory containing *_pred.nii.gz and *_gt.nii.gz files"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results_plots",
        help="Directory to save the output plot"
    )
    parser.add_argument(
        "--iou_low", type=float, default=0.5,
        help="Lower IoU threshold (inclusive). Only scans with IoU >= this value are plotted."
    )
    parser.add_argument(
        "--iou_high", type=float, default=0.85,
        help="Upper IoU threshold (inclusive). Only scans with IoU <= this value are plotted."
    )
    args = parser.parse_args()

    pred_dir = args.pred_dir
    output_dir = args.output_dir
    iou_low = args.iou_low
    iou_high = args.iou_high
    os.makedirs(output_dir, exist_ok=True)

    gt_files = sorted(glob.glob(os.path.join(pred_dir, "*_gt.nii.gz")))
    if not gt_files:
        print(f"No *_gt.nii.gz files found in {pred_dir}")
        return

    volumes = []
    ious = []
    labels = []

    for gt_path in gt_files:
        pred_path = gt_path.replace("_gt.nii.gz", "_pred.nii.gz")
        if not os.path.exists(pred_path):
            print(f"Prediction not found for {gt_path}, skipping.")
            continue

        gt_img = sitk.ReadImage(gt_path)
        pred_img = sitk.ReadImage(pred_path)

        spacing = gt_img.GetSpacing()

        gt_arr = sitk.GetArrayFromImage(gt_img)
        pred_arr = sitk.GetArrayFromImage(pred_img)

        iou = compute_iou(gt_arr, pred_arr)
        vol = compute_volume_ml(gt_arr, spacing)

        volumes.append(vol)
        ious.append(iou)
        labels.append(os.path.basename(gt_path).replace("_gt.nii.gz", ""))

        print(f"{labels[-1]:>40s}  |  Volume: {vol:8.1f} mL  |  IoU: {iou:.4f}")

    if not volumes:
        print("No valid prediction/ground-truth pairs found.")
        return

    volumes = np.array(volumes)
    ious = np.array(ious)

    mask = (ious >= iou_low) & (ious <= iou_high)
    volumes = volumes[mask]
    ious = ious[mask]
    labels = [l for l, m in zip(labels, mask) if m]

    print(f"\n{'='*60}")
    print(f"IoU filter: [{iou_low}, {iou_high}]")
    print(f"Scans after filtering: {len(volumes)}")
    if len(volumes) == 0:
        print("No scans in the given IoU range.")
        return
    print(f"Mean IoU:    {ious.mean():.4f}  (std {ious.std():.4f})")
    print(f"Volume range: {volumes.min():.1f} – {volumes.max():.1f} mL")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(12, 7))

    scatter = ax.scatter(
        volumes, ious,
        c=ious, cmap="RdYlGn", edgecolors="black",
        linewidths=0.6, s=70, alpha=0.85, vmin=0, vmax=1
    )

    z = np.polyfit(volumes, ious, 1)
    p = np.poly1d(z)
    x_line = np.linspace(volumes.min(), volumes.max(), 200)
    ax.plot(x_line, p(x_line), color="steelblue", linewidth=2, linestyle="--",
            label=f"Linear fit (slope={z[0]:.2e})")

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("IoU Score", fontsize=13)

    ax.set_xlabel("Lung Volume (mL)", fontsize=14)
    ax.set_ylabel("IoU Score", fontsize=14)
    ax.set_title("IoU vs Lung Volume", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_path = os.path.join(output_dir, "iou_vs_lung_volume_.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
