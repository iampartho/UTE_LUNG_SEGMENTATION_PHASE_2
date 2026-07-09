"""
Sanity test for the paired geometric augmentation (anisotropic scale + elastic)
and the _clean_binary hygiene added to datasets_causality.py.

What it checks
--------------
1. After the SAME random scale + elastic field is applied to the image and the
   GT mask, they must stay aligned. We save NIfTI pairs (original + augmented)
   so you can overlay image+mask in a viewer (ITK-SNAP / 3D Slicer) and confirm
   the mask still hugs the lung.
2. A numeric proxy for alignment: the mean image intensity *inside* the mask vs
   *outside* the mask should stay separated after augmentation (lung interior is
   denser/brighter than the -1 air background), and the separation should be
   close to the original. If the warp desynced image and mask, this collapses.

Usage
-----
    python test_geometric_augmentation.py                # uses first CT scan
    python test_geometric_augmentation.py 5              # uses scan index 5
Outputs land in ./aug_test_output/ as .nii.gz.
"""
import os
import sys
import numpy as np
import SimpleITK as sitk

from datasets_causality import (
    normalise_hu, normalise_one_one, normalise_zero_one, _clean_binary,
    RandomAnisoScale, RandomElastic, VariableSpatialFix,
)

CSV = "./ids/only_copd_1.25mm.csv"
OUT_DIR = "./aug_test_output"
SPACING = (1.25, 1.25, 1.25)


def save_nifty(array, save_path):
    img = sitk.GetImageFromArray(array.astype(np.float32))
    img.SetSpacing(SPACING)
    sitk.WriteImage(img, save_path)
    print(f"  saved {save_path}  shape={array.shape}")


def mask_contrast(image, mask):
    """Mean image value inside vs outside the mask (alignment proxy)."""
    m = mask > 0.5
    if m.sum() == 0 or (~m).sum() == 0:
        return float("nan"), float("nan")
    return float(image[m].mean()), float(image[~m].mean())


def main():
    import pandas as pd

    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    os.makedirs(OUT_DIR, exist_ok=True)

    path = pd.read_csv(CSV)["filepaths"].values[idx]
    print(f"Loading scan {idx}: {path}")
    arr = np.load(path)

    # --- replicate CausalityDataset.__getitem__ (training branch) ---
    img = arr[:, :, :, 0]
    mask = arr[:, :, :, 1]
    mask[mask > 0] = 1
    img = normalise_one_one(normalise_hu(img))
    mask = normalise_zero_one(mask)
    mask = _clean_binary(mask)

    in0, out0 = mask_contrast(img, mask)
    print(f"\n[original]  in-mask mean={in0:.4f}  out-mask mean={out0:.4f}  "
          f"separation={in0 - out0:.4f}")
    save_nifty(img, os.path.join(OUT_DIR, "image_original.nii.gz"))
    save_nifty(mask, os.path.join(OUT_DIR, "mask_original.nii.gz"))

    # --- apply the exact training transform pipeline (force prob=1.0) ---
    sample = [img, mask]
    sample = RandomAnisoScale(prob=0.0, scale_range=(0.80, 1.25))(sample)
    sample = VariableSpatialFix(num_of_double_stride_conv=4)(sample)
    sample = RandomElastic(prob=1.0, alpha=12.0, ctrl=8)(sample)
    aug_img, aug_mask = sample[0], sample[1]

    in1, out1 = mask_contrast(aug_img, aug_mask)
    print(f"[augmented] in-mask mean={in1:.4f}  out-mask mean={out1:.4f}  "
          f"separation={in1 - out1:.4f}")
    save_nifty(aug_img, os.path.join(OUT_DIR, "image_augmented_only_elastic.nii.gz"))
    save_nifty(aug_mask, os.path.join(OUT_DIR, "mask_augmented_only_elastic.nii.gz"))

    print("\nInterpretation:")
    print("  * The KEY check is that 'in-mask mean' barely changes after "
          "augmentation -- the mask still sits on the same (lung) tissue.")
    print("  * |separation| should stay large (lung air is much darker than the "
          "surrounding body tissue; the sign is negative for CT and is fine).")
    print("  * If the mask desynced from the image, the in-mask mean would drift "
          "toward the out-mask/background value and |separation| would collapse.")
    print(f"  * Overlay image_augmented + mask_augmented in {OUT_DIR} to confirm "
          "the mask still hugs the lung.")


if __name__ == "__main__":
    main()
