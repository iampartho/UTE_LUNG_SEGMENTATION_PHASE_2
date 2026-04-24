"""
save_lc_nifti.py

Creates a NIfTI image from voxel-level Local Complexity results.
Called by compute_local_complexity_unet.py when SAVE_LC_NIFTI = True.

Encoding in the output volume:
    0  -- voxel not in voxel_indices (background)
    1  -- total_lc == 0  (no activation changes)
    2  -- total_lc == 1  (one activation change)
    3  -- total_lc >  1  (multiple activation changes)

Metadata:
    spacing   : 1.25 x 1.25 x 1.25 mm (isotropic)
    origin    : (0, 0, 0)
    direction : identity
"""

import os
import numpy as np
import SimpleITK as sitk


def save_lc_nifti(result, scan_path, volume_shape, output_dir):
    """
    Parameters
    ----------
    result        : dict returned by UNetLocalComplexity.compute()
    scan_path     : str  -- original .npy path; used to derive the output filename
                           (same stem logic as the CSV writer)
    volume_shape  : tuple (D, H, W) -- shape of the ground-truth volume
    output_dir    : str  -- directory where the .nii.gz file is written
    """
    if "voxel_lc" not in result or len(result["voxel_lc"]) == 0:
        print("  [NIfTI] No voxel LC data in result -- skipping.")
        return

    vol = np.zeros(volume_shape, dtype=np.uint8)

    print("\n\n\nresult['voxel_lc']['per_layer_lc'].keys(): ", result["voxel_lc"][0]["per_layer_lc"].keys(), "\n\n\n")

    for vr in result["voxel_lc"]:
        d, h, w = vr["voxel"]
        lc = vr["total_lc"]#vr["per_layer_lc"]["enc_3_1"]#vr["total_lc"]
        if lc == 0:
            vol[d, h, w] = 1
        elif lc == 1:
            vol[d, h, w] = 2
        else:
            vol[d, h, w] = 3

    # SimpleITK reads numpy (D, H, W) and stores it as (W, H, D) internally,
    # which is standard NIfTI / ITK convention.
    def generate_nifty(arr_np, spacing, output_path):
        img_sitk = sitk.GetImageFromArray(arr_np)
        img_sitk.SetSpacing(spacing)
        img_sitk.SetOrigin((0.0, 0.0, 0.0))
        img_sitk.SetDirection((1.0, 0.0, 0.0,
                               0.0, 1.0, 0.0,
                               0.0, 0.0, 1.0))
        sitk.WriteImage(img_sitk, output_path)

    # Derive stem the same way log_results_to_csv does
    arr_imp = np.load(scan_path)[:,:,:,0]
    scan_name = '-'.join(scan_path.split('/')[1:])[:-4]

    out_path = os.path.join(output_dir, f"{scan_name}_roughness_enforced.nii.gz")
    out_path_image = os.path.join(output_dir, f"{scan_name}_image.nii.gz")
    os.makedirs(output_dir, exist_ok=True)
    generate_nifty(vol, (1.25, 1.25, 1.25), out_path)
    generate_nifty(arr_imp, (1.25, 1.25, 1.25), out_path_image)
    print(f"  [NIfTI] Saved -> {out_path}")
    print(f"  [NIfTI] Saved -> {out_path_image}")
