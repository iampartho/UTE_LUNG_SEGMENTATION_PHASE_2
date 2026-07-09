"""
Evaluate a model that was trained on 1.25mm data and write 1mm NIfTI outputs.

Two modes (controlled by GT_1MM_DIR):

  Eval mode  (GT_1MM_DIR set):
    - Load 1mm GT NIfTI (matched by stem to the 1.25mm npy).
    - Use the GT NIfTI as the reference for shape + spacing/origin/direction.
    - Save: prediction (resampled), GT (binarized), confusion mask.

  Predict-only mode  (GT_1MM_DIR == ''):
    - Requires IMG_1MM_DIR pointing to a folder of 1mm reference image NIfTIs.
    - Use the 1mm image NIfTI as the reference for shape + metadata.
    - Save: prediction only (no GT, no confusion mask).

Pipeline per scan:
    1. Load image from the 1.25mm CSV (npy stack[:, :, :, 0]).
    2. Run the model at 1.25mm.
    3. Wrap the prediction as a SimpleITK image with spacing 1.25mm and
       resample it to 1mm using preprocess_COPDgene_CT.resample_iso
       (nearest-neighbor, since it's a binary mask).
    4. Match shape and spatial metadata to the 1mm reference NIfTI
       (GT in eval mode; image in predict-only mode).
    5. Save 1mm NIfTI outputs as described above.
"""

import os
import glob
import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk
from torchvision import transforms
import tqdm

from basic_unet_disentangled import BasicUNet
from preprocess_COPDgene_CT import resample_iso
from eval_step_1 import (
    normalise_one_one,
    generate_confusion_mask,
    VariableSpatialFix,
    ToTensor,
)


# ============================ Configuration ============================
CSV_125MM   = './ids/UTE_MRI_OECLAD_ORIGINAL_SPACING_resamples_1.25mm_only_img.csv'
# 1mm GT NIfTI folder. Leave as '' to skip GT/confusion and only write predictions.
GT_1MM_DIR  = '/Shared/lss_segerard/data/UTE_MRI_OECLAD/MASKS_better'
# Required when GT_1MM_DIR == ''. 1mm image NIfTI folder used as the
# shape + spacing/origin/direction reference for the saved prediction.
IMG_1MM_DIR = ''

CHECKPOINT_PATH = './save_models/current_best_model_june_2026/best_bunet_td1_lc_monitor_70_RE_80_GIN_instance_norm_new_aug_0.065.pth' #best_bunet_causality_paper_ct_train_UTE_test_w_tversky_wo_kl_only_gin_roughness_enforced_5_normalised_gin.pth' #best_train_bunet_joint_train_with_gin_on_ute_with_logging_roughness_enforced.pth'
OUTPUT_DIR = './prediction/temp'

INPUT_SPACING_MM  = (1.25, 1.25, 1.25)   # spacing of the model-fed npy stack
OUTPUT_SPACING_MM = (1.0, 1.0, 1.0)      # fallback spacing if metadata is missing

NIFTY_EXTS = ('.nii.gz', '.nii')
GT_MASK_SUFFIX = '_mask'

NUM_DOUBLE_STRIDE_CONV = 4
PAD_VAL = 0.5
THRESHOLD = 0.5

# Largest-connected-component (LCC) post-processing.
# When enabled, keep only the top-N largest connected foreground components
# in the resampled 1mm prediction (helps remove disconnected false positives).
KEEP_LARGEST_CC = True
NUM_LARGEST_CC = 2                   # how many largest components to keep
LCC_FULL_CONNECTIVITY = True         # 26-connectivity in 3D (False -> 6-conn)


# ============================ Helpers ============================
def _stem_npy(p):
    return os.path.basename(p).split('.npy')[0]


def _stem_nifty(p):
    """Return the GT NIfTI stem matching the corresponding npy basename.

    Strips the NIfTI extension and any trailing `_mask` suffix, so e.g.
    `patient_001_mask.nii.gz`  ->  `patient_001`.
    """
    base = os.path.basename(p)
    for ext in NIFTY_EXTS:
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    else:
        base = os.path.splitext(base)[0]

    if base.endswith(GT_MASK_SUFFIX):
        base = base[: -len(GT_MASK_SUFFIX)]
    return base


def build_pair_table(csv_125_path, ref_dir, ref_col_name):
    """Pair 1.25mm CSV rows with NIfTI files in `ref_dir` by stem.

    `ref_col_name` is the merged-table column name for the NIfTI path
    (e.g. 'gt_path' in eval mode, 'img_path' in predict-only mode).
    """
    df_125 = pd.read_csv(csv_125_path)
    df_125['__stem__'] = df_125['filepaths'].apply(_stem_npy)

    ref_files = []
    for ext in NIFTY_EXTS:
        ref_files.extend(glob.glob(os.path.join(ref_dir, f'*{ext}')))

    if not ref_files:
        raise ValueError(f"No NIfTI files found in {ref_dir}")

    ref_df = pd.DataFrame({ref_col_name: ref_files})
    ref_df['__stem__'] = ref_df[ref_col_name].apply(_stem_nifty)

    merged = df_125.merge(ref_df, on='__stem__')
    if merged.empty:
        raise ValueError(
            f"No matching basenames between the 1.25mm CSV and {ref_dir}. "
            "Make sure 1.25mm npy filenames and 1mm NIfTI filenames share stems."
        )
    return merged


def predict_at_125mm(model, image_np, transform, device):
    """Run model on 1.25mm image, return binary uint8 prediction cropped to
    the original input shape (undoing VariableSpatialFix padding)."""
    h, w, d = image_np.shape
    img_norm = normalise_one_one(image_np)

    img_tensor = transform(img_norm).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        prob = torch.sigmoid(logits)
        binary = (prob > THRESHOLD).float()

    pred_np = binary.detach().squeeze().cpu().numpy().astype(np.uint8)
    return pred_np[:h, :w, :d]


def numpy_to_sitk(arr_np, spacing):
    img = sitk.GetImageFromArray(arr_np)
    img.SetSpacing(tuple(float(s) for s in spacing))
    return img


def resample_pred_to_1mm(pred_125_np, in_spacing, out_spacing):
    """Resample a 1.25mm binary prediction to 1mm using nearest neighbor."""
    if not all(s == out_spacing[0] for s in out_spacing):
        raise ValueError("resample_iso only supports isotropic output spacing.")
    pred_125_sitk = numpy_to_sitk(pred_125_np, in_spacing)
    pred_1mm_sitk = resample_iso(
        pred_125_sitk,
        out_spacing[0],
        0,
        sitk.sitkNearestNeighbor,
    )
    return sitk.GetArrayFromImage(pred_1mm_sitk).astype(np.uint8)


def match_shape(arr, target_shape, fill_value=0):
    """Crop/zero-pad `arr` per-axis so its shape matches target_shape exactly."""
    a = arr
    for axis, (cur, tgt) in enumerate(zip(a.shape, target_shape)):
        if cur == tgt:
            continue
        if cur > tgt:
            slicer = [slice(None)] * a.ndim
            slicer[axis] = slice(0, tgt)
            a = a[tuple(slicer)]
        else:
            pad = [(0, 0)] * a.ndim
            pad[axis] = (0, tgt - cur)
            a = np.pad(a, pad, mode='constant', constant_values=fill_value)
    return a


def save_with_reference(arr_np, reference_img, output_path):
    """Write `arr_np` as a NIfTI sharing spacing / origin / direction with
    `reference_img` (a SimpleITK image)."""
    out_img = sitk.GetImageFromArray(arr_np)
    out_img.CopyInformation(reference_img)
    sitk.WriteImage(out_img, output_path)


def keep_largest_connected_components(arr_np, k=1, full_connectivity=True):
    """Keep the `k` largest connected foreground components of a binary mask.

    Parameters
    ----------
    arr_np : np.ndarray
        Binary mask (any integer/bool dtype). 0 = background, non-zero = foreground.
    k : int
        Number of largest components to keep. If <= 0, returns the input unchanged.
    full_connectivity : bool
        If True, uses full (26-)connectivity in 3D; otherwise face (6-)connectivity.

    Returns
    -------
    np.ndarray of np.uint8 with the same shape as `arr_np`.
    """
    if k is None or k <= 0:
        return arr_np.astype(np.uint8)
    if arr_np.sum() == 0:
        return arr_np.astype(np.uint8)

    binary = (arr_np > 0).astype(np.uint8)
    sitk_img = sitk.GetImageFromArray(binary)

    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(bool(full_connectivity))
    labels = cc_filter.Execute(sitk_img)

    # RelabelComponent sorts by size descending: label 1 = largest, 2 = second, ...
    relabeled = sitk.RelabelComponent(labels, sortByObjectSize=True)
    relabeled_np = sitk.GetArrayFromImage(relabeled)

    keep_mask = (relabeled_np >= 1) & (relabeled_np <= int(k))
    return keep_mask.astype(np.uint8)


def canonicalize_flip_axes(reference_img):
    """Return per-axis flip flags (x, y, z) that `preprocess_COPDgene_CT.py`
    would have applied to canonicalize this image's orientation.

    Mirrors the logic in `preprocess_COPDgene_CT.py` (lines 98-104):
        dirs = img.GetDirection()
        dirs_diag = np.sign([dirs[0], dirs[4], dirs[8]])
        flips = [False if x == 1 else True for x in dirs_diag]
    """
    dirs = reference_img.GetDirection()
    dirs_diag = np.sign(np.array([dirs[0], dirs[4], dirs[8]]))
    return [bool(x != 1) for x in dirs_diag]


def restore_original_orientation(arr_np, flips_xyz):
    """Reverse the canonicalization flip applied by preprocessing so that
    `arr_np` lines up with the reference NIfTI's original direction matrix.

    The model is fed an array that was canonicalized in preprocessing via
    `sitk.Flip`, so its prediction is in canonical orientation. The reference
    NIfTI used for saving still has the original direction (possibly with
    negative diagonal entries); without this step `CopyInformation` would
    stamp those negatives onto a canonical-orientation array, producing a
    prediction that appears flipped relative to the original image / GT.

    SimpleITK arrays are in (z, y, x) numpy order, so SimpleITK's per-axis
    flip flags (x, y, z) map to numpy axes (2, 1, 0).
    """
    out = arr_np
    if flips_xyz[0]:
        out = np.flip(out, axis=2)
    if flips_xyz[1]:
        out = np.flip(out, axis=1)
    if flips_xyz[2]:
        out = np.flip(out, axis=0)
    return np.ascontiguousarray(out)


# ============================ Main ============================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    gt_mode = bool(GT_1MM_DIR)
    if gt_mode:
        ref_dir = GT_1MM_DIR
        ref_col = 'gt_path'
        mode_label = 'GT (eval) mode'
    else:
        if not IMG_1MM_DIR:
            raise ValueError(
                "GT_1MM_DIR is empty -> IMG_1MM_DIR must be set to a folder of "
                "1mm reference image NIfTIs (used for shape and metadata)."
            )
        ref_dir = IMG_1MM_DIR
        ref_col = 'img_path'
        mode_label = 'predict-only mode (no GT)'

    pair_df = build_pair_table(CSV_125MM, ref_dir, ref_col)
    print(f"[eval] {mode_label}: {len(pair_df)} scans matched between "
          f"1.25mm CSV and 1mm reference folder ({ref_dir}).")
    print("The head of the pair_df is:")
    print(pair_df.head())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BasicUNet().to(device)
    if CHECKPOINT_PATH:
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device.type)
        # Older checkpoints (from the disentangled BasicUNet variant) carry an
        # extra parallel set of `*_ct` normalisation layers that the current
        # `basic_unet_disentangled_no_normalising.BasicUNet` no longer defines.
        # `strict=False` lets us load every matching tensor and just drop the
        # unused `_ct` ones. We log what was skipped so silent drift is visible.
        load_result = model.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys:
            print(f"[load_state_dict] missing keys ({len(load_result.missing_keys)}): "
                  f"{load_result.missing_keys[:8]}{' ...' if len(load_result.missing_keys) > 8 else ''}")
        if load_result.unexpected_keys:
            print(f"[load_state_dict] ignored unexpected keys ({len(load_result.unexpected_keys)}): "
                  f"{load_result.unexpected_keys[:8]}{' ...' if len(load_result.unexpected_keys) > 8 else ''}")
    model.eval()

    transform_125 = transforms.Compose([
        VariableSpatialFix(num_of_double_stride_conv=NUM_DOUBLE_STRIDE_CONV, padval=PAD_VAL),
        ToTensor(),
    ])

    for _, row in tqdm.tqdm(pair_df.iterrows(), total=len(pair_df)):
        path_125 = row['filepaths']
        ref_path = row[ref_col]
        stem     = row['__stem__']

        arr_125 = np.load(path_125)
        img_125_np = arr_125[:, :, :, 0] if arr_125.ndim == 4 else arr_125

        pred_125_np = predict_at_125mm(model, img_125_np, transform_125, device)
        pred_1mm_np = resample_pred_to_1mm(pred_125_np, INPUT_SPACING_MM, OUTPUT_SPACING_MM)

        ref_img_sitk = sitk.ReadImage(ref_path)
        ref_arr = sitk.GetArrayFromImage(ref_img_sitk)

        if gt_mode:
            gt_1mm_np = ref_arr.astype(np.uint8)
            gt_1mm_np[gt_1mm_np > 0] = 1
            target_shape = gt_1mm_np.shape
        else:
            target_shape = ref_arr.shape

        # Undo the canonicalization flip from preprocess_COPDgene_CT.py so the
        # prediction array lines up with the reference NIfTI's original
        # direction matrix before CopyInformation is applied.
        flips_xyz = canonicalize_flip_axes(ref_img_sitk)
        pred_1mm_np = restore_original_orientation(pred_1mm_np, flips_xyz)

        pred_1mm_np = match_shape(pred_1mm_np, target_shape, fill_value=0)

        if KEEP_LARGEST_CC:
            pred_1mm_np = keep_largest_connected_components(
                pred_1mm_np,
                k=NUM_LARGEST_CC,
                full_connectivity=LCC_FULL_CONNECTIVITY,
            )

        out_pred = f"{OUTPUT_DIR}/{stem}_pred_1mm.nii.gz"
        save_with_reference(pred_1mm_np, ref_img_sitk, out_pred)

        if gt_mode:
            out_gt   = f"{OUTPUT_DIR}/{stem}_gt_1mm.nii.gz"
            out_conf = f"{OUTPUT_DIR}/{stem}_confusion_unet_1mm.nii.gz"
            save_with_reference(gt_1mm_np, ref_img_sitk, out_gt)
            generate_confusion_mask(out_gt, out_pred, out_conf)


if __name__ == "__main__":
    main()
