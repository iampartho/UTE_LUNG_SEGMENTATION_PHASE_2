"""
eval_step_1_mri_to_ct.py
========================

Two-stage MRI lung-segmentation pipeline using CycleGAN-based domain translation.

Stage 1 — Translation  (MRI → fake CT)
    Load each UTE/MRI .npy file, normalise to [-1, 1] with ``normalise_one_one``
    (matching the CycleGAN training pre-process), then run the trained
    ``G_UTE_to_CT`` generator with a SlidingWindowInferer that mirrors the
    CycleGAN validation setup (roi_size=256³, overlap=0, gaussian blending).
    The generator ends with a Tanh layer, so the output is already in [-1, 1];
    no further normalisation is applied.

Stage 2 — Segmentation  (fake CT → lung mask)
    Apply the same VariableSpatialFix + ToTensor transforms used in
    eval_step_1.py, then run the trained BasicUNet segmentation model.
    Because the segmentation model was trained on CT (and CT-like augmented
    data), feeding it a fake CT produced from the MRI lets it generalise to
    the MRI domain without any fine-tuning.

Dataset modes (mirrors eval_step_1_resample_to_1mm.py):

  Simple mode  (GT_1MM_DIR == '' and IMG_1MM_DIR == ''):
    - Read TEST_CSV directly; each row is a 1.25mm .npy with image (+ optional GT).
    - Save pred / gt / fake_ct at 1.25mm spacing.

  Eval mode  (GT_1MM_DIR set):
    - Pair 1.25mm CSV rows with 1mm GT NIfTIs by stem (ILD-style evaluation).
    - Run inference at 1.25mm, resample prediction to 1mm, align to reference
      NIfTI shape + orientation, optionally keep largest CCs.
    - Save: pred, GT, confusion mask (all at 1mm with reference metadata).

  Predict-only mode  (GT_1MM_DIR == '' and IMG_1MM_DIR set):
    - Pair 1.25mm CSV rows with 1mm image NIfTIs by stem (OECLAD-style).
    - Same 1.25mm → 1mm pipeline; save prediction only.

This script is intentionally kept separate from eval_step_1.py so that the
original evaluation path is not altered in any way.
"""

import os
import sys
import glob

import torch
import pandas as pd
import numpy as np
from torchvision import transforms
import SimpleITK as sitk
import tqdm
from scipy import ndimage as ndi
from monai.inferers import SlidingWindowInferer

from basic_unet_disentangled import BasicUNet
from preprocess_COPDgene_CT import resample_iso

_CYCLEGAN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cyclegan")
if _CYCLEGAN_DIR not in sys.path:
    sys.path.insert(0, _CYCLEGAN_DIR)
from models import CycleGAN  # noqa: E402

from eval_step_1 import (  # noqa: E402
    normalise_one_one,
    normalise_zero_one,
    generate_confusion_mask,
    VariableSpatialFix,
    ToTensor,
    generate_nifty,
)


# ======================================================================
#  CONFIG  — edit then run; no argparse by design
# ======================================================================

# 1.25mm input CSV. For ILD / OECLAD, point this at the resampled 1.25mm CSV.
TEST_CSV = "./ids/only_ute_1.25mm.csv"

# 1mm GT NIfTI folder. Set for ILD eval; leave '' for other modes.
GT_1MM_DIR = ""
# Required when GT_1MM_DIR == '' and using extended mode. 1mm image NIfTIs
# used as the shape + spacing/origin/direction reference (OECLAD predict-only).
IMG_1MM_DIR = ""

SEG_CHECKPOINT = (
    "./save_models/"
    "best_unet_td1.pth"
    #"./save_models/current_best_may_2026/"
    #"best_bunet_causality_paper_ct_train_UTE_test_w_tversky_wo_kl_only_gin_roughness_enforced.pth"
)

CYCLEGAN_CHECKPOINT = (
    "./cyclegan/"
    "Epoch_30_cycleGAN_seg_loss_recon_ct_weighted_by_50_ct_only_seg_no_leakage_trained_on_frc+rv_only_train_crop_192_volmatched.pth"
)

OUTPUT_DIR = "./prediction/mri_to_ct_cyclegan_2_marissa_data"

INPUT_SPACING_MM  = (1.25, 1.25, 1.25)
OUTPUT_SPACING_MM = (1.0, 1.0, 1.0)

CYCLEGAN_ROI     = (192, 192, 192)
CYCLEGAN_OVERLAP = 0.9

NIFTY_EXTS = (".nii.gz", ".nii")
GT_MASK_SUFFIX = "_mask"

NUM_DOUBLE_STRIDE_CONV = 4
PAD_VAL = 0.5
THRESHOLD = 0.5

KEEP_LARGEST_CC = True
NUM_LARGEST_CC = 2
LCC_FULL_CONNECTIVITY = True

# ---------------------------------------------------------------------
#  Fake-CT background denoising  (body-mask + hole-fill)
# ---------------------------------------------------------------------
# The SlidingWindowInferer (high overlap) leaves bright speckle noise in the
# air background of the fake CT. Since CT tissue is bright and lungs/air are
# dark, we threshold to get the bright body shell, keep its largest connected
# component (dropping the disconnected background specks), then fill holes so
# the dark lung cavities enclosed by the body wall become part of the mask.
# Everything outside the body mask is set to clean air, matching real CT.
# Toggle APPLY_FAKE_CT_DENOISE to compare with/without.
APPLY_FAKE_CT_DENOISE   = True
DENOISE_TISSUE_THRESHOLD = -0.3   # voxels brighter than this are candidate body tissue
DENOISE_NUM_BODY_CC      = 1      # keep this many largest components as the body
DENOISE_CLOSING_ITERS    = 2      # binary closing to seal body-wall gaps before hole-fill
DENOISE_DILATE_ITERS     = 1      # grow body mask so a rim of chest wall is not clipped
DENOISE_BG_FILL          = -1.0   # value assigned outside the body (air in the [-1, 1] CT range)


# ======================================================================
#  Model loading helpers
# ======================================================================

def load_cyclegan_generator(checkpoint_path: str, device: torch.device):
    """Load the full CycleGAN checkpoint and return the UTE→CT generator."""
    cyclegan = CycleGAN().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    cyclegan.load_state_dict(state_dict)
    cyclegan.eval()
    return cyclegan.G_UTE_to_CT


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
#  Inference helpers
# ======================================================================

def translate_mri_to_ct(
    mri_norm: np.ndarray,
    generator,
    inferer: SlidingWindowInferer,
    device: torch.device,
) -> np.ndarray:
    """Translate a normalised MRI volume to a fake CT via the CycleGAN generator."""
    mri_tensor = (
        torch.from_numpy(mri_norm).unsqueeze(0).unsqueeze(0).float().to(device)
    )
    with torch.no_grad():
        fake_ct_tensor = inferer(inputs=mri_tensor, network=generator)
    return fake_ct_tensor.squeeze().detach().cpu().numpy().astype(np.float32)


def denoise_fake_ct(
    fake_ct_np: np.ndarray,
    threshold: float = DENOISE_TISSUE_THRESHOLD,
    num_cc: int = DENOISE_NUM_BODY_CC,
    closing_iters: int = DENOISE_CLOSING_ITERS,
    dilate_iters: int = DENOISE_DILATE_ITERS,
    bg_fill: float = DENOISE_BG_FILL,
) -> np.ndarray:
    """Suppress background speckle noise in a fake CT via a body mask.

    Steps: threshold bright tissue -> (optional) close body-wall gaps ->
    keep the ``num_cc`` largest connected components (the body, dropping the
    disconnected background specks) -> fill holes so the dark lung cavities
    enclosed by the body become foreground -> (optional) dilate for a safety
    margin. Voxels outside the resulting body mask are set to ``bg_fill``
    (clean air); voxels inside the body — including the lungs — are untouched.
    """
    body = fake_ct_np > threshold
    if not body.any():
        return fake_ct_np

    if closing_iters > 0:
        body = ndi.binary_closing(body, iterations=closing_iters)

    labels, n = ndi.label(body)
    if n > 1:
        counts = np.bincount(labels.ravel())
        counts[0] = 0  # ignore background label
        keep = np.argsort(counts)[::-1][:max(1, num_cc)]
        body = np.isin(labels, keep)

    body = ndi.binary_fill_holes(body)

    if dilate_iters > 0:
        body = ndi.binary_dilation(body, iterations=dilate_iters)

    out = fake_ct_np.copy()
    out[~body] = bg_fill
    return out.astype(np.float32)


def segment_fake_ct(
    fake_ct_np: np.ndarray,
    seg_model: BasicUNet,
    transform,
    device: torch.device,
) -> np.ndarray:
    """Run the segmentation model on a fake CT volume at 1.25mm.

    Returns a binary uint8 prediction cropped to the original input shape
    (undoing VariableSpatialFix padding).
    """
    d, h, w = fake_ct_np.shape
    fake_ct_tensor = transform(fake_ct_np).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = seg_model(fake_ct_tensor)
        prob = torch.sigmoid(logits)
        binary = (prob > THRESHOLD).float()

    pred_np = binary.detach().squeeze().cpu().numpy().astype(np.uint8)
    return pred_np[:d, :h, :w]


# ======================================================================
#  1mm resampling / reference helpers  (from eval_step_1_resample_to_1mm.py)
# ======================================================================

def _stem_npy(p):
    return os.path.basename(p).split(".npy")[0]


def _stem_nifty(p):
    """Return the NIfTI stem matching the corresponding npy basename."""
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
    """Pair 1.25mm CSV rows with NIfTI files in ``ref_dir`` by stem."""
    df_125 = pd.read_csv(csv_125_path)
    df_125["__stem__"] = df_125["filepaths"].apply(_stem_npy)

    ref_files = []
    for ext in NIFTY_EXTS:
        ref_files.extend(glob.glob(os.path.join(ref_dir, f"*{ext}")))

    if not ref_files:
        raise ValueError(f"No NIfTI files found in {ref_dir}")

    ref_df = pd.DataFrame({ref_col_name: ref_files})
    ref_df["__stem__"] = ref_df[ref_col_name].apply(_stem_nifty)

    merged = df_125.merge(ref_df, on="__stem__")
    if merged.empty:
        raise ValueError(
            f"No matching basenames between the 1.25mm CSV and {ref_dir}. "
            "Make sure 1.25mm npy filenames and 1mm NIfTI filenames share stems."
        )
    return merged


def numpy_to_sitk(arr_np, spacing):
    img = sitk.GetImageFromArray(arr_np)
    img.SetSpacing(tuple(float(s) for s in spacing))
    return img


def resample_to_1mm(arr_125_np, in_spacing, out_spacing, interp=sitk.sitkNearestNeighbor):
    """Resample a 1.25mm volume to 1mm using ``resample_iso``."""
    if not all(s == out_spacing[0] for s in out_spacing):
        raise ValueError("resample_iso only supports isotropic output spacing.")
    arr_125_sitk = numpy_to_sitk(arr_125_np, in_spacing)
    arr_1mm_sitk = resample_iso(
        arr_125_sitk,
        out_spacing[0],
        0,
        interp,
    )
    return sitk.GetArrayFromImage(arr_1mm_sitk)


def match_shape(arr, target_shape, fill_value=0):
    """Crop/zero-pad ``arr`` per-axis so its shape matches ``target_shape``."""
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
            a = np.pad(a, pad, mode="constant", constant_values=fill_value)
    return a


def save_with_reference(arr_np, reference_img, output_path):
    """Write ``arr_np`` as a NIfTI sharing metadata with ``reference_img``."""
    out_img = sitk.GetImageFromArray(arr_np)
    out_img.CopyInformation(reference_img)
    sitk.WriteImage(out_img, output_path)


def keep_largest_connected_components(arr_np, k=1, full_connectivity=True):
    """Keep the ``k`` largest connected foreground components of a binary mask."""
    if k is None or k <= 0:
        return arr_np.astype(np.uint8)
    if arr_np.sum() == 0:
        return arr_np.astype(np.uint8)

    binary = (arr_np > 0).astype(np.uint8)
    sitk_img = sitk.GetImageFromArray(binary)

    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(bool(full_connectivity))
    labels = cc_filter.Execute(sitk_img)

    relabeled = sitk.RelabelComponent(labels, sortByObjectSize=True)
    relabeled_np = sitk.GetArrayFromImage(relabeled)

    keep_mask = (relabeled_np >= 1) & (relabeled_np <= int(k))
    return keep_mask.astype(np.uint8)


def canonicalize_flip_axes(reference_img):
    """Return per-axis flip flags that preprocessing would have applied."""
    dirs = reference_img.GetDirection()
    dirs_diag = np.sign(np.array([dirs[0], dirs[4], dirs[8]]))
    return [bool(x != 1) for x in dirs_diag]


def restore_original_orientation(arr_np, flips_xyz):
    """Reverse canonicalization flips so arrays align with the reference NIfTI."""
    out = arr_np
    if flips_xyz[0]:
        out = np.flip(out, axis=2)
    if flips_xyz[1]:
        out = np.flip(out, axis=1)
    if flips_xyz[2]:
        out = np.flip(out, axis=0)
    return np.ascontiguousarray(out)


def align_to_reference(arr_np, ref_img_sitk, fill_value=0):
    """Restore orientation and crop/pad to match the reference NIfTI shape."""
    flips_xyz = canonicalize_flip_axes(ref_img_sitk)
    arr_np = restore_original_orientation(arr_np, flips_xyz)
    ref_shape = sitk.GetArrayFromImage(ref_img_sitk).shape
    return match_shape(arr_np, ref_shape, fill_value=fill_value)


# ======================================================================
#  Main
# ======================================================================

def run_simple_mode(device, generator, cyclegan_inferer, seg_model, transform):
    """Original UTE-style loop: GT in npy, outputs at 1.25mm."""
    test_df = pd.read_csv(TEST_CSV)
    all_files = test_df["filepaths"].values

    print(f"\n[simple mode] Running two-stage inference on {len(all_files)} files ...\n")

    for _, each_file in tqdm.tqdm(enumerate(all_files), total=len(all_files)):
        arr = np.load(each_file)
        img_np = arr[:, :, :, 0]
        img_gt = arr[:, :, :, 1] if arr.ndim == 4 else None

        mri_norm = normalise_one_one(img_np)
        gt_norm = normalise_zero_one(img_gt) if img_gt is not None else None

        fake_ct_np = translate_mri_to_ct(mri_norm, generator, cyclegan_inferer, device)
        if APPLY_FAKE_CT_DENOISE:
            fake_ct_np = denoise_fake_ct(fake_ct_np)
        pred_np = segment_fake_ct(fake_ct_np, seg_model, transform, device)

        filename = f"{'_'.join(each_file.split('/')[-4:])}".split(".npy")[0]

        generate_nifty(pred_np, INPUT_SPACING_MM, f"{OUTPUT_DIR}/{filename}_pred.nii.gz")

        if gt_norm is not None:
            _, gt_tensor = transform([fake_ct_np, gt_norm])
            generate_nifty(
                gt_tensor.squeeze().cpu().numpy(),
                INPUT_SPACING_MM,
                f"{OUTPUT_DIR}/{filename}_gt.nii.gz",
            )

        generate_nifty(
            fake_ct_np,
            INPUT_SPACING_MM,
            f"{OUTPUT_DIR}/{filename}_fake_ct.nii.gz",
        )


def run_extended_mode(device, generator, cyclegan_inferer, seg_model, transform):
    """ILD / OECLAD loop: pair with 1mm reference NIfTIs, save at 1mm."""
    gt_mode = bool(GT_1MM_DIR)
    if gt_mode:
        ref_dir = GT_1MM_DIR
        ref_col = "gt_path"
        mode_label = "GT (eval) mode"
    else:
        if not IMG_1MM_DIR:
            raise ValueError(
                "GT_1MM_DIR is empty -> IMG_1MM_DIR must be set to a folder of "
                "1mm reference image NIfTIs (used for shape and metadata)."
            )
        ref_dir = IMG_1MM_DIR
        ref_col = "img_path"
        mode_label = "predict-only mode (no GT)"

    pair_df = build_pair_table(TEST_CSV, ref_dir, ref_col)
    print(
        f"\n[extended mode] {mode_label}: {len(pair_df)} scans matched between "
        f"1.25mm CSV and 1mm reference folder ({ref_dir})."
    )

    for _, row in tqdm.tqdm(pair_df.iterrows(), total=len(pair_df)):
        path_125 = row["filepaths"]
        ref_path = row[ref_col]
        stem = row["__stem__"]

        arr_125 = np.load(path_125)
        img_125_np = arr_125[:, :, :, 0] if arr_125.ndim == 4 else arr_125

        mri_norm = normalise_one_one(img_125_np)
        fake_ct_np = translate_mri_to_ct(mri_norm, generator, cyclegan_inferer, device)
        if APPLY_FAKE_CT_DENOISE:
            fake_ct_np = denoise_fake_ct(fake_ct_np)
        pred_125_np = segment_fake_ct(fake_ct_np, seg_model, transform, device)

        pred_1mm_np = resample_to_1mm(
            pred_125_np, INPUT_SPACING_MM, OUTPUT_SPACING_MM, sitk.sitkNearestNeighbor
        )
        fake_ct_1mm_np = resample_to_1mm(
            fake_ct_np.astype(np.float32),
            INPUT_SPACING_MM,
            OUTPUT_SPACING_MM,
            sitk.sitkLinear,
        )

        ref_img_sitk = sitk.ReadImage(ref_path)
        ref_arr = sitk.GetArrayFromImage(ref_img_sitk)

        if gt_mode:
            gt_1mm_np = ref_arr.astype(np.uint8)
            gt_1mm_np[gt_1mm_np > 0] = 1

        pred_1mm_np = align_to_reference(pred_1mm_np, ref_img_sitk, fill_value=0)
        fake_ct_1mm_np = align_to_reference(fake_ct_1mm_np, ref_img_sitk, fill_value=0.0)

        if KEEP_LARGEST_CC:
            pred_1mm_np = keep_largest_connected_components(
                pred_1mm_np,
                k=NUM_LARGEST_CC,
                full_connectivity=LCC_FULL_CONNECTIVITY,
            )

        out_pred = f"{OUTPUT_DIR}/{stem}_pred_1mm.nii.gz"
        save_with_reference(pred_1mm_np, ref_img_sitk, out_pred)

        out_fake_ct = f"{OUTPUT_DIR}/{stem}_fake_ct_1mm.nii.gz"
        save_with_reference(fake_ct_1mm_np.astype(np.float32), ref_img_sitk, out_fake_ct)

        if gt_mode:
            out_gt = f"{OUTPUT_DIR}/{stem}_gt_1mm.nii.gz"
            out_conf = f"{OUTPUT_DIR}/{stem}_confusion_unet_1mm.nii.gz"
            save_with_reference(gt_1mm_np, ref_img_sitk, out_gt)
            generate_confusion_mask(out_gt, out_pred, out_conf)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extended_mode = bool(GT_1MM_DIR or IMG_1MM_DIR)

    print(f"[config] device            = {device}")
    print(f"[config] test csv          = {TEST_CSV}")
    print(f"[config] seg checkpoint    = {SEG_CHECKPOINT}")
    print(f"[config] cyclegan ckpt     = {CYCLEGAN_CHECKPOINT}")
    print(f"[config] output dir        = {OUTPUT_DIR}")
    print(f"[config] cyclegan roi      = {CYCLEGAN_ROI}, overlap = {CYCLEGAN_OVERLAP}")
    print(f"[config] fake-ct denoise   = {APPLY_FAKE_CT_DENOISE}"
          + (f" (thr={DENOISE_TISSUE_THRESHOLD}, cc={DENOISE_NUM_BODY_CC}, "
             f"close={DENOISE_CLOSING_ITERS}, dilate={DENOISE_DILATE_ITERS}, "
             f"bg={DENOISE_BG_FILL})" if APPLY_FAKE_CT_DENOISE else ""))
    print(f"[config] mode              = {'extended (1mm)' if extended_mode else 'simple (1.25mm)'}")
    if extended_mode:
        print(f"[config] gt_1mm_dir        = {GT_1MM_DIR or '(not set)'}")
        print(f"[config] img_1mm_dir       = {IMG_1MM_DIR or '(not set)'}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[1/2] Loading CycleGAN generator (UTE → CT) ...")
    generator = load_cyclegan_generator(CYCLEGAN_CHECKPOINT, device)

    print("[2/2] Loading lung-segmentation model ...")
    seg_model = load_seg_model(SEG_CHECKPOINT, device)

    cyclegan_inferer = SlidingWindowInferer(
        roi_size=CYCLEGAN_ROI,
        sw_batch_size=1,
        overlap=CYCLEGAN_OVERLAP,
        mode="gaussian",
    )

    transform = transforms.Compose([
        VariableSpatialFix(num_of_double_stride_conv=NUM_DOUBLE_STRIDE_CONV, padval=PAD_VAL),
        ToTensor(),
    ])

    if extended_mode:
        run_extended_mode(device, generator, cyclegan_inferer, seg_model, transform)
        print(f"\nDone.  Results written to {OUTPUT_DIR}")
        print("  *_pred_1mm.nii.gz           — segmentation prediction at 1mm")
        print("  *_fake_ct_1mm.nii.gz        — CycleGAN-translated CT at 1mm (for QC)")
        if GT_1MM_DIR:
            print("  *_gt_1mm.nii.gz             — ground-truth mask at 1mm")
            print("  *_confusion_unet_1mm.nii.gz — confusion mask")
    else:
        run_simple_mode(device, generator, cyclegan_inferer, seg_model, transform)
        print(f"\nDone.  Results written to {OUTPUT_DIR}")
        print("  *_pred.nii.gz    — segmentation prediction")
        print("  *_gt.nii.gz      — ground-truth mask")
        print("  *_fake_ct.nii.gz — CycleGAN-translated CT (for visual QC)")
