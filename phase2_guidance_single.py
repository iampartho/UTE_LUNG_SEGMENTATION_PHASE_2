"""
Phase 2 activation guidance for a SINGLE scan -- a thin driver around
``phase2_guidance.guide_scan`` for tuning the leashes on one bad prediction.

WHY THIS FILE
-------------
``phase2_guidance.py`` loops over a CSV of scans. This script runs the *exact*
same machinery (frozen seg net + frozen cross-modal energy + x4-delta activation
guidance + every leash) on ONE scan so you can iterate on "how much can I rescue
pred0 by changing the leashes my own way". It does NOT re-implement anything: it
imports ``guide_scan`` and friends, then overrides the leash globals on the
``phase2_guidance`` module from the editable ``LEASHES`` dict below. Because
``guide_scan`` reads those globals at call time, editing ``LEASHES`` here is all
you need -- no argparse, by request.

TWO INPUT MODES (set INPUT_KIND)
--------------------------------
  INPUT_KIND = "numpy":
      IMG_PATH is a .npy 3D image and NUMPY_SPACING_MM is its voxel spacing in
      SimpleITK (x, y, z) world order. The image is resampled to 1.25mm isotropic
      (B-spline), guided, and the mask is written as a 1.25mm NIfTI.

  INPUT_KIND = "nifti":
      IMG_PATH is a .nii/.nii.gz image. Its orientation is canonicalized (the same
      diagonal-sign flip as preprocess_COPDgene_CT), resampled to 1.25mm isotropic,
      guided, then the mask is resampled BACK to the native spacing/grid and saved
      as a NIfTI sharing the original image's spacing/origin/direction.

GROUND TRUTH (optional)
-----------------------
GT_PATH may be a .npy or .nii/.nii.gz mask. When set, Dice is reported for pred0
(baseline m0) and the refined mask. Dice is always computed in the 1.25mm guidance
frame (apples-to-apples for the leash effect). For the nifti mode it is ALSO
reported in the native frame of the saved deliverable.

Everything is frozen; only the bottleneck perturbation delta is learned. Edit the
CONFIG + LEASHES blocks, then run.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

import phase2_guidance as p2
from phase2_guidance import guide_scan, load_energy, dice, keep_largest_cc_capped
from basic_unet_disentangled import BasicUNet
from eval_step_1 import generate_nifty, generate_confusion_mask
from eval_step_1_resample_to_1mm import (
    match_shape,
    canonicalize_flip_axes,
    restore_original_orientation,
    save_with_reference,
)
from preprocess_COPDgene_CT import resample_iso, resample_ref


# ===========================================================================
# CONFIG -- change these, then run.
# ===========================================================================
INPUT_KIND = "numpy"          # "numpy" | "nifti"

IMG_PATH = "/Shared/lss_segerard/parthghosh/data/UTE_new_data_numpy/103-035/20180424/AnatCorrLungs.npy"
GT_PATH  = "/Shared/lss_segerard/parthghosh/data/UTE_new_data_numpy/103-035/20180424/AnatCorrLungs.npy"  # '' -> no Dice

# Only used when INPUT_KIND == "numpy": voxel spacing of IMG_PATH in SimpleITK
# (x, y, z) order (i.e. reversed numpy axes). Scalar or 3-tuple. The GT npy, if
# given, is assumed to share this spacing.
NUMPY_SPACING_MM = (1.25, 1.25, 1.25)

OUTPUT_DIR = "./prediction/phase2_guidance_new_run/single_scan"
STEM = ""                     # output filename stem; '' -> derived from IMG_PATH

# Sweep log: every run appends ONE row (timestamp + stem + all leashes + metrics
# + Dice) so you can compare leash configs. '' -> disable logging.
METRICS_CSV = "./phase2_guidance_logs/single_scan/phase2_guidance_single_sweep.csv"

MODALITY = "UTE"              # 'UTE' | 'CT'  (must match the seg net / energy)

PROC_SPACING_MM = (1.25, 1.25, 1.25)   # guidance always runs here (UTE native)


# ===========================================================================
# LEASHES -- THIS is what you tune. Pushed onto the phase2_guidance module
# before guide_scan() runs (see _apply_leashes). Names + meanings are identical
# to phase2_guidance.py; see its header for the full rationale.
# ===========================================================================
LEASHES = dict(
    USE_GUIDANCE=True,         # False -> emit baseline m0 (zero-leash anchor)

    MAX_ITERS=200,
    LR=0.05,
    OPTIMIZER="adam",          # 'adam' | 'sgd'

    # Leash 1: proximal term on ||delta||.
    USE_PROX=True,
    LAMBDA_PROX=0.01,

    # Leash 2: hard trust region on ||delta||.
    USE_DELTA_CAP=True,
    DELTA_MAX_NORM=80.0,

    # Leash 3: energy-plateau early stop (protects already-good predictions).
    USE_EARLY_STOP=True,
    ENERGY_REL_TOL=1e-3,
    PATIENCE=3,

    # Leash 4: energy-gated volume guard.
    USE_VOLUME_GUARD=False,
    VOLUME_REL_TOL=0.25,
    VOLUME_GUARD_ENERGY_BYPASS=0.15,

    # Leash 5: component guard.
    USE_COMPONENT_GUARD=True,
    MAX_NEW_COMPONENTS=1,
    COMPONENT_FULL_CONN=True,

    # Leash 6: keep-largest-components post-process.
    USE_LCC_POST=True,
    NUM_LARGEST_CC=2,
    LCC_FULL_CONNECTIVITY=True,
    LCC_MIN_FRAC=0.10,

    # Leash 7: energy fallback (revert to m0 unless E beats it by a margin).
    USE_ENERGY_FALLBACK=True,
    ENERGY_FALLBACK_MARGIN=0.10,

    # Seg-net / energy knobs (kept here so single-scan tuning is self-contained).
    SEG_THRESHOLD=0.5,
    GAUSSIAN_SIGMA=1.0,
    TARGET_NORM=True,
)

DEVICE = p2.DEVICE


# ===========================================================================
# Wiring: push LEASHES onto phase2_guidance, then load the frozen models.
# ===========================================================================
def _apply_leashes():
    for k, v in LEASHES.items():
        if not hasattr(p2, k):
            raise AttributeError(f"LEASHES key '{k}' is not a phase2_guidance global")
        setattr(p2, k, v)
    print("[leashes] applied:")
    for k in sorted(LEASHES):
        print(f"    {k} = {LEASHES[k]}")


def load_models():
    seg = BasicUNet().to(DEVICE)
    state = torch.load(p2.SEG_CKPT, map_location=DEVICE)
    res = seg.load_state_dict(state, strict=False)
    if res.missing_keys:
        print(f"[seg] missing keys ({len(res.missing_keys)}): {res.missing_keys[:6]} ...")
    if res.unexpected_keys:
        print(f"[seg] unexpected keys ({len(res.unexpected_keys)}): {res.unexpected_keys[:6]} ...")
    seg.eval()
    for p in seg.parameters():
        p.requires_grad_(False)
    print(f"[seg] loaded {p2.SEG_CKPT}")

    energy = load_energy()
    energy.eval()
    for p in energy.parameters():
        p.requires_grad_(False)
    return seg, energy


# ===========================================================================
# I/O helpers
# ===========================================================================
def _spacing_xyz(spacing):
    if np.isscalar(spacing):
        return (float(spacing),) * 3
    return tuple(float(s) for s in spacing)


def _binarize(arr):
    return (arr > 0.5).astype(np.uint8)


def _load_mask_any(path, spacing_for_npy):
    """Load a GT mask from .npy or NIfTI as a raw (uint8, sitk_img_or_None) pair.

    For NIfTI we also return the sitk image so the nifti path can resample/align
    it; for npy there is no geometry so the second item is None.
    """
    if path.endswith(".npy"):
        return _binarize(np.load(path)[...,1]), None
    img = sitk.ReadImage(path)
    return _binarize(sitk.GetArrayFromImage(img)), img


def numpy_to_125(arr, spacing_xyz, interp):
    """Wrap a bare numpy array with the given spacing and resample to 1.25mm iso."""
    im = sitk.GetImageFromArray(arr)
    im.SetSpacing(spacing_xyz)
    im_125 = resample_iso(im, PROC_SPACING_MM[0], 0, interp)
    return sitk.GetArrayFromImage(im_125)


def mask_125_to_native(mask_125, img_125_sitk, img_flipped_sitk, flips, orig_sitk):
    """Resample a 1.25mm (canonical-orientation) binary mask back onto the native
    NIfTI grid: resample -> un-flip orientation -> shape-match -> optional LCC."""
    m = sitk.GetImageFromArray(mask_125.astype(np.uint8))
    m.CopyInformation(img_125_sitk)
    m_native = resample_ref(m, img_flipped_sitk, 0, sitk.sitkNearestNeighbor)
    arr = sitk.GetArrayFromImage(m_native).astype(np.uint8)
    arr = restore_original_orientation(arr, flips)
    arr = match_shape(arr, sitk.GetArrayFromImage(orig_sitk).shape, fill_value=0)
    if LEASHES.get("USE_LCC_POST"):
        arr = keep_largest_cc_capped(
            arr, k=LEASHES["NUM_LARGEST_CC"], keep_frac=LEASHES["LCC_MIN_FRAC"],
            full_connectivity=LEASHES["LCC_FULL_CONNECTIVITY"]).astype(np.uint8)
    return arr


# ===========================================================================
# Per-mode pipelines
# ===========================================================================
def run_numpy(seg, energy, stem):
    spc = _spacing_xyz(NUMPY_SPACING_MM)
    img_native = np.load(IMG_PATH)[...,0].astype(np.float32)
    if img_native.ndim == 4:                      # tolerate stacked [...,0]=img
        img_native = img_native[..., 0].astype(np.float32)
    print(f"[numpy] {IMG_PATH} shape={img_native.shape} spacing(xyz)={spc}")
    img_125 = numpy_to_125(img_native, spc, sitk.sitkBSpline)
    print(f"[numpy] resampled to 1.25mm -> {img_125.shape}")

    m0, refined, mtr = guide_scan(seg, energy, img_125, MODALITY)

    generate_nifty(refined.astype(np.uint8), PROC_SPACING_MM,
                   f"{OUTPUT_DIR}/{stem}_pred.nii.gz")
    generate_nifty(m0.astype(np.uint8), PROC_SPACING_MM,
                   f"{OUTPUT_DIR}/{stem}_pred0.nii.gz")

    if GT_PATH:
        gt_raw, _ = _load_mask_any(GT_PATH, spc)
        gt_125 = numpy_to_125(gt_raw, spc, sitk.sitkNearestNeighbor)
        gt_125 = match_shape(gt_125, refined.shape, fill_value=0)
        generate_nifty(gt_125.astype(np.uint8), PROC_SPACING_MM,
                       f"{OUTPUT_DIR}/{stem}_gt.nii.gz")
        generate_confusion_mask(f"{OUTPUT_DIR}/{stem}_gt.nii.gz",
                                f"{OUTPUT_DIR}/{stem}_pred.nii.gz",
                                f"{OUTPUT_DIR}/{stem}_confusion.nii.gz")
        mtr["dice_m0_1.25mm"] = dice(m0, gt_125)
        mtr["dice_refined_1.25mm"] = dice(refined, gt_125)
    return mtr


def run_nifti(seg, energy, stem):
    img_sitk = sitk.ReadImage(IMG_PATH)
    flips = canonicalize_flip_axes(img_sitk)
    img_flipped = sitk.Flip(img_sitk, flips)
    img_125_sitk = resample_iso(img_flipped, PROC_SPACING_MM[0], 0, sitk.sitkBSpline)
    img_125 = sitk.GetArrayFromImage(img_125_sitk).astype(np.float32)
    print(f"[nifti] {IMG_PATH}")
    print(f"[nifti] native shape={sitk.GetArrayFromImage(img_sitk).shape} "
          f"spacing(xyz)={img_sitk.GetSpacing()} flips={flips}")
    print(f"[nifti] resampled to 1.25mm -> {img_125.shape}")

    m0, refined, mtr = guide_scan(seg, energy, img_125, MODALITY)

    # Save the refined + baseline masks at NATIVE spacing.
    refined_native = mask_125_to_native(refined, img_125_sitk, img_flipped, flips, img_sitk)
    m0_native = mask_125_to_native(m0, img_125_sitk, img_flipped, flips, img_sitk)
    save_with_reference(refined_native, img_sitk, f"{OUTPUT_DIR}/{stem}_pred.nii.gz")
    save_with_reference(m0_native, img_sitk, f"{OUTPUT_DIR}/{stem}_pred0.nii.gz")

    if GT_PATH:
        gt_raw, gt_sitk = _load_mask_any(GT_PATH, None)
        # 1.25mm-frame Dice (guidance frame): align GT to the 1.25mm image grid.
        if gt_sitk is not None:
            gt_flipped = sitk.Flip(sitk.Cast(gt_sitk, sitk.sitkUInt8), flips)
            gt_125 = sitk.GetArrayFromImage(
                resample_ref(gt_flipped, img_125_sitk, 0, sitk.sitkNearestNeighbor)
            ).astype(np.uint8)
        else:                                     # GT given as npy: best-effort resample
            gt_125 = numpy_to_125(gt_raw, _spacing_xyz(NUMPY_SPACING_MM),
                                  sitk.sitkNearestNeighbor)
        gt_125 = match_shape(gt_125, refined.shape, fill_value=0)
        mtr["dice_m0_1.25mm"] = dice(m0, gt_125)
        mtr["dice_refined_1.25mm"] = dice(refined, gt_125)

        # Native-frame Dice (the saved deliverable) + GT/confusion NIfTIs.
        if gt_sitk is not None:
            gt_native = match_shape(_binarize(sitk.GetArrayFromImage(gt_sitk)),
                                    sitk.GetArrayFromImage(img_sitk).shape, fill_value=0)
            save_with_reference(gt_native.astype(np.uint8), img_sitk,
                                f"{OUTPUT_DIR}/{stem}_gt.nii.gz")
            generate_confusion_mask(f"{OUTPUT_DIR}/{stem}_gt.nii.gz",
                                    f"{OUTPUT_DIR}/{stem}_pred.nii.gz",
                                    f"{OUTPUT_DIR}/{stem}_confusion.nii.gz")
            mtr["dice_m0_native"] = dice(m0_native, gt_native)
            mtr["dice_refined_native"] = dice(refined_native, gt_native)
    return mtr


def append_sweep_row(stem, mtr):
    """Append one row (timestamp + stem + every leash + metrics + Dice) to
    METRICS_CSV. Re-reads + re-writes the whole file so the column set stays
    aligned even when you change which leashes are present between runs."""
    if not METRICS_CSV:
        return
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"),
           "stem": stem, "input_kind": INPUT_KIND}
    row.update(LEASHES)                       # the config that produced this row
    row.update(mtr)                           # E0/E_final/iters/dice_*/...
    os.makedirs(os.path.dirname(METRICS_CSV) or ".", exist_ok=True)
    df_new = pd.DataFrame([row])
    if os.path.exists(METRICS_CSV):
        df_new = pd.concat([pd.read_csv(METRICS_CSV), df_new], ignore_index=True)
    df_new.to_csv(METRICS_CSV, index=False)
    print(f" sweep row appended -> {METRICS_CSV} ({len(df_new)} rows)")


# ===========================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = STEM or os.path.basename(IMG_PATH).split(".npy")[0].split(".nii")[0]
    print(f"device={DEVICE} | INPUT_KIND={INPUT_KIND} | stem={stem}")

    _apply_leashes()
    seg, energy = load_models()

    if INPUT_KIND == "numpy":
        mtr = run_numpy(seg, energy, stem)
    elif INPUT_KIND == "nifti":
        mtr = run_nifti(seg, energy, stem)
    else:
        raise ValueError(f"INPUT_KIND must be 'numpy' or 'nifti', got {INPUT_KIND!r}")

    print("\n================ single-scan summary ================")
    print(f" scan          : {stem}")
    print(f" energy E      : {mtr['E0']:.4f} -> {mtr['E_final']:.4f}")
    print(f" iters         : {mtr['n_iters']}  fell_back={mtr['fell_back']} ({mtr['reason']})")
    print(f" volume        : {mtr['vol0']:.0f} -> {mtr['vol_final']:.0f} voxels")
    print(f" #components   : {mtr['nc0']} -> {mtr['nc_final']}")
    if "dice_m0_1.25mm" in mtr:
        d0, d1 = mtr["dice_m0_1.25mm"], mtr["dice_refined_1.25mm"]
        print(f" Dice @1.25mm  : {d0:.4f} -> {d1:.4f}  (delta {d1 - d0:+.4f})")
    if "dice_m0_native" in mtr:
        d0, d1 = mtr["dice_m0_native"], mtr["dice_refined_native"]
        print(f" Dice @native  : {d0:.4f} -> {d1:.4f}  (delta {d1 - d0:+.4f})")
    print(f" outputs -> {OUTPUT_DIR}/{stem}_pred.nii.gz (+ _pred0 / _gt / _confusion)")
    append_sweep_row(stem, mtr)
    print("Done!")


if __name__ == "__main__":
    main()
