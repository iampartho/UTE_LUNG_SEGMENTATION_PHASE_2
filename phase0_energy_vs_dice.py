"""
Phase 0 (no-retraining primary screening): does the V-JEPA mask-shape *energy*
rank lung masks by quality?

This script answers a single question before we build any guidance pipeline:
    "For a fixed scan, does a LOWER energy correspond to a BETTER mask?"
If yes (energy falls as Dice rises), the energy is a trustworthy compass and we
can later follow its gradient to refine masks. If not, we must fix the meter
first.

WHAT 'ENERGY' IS HERE
---------------------
We use the FRESH *mask-only* V-JEPA model trained in train_mask_jepa.py
(save_models/best_mask_jepa_ct256.pth, CT masks ONLY, fixed 256^3 input, patch 32
-> 8x8x8 = 512 tokens). The energy of a candidate mask is exactly the V-JEPA
training objective evaluated on that mask:
    1. preprocess the mask like dataset_mask_jepa.py
       (binarise -> CLEAN [keep major connected component(s) + fill holes,
       matching the encoder's training input] -> fit to 256^3 via the
       checkpoint's fit_mode [pad_crop or resize] -> Gaussian blur sigma=1.0),
    2. hide a random multi-block subset of the 8x8x8=512 patch tokens,
    3. let the encoder+predictor predict the hidden tokens' embeddings,
    4. measure the L1 mismatch against the EMA target encoder's embeddings,
    5. average that mismatch over NUM_MASK_SAMPLES *fixed* random hide-patterns
       (the SAME patterns are reused for every candidate so comparisons are fair).
Low energy  = "this looks like a predictable / plausible lung-mask shape".
High energy = "the hidden parts are hard to predict from the visible parts".

NOTE: this is still the MASK-ONLY (shape) energy used in the no-retrain primary
screening -- it does not look at the image. The cross-modal image->mask energy
(Phase 1 step 2) is a separate, later change.

HOW TO USE
----------
Edit the CONFIG block below (mainly MODE and MODALITY), then run the file. It
produces ONE figure per run, saved into OUT_DIR (and shown if a display exists).

MODES
-----
  'raw_scatter'        : energy vs Dice for each scan's ground-truth (GT) mask
                         and its segmentation-model prediction. First sanity look.
  'morph_sweep'        : take a few scans' GT and erode/dilate it across a range
                         of radii; plot energy vs Dice (and energy vs signed
                         radius). Tests whether energy is minimised AT the GT and
                         rises as the mask gets too small or too big.
  'random_aug_scatter' : apply assorted random degradations (shift/rotate/erode/
                         dilate/elastic/dropout/boundary-noise) to each scan's GT
                         to span the quality range; scatter energy vs Dice and
                         report Spearman correlation.

No argparse by design: just change the variables.
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")  # safe on headless machines; figures are saved to OUT_DIR
import matplotlib.pyplot as plt

from skimage.transform import resize as sk_resize
from scipy.ndimage import (
    gaussian_filter,
    binary_erosion,
    binary_dilation,
    generate_binary_structure,
    shift as nd_shift,
    rotate as nd_rotate,
    map_coordinates,
)
from scipy.stats import spearmanr

from model_vjepa import VJEPA, generate_masks, _gather_tokens
from dataset_mask_jepa import _center_pad_crop, _clean_binary
from basic_unet_disentangled import BasicUNet


# ===========================================================================
# CONFIG  -- change these, then run.
# ===========================================================================
MODE = "morph_sweep"          # 'raw_scatter' | 'morph_sweep' | 'random_aug_scatter'
MODALITY = "UTE"               # 'CT' | 'UTE'

# --- weights -----------------------------------------------------------------
# Fresh CT-only 256^3 mask encoder from train_mask_jepa.py (now retrained with
# mask cleaning + symmetric anisotropic-scale/elastic shape augmentation).
VJEPA_CKPT = "./save_models/latest_mask_jepa_ct256.pth"
SEG_CKPT = ("./save_models/current_best_may_2026/"
            "best_bunet_causality_paper_ct_train_UTE_test_w_tversky_wo_kl_only_gin_roughness_enforced.pth")

# How to fit a candidate mask to the encoder's 256^3 input. None = read from the
# checkpoint's saved 'fit_mode' (falls back to 'pad_crop'); set to 'pad_crop' or
# 'resize' to force. MUST match how the encoder was trained (dataset_mask_jepa.py).
FIT_MODE = None

# --- mask cleaning (match the encoder's training input distribution) ---------
# The retrained encoder cleans every mask (keep major connected component(s) +
# fill holes) before fitting, to strip the UTE speck/hole artifact. We mirror
# that here so the energy is computed on the same distribution the encoder saw.
# Cleaning is applied to EVERY candidate (GT, prediction, morphed, augmented)
# BEFORE both its Dice and its energy, so the two always describe one mask.
#   None  -> read from the checkpoint's saved 'aug' block (falls back to True)
#   True / False -> force on / off.
CLEAN_MASKS = None
CLEAN_KEEP_FRAC = None         # None -> ckpt aug.clean_keep_frac (falls back 0.10)

# Tag appended to output filenames so these plots do not overwrite the earlier
# (96^3 best_vjepa) screening plots.
RUN_TAG = "mask_jepa_ct256"

# --- data --------------------------------------------------------------------
# Both CT and UTE are taken from the V-JEPA *validation* CSV so the energy meter
# never saw these masks during its own training (unbiased screening). CT = the 69
# COPDgene scans (0 overlap with vjepa train); UTE = the 11 held-out AnatCorrLungs.
DATA_CSV = "./ids/only_ute_1.25mm.csv" #causality_test_ute_copd.csv"
CT_SUBSTR = "COPDgene"        # filepath marker for CT scans
UTE_SUBSTR = "UTE_new_data"   # filepath marker for UTE scans
# Optional path remap if the CSV roots differ from your run machine, e.g.
# ("/Shared/", "/Volumes/"). Leave None to use paths verbatim.
PATH_REPLACE = None

# Number of scans to use for raw_scatter / random_aug_scatter (None = all).
NUM_SCANS = None

# --- energy parameters (must mirror train_vjepa.py) --------------------------
GAUSSIAN_SIGMA = 1.0
NUM_MASK_SAMPLES = 16          # K hide-patterns averaged per energy value
MASK_SEED = 0                  # fixed -> identical hide-patterns for every mask
MASK_RATIO = 0.6
NUM_BLOCKS = 4
BLOCK_SCALE = (0.05, 0.25)
BLOCK_ASPECT = (0.75, 1.5)
TARGET_NORM = True

# --- segmentation prediction parameters (mirror datasets_causality.py) -------
NUM_DOUBLE_STRIDE = 4          # pad each dim up to a multiple of 2**this
PADVAL = -1.0                  # image pad value (images are normalised to [-1, 1])
SEG_THRESHOLD = 0.5

# --- morph_sweep parameters --------------------------------------------------
MORPH_RADII = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]  # <0 erode, 0 GT, >0 dilate
MORPH_NUM_SCANS = 5            # how many individual scans to draw (readability)

# --- random_aug_scatter parameters -------------------------------------------
RANDOM_AUG_PER_SCAN = 12
RANDOM_AUG_SEED = 0
# Which random degradations are allowed (each augmented sample randomly composes
# 1-3 of these). Drop 'elastic' if it is too slow on large volumes.
AUG_OPS = ["shift", "rotate", "erode", "dilate", "dropout", "boundary_noise", "elastic"]

# --- output ------------------------------------------------------------------
OUT_DIR = "./phase0_plots/new_aug_mask_model/prev_model/"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STRUCT3D = generate_binary_structure(3, 1)  # 6-connectivity for morphology

# Cleaning settings resolved from the checkpoint inside load_vjepa(); the CONFIG
# values above override them when not None.
_CLEAN_ENABLED = True
_CLEAN_KEEP_FRAC = 0.10


# ===========================================================================
# Small utilities
# ===========================================================================
def maybe_clean(mask):
    """Clean a candidate mask (keep major components + fill holes) iff enabled.

    Reuses dataset_mask_jepa._clean_binary so the cleaning is bit-for-bit what
    the encoder saw in training. Applied to every candidate (GT / prediction /
    morphed / augmented) before BOTH its Dice and its energy, so the quality
    label and the energy always refer to the identical mask.
    """
    if _CLEAN_ENABLED:
        return _clean_binary(mask, keep_frac=_CLEAN_KEEP_FRAC)
    return mask.astype(np.float32)


def resolve_path(p):
    if PATH_REPLACE is not None:
        p = p.replace(PATH_REPLACE[0], PATH_REPLACE[1])
    return p


def list_scans(modality):
    df = pd.read_csv(DATA_CSV)
    substr = CT_SUBSTR if modality == "CT" else UTE_SUBSTR
    files = [f for f in df["filepaths"].values if substr in f]
    files = [resolve_path(f) for f in files]
    if NUM_SCANS is not None:
        files = files[:NUM_SCANS]
    return files


def load_volume(path):
    """Return (image_native, gt_mask_native) both float32, gt binarised 0/1."""
    arr = np.load(path)
    img = arr[:, :, :, 0].astype(np.float32)
    gt = arr[:, :, :, 1].astype(np.float32)
    gt = (gt > 0).astype(np.float32)
    return img, gt


def dice(a, b):
    """Dice overlap between two binary volumes (1.0 = identical)."""
    a = a > 0.5
    b = b > 0.5
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0  # both empty -> treat as perfect agreement
    return float(2.0 * inter / denom)


# ===========================================================================
# Image normalisation (per modality) for the segmentation model input
# ===========================================================================
def _normalise_zero_one(x):
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else x * 0.0


def _normalise_one_one(x):
    return _normalise_zero_one(x) * 2.0 - 1.0


def _normalise_hu(x, hu_range=(-1024.0, 200.0)):
    return np.clip(x, hu_range[0], hu_range[1]).astype(np.float32)


def normalise_image(img, modality):
    if modality == "CT":
        return _normalise_one_one(_normalise_hu(img))
    return _normalise_one_one(img)  # UTE


# ===========================================================================
# V-JEPA energy
# ===========================================================================
def load_vjepa():
    ckpt = torch.load(VJEPA_CKPT, map_location=DEVICE)
    cfg = ckpt["config"]
    model = VJEPA(**cfg).to(DEVICE)
    model.encoder.load_state_dict(ckpt["encoder"])
    model.predictor.load_state_dict(ckpt["predictor"])
    model.target_encoder.load_state_dict(ckpt["target_encoder"])
    model.eval()
    input_size = tuple(cfg["input_size"])
    grid = model.encoder.grid
    fit_mode = FIT_MODE if FIT_MODE is not None else ckpt.get("fit_mode", "pad_crop")

    # Resolve mask-cleaning from the checkpoint's 'aug' provenance (CONFIG overrides).
    global _CLEAN_ENABLED, _CLEAN_KEEP_FRAC
    aug = ckpt.get("aug", {}) or {}
    _CLEAN_ENABLED = aug.get("clean_masks", True) if CLEAN_MASKS is None else CLEAN_MASKS
    _CLEAN_KEEP_FRAC = (aug.get("clean_keep_frac", 0.10)
                        if CLEAN_KEEP_FRAC is None else CLEAN_KEEP_FRAC)

    print(f"[vjepa] loaded {VJEPA_CKPT} | input_size={input_size} grid={grid} "
          f"tokens={model.encoder.num_patches} | fit_mode={fit_mode}")
    print(f"[vjepa] mask cleaning: enabled={_CLEAN_ENABLED} keep_frac={_CLEAN_KEEP_FRAC}")
    return model, input_size, grid, fit_mode


def precompute_mask_patterns(grid):
    """K fixed (keep_idx, mask_idx) hide-patterns, reused for every candidate."""
    rng = np.random.RandomState(MASK_SEED)
    patterns = []
    for _ in range(NUM_MASK_SAMPLES):
        keep_idx, mask_idx = generate_masks(
            1, grid,
            mask_ratio=MASK_RATIO,
            num_blocks=NUM_BLOCKS,
            block_scale=BLOCK_SCALE,
            block_aspect=BLOCK_ASPECT,
            device=DEVICE,
            rng=rng,
        )
        patterns.append((keep_idx, mask_idx))
    return patterns


def preprocess_mask_for_energy(mask_native, fit_mode, input_size):
    """Mirror dataset_mask_jepa.py: binarise -> fit to input_size -> Gaussian blur.

    fit_mode 'pad_crop' centre-crops/pads at native scale (preserves 1.25mm
    voxels); 'resize' rescales the whole volume. UTE is natively 256^3 so both
    modes are a no-op for it.
    """
    m = (mask_native > 0.5).astype(np.float32)
    m = _normalise_zero_one(m).astype(np.float32)
    if tuple(m.shape) != tuple(input_size):
        if fit_mode == "pad_crop":
            m = _center_pad_crop(m, input_size, pad_value=0.0).astype(np.float32)
        else:
            m = sk_resize(m, input_size, order=0, preserve_range=True,
                          anti_aliasing=False).astype(np.float32)
    if GAUSSIAN_SIGMA > 0.0:
        m = gaussian_filter(m, sigma=GAUSSIAN_SIGMA).astype(np.float32)
        m = np.clip(m, 0.0, 1.0)
    return torch.from_numpy(m).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,*input_size)


@torch.no_grad()
def compute_energy(model, mask_native, fit_mode, input_size, patterns):
    """Average masked-token L1 over the fixed hide-patterns.

    The EMA target-encoder forward is independent of the hide-pattern, so we run
    it ONCE and gather/normalise per pattern (numerically identical to calling
    model(...) per pattern, but avoids re-running the full target encoder 16x --
    a meaningful saving at 256^3).
    """
    x = preprocess_mask_for_energy(mask_native, fit_mode, input_size)
    grid = model.encoder.grid_of(x)
    target_full = model.target_encoder(x, keep_idx=None)  # (1, N, dim)

    total = 0.0
    for keep_idx, mask_idx in patterns:
        ctx = model.encoder(x, keep_idx=keep_idx)
        pred = model.predictor(ctx, keep_idx, mask_idx, grid)
        target = _gather_tokens(target_full, mask_idx)
        if TARGET_NORM:
            target = F.layer_norm(target, (target.shape[-1],))
        total += F.l1_loss(pred, target).item()
    return total / len(patterns)


# ===========================================================================
# Segmentation model prediction
# ===========================================================================
def load_seg_model():
    model = BasicUNet().to(DEVICE)
    state = torch.load(SEG_CKPT, map_location=DEVICE)
    res = model.load_state_dict(state, strict=False)
    if res.missing_keys:
        print(f"[seg] missing keys ({len(res.missing_keys)}): {res.missing_keys[:6]}...")
    if res.unexpected_keys:
        print(f"[seg] unexpected keys ({len(res.unexpected_keys)}): {res.unexpected_keys[:6]}...")
    model.eval()
    print(f"[seg] loaded {SEG_CKPT}")
    return model


@torch.no_grad()
def predict_mask(seg_model, img_native, modality):
    """Run the U-Net and return a native-resolution binary prediction."""
    img = normalise_image(img_native, modality)
    h, w, d = img.shape

    mul = 2 ** NUM_DOUBLE_STRIDE
    nh, nw, nd = (mul * int(np.ceil(s / mul)) for s in (h, w, d))
    img = np.pad(img, ((0, nh - h), (0, nw - w), (0, nd - d)),
                 mode="constant", constant_values=PADVAL)

    x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)
    logits = seg_model(x)
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    pred = (prob > SEG_THRESHOLD).astype(np.float32)
    return pred[:h, :w, :d]  # crop padding away


# ===========================================================================
# Mask degradations
# ===========================================================================
def morph_mask(mask_native, radius):
    """radius<0 erode |radius| iters, radius>0 dilate, radius==0 identity."""
    m = mask_native > 0.5
    if radius < 0:
        m = binary_erosion(m, structure=STRUCT3D, iterations=-radius)
    elif radius > 0:
        m = binary_dilation(m, structure=STRUCT3D, iterations=radius)
    return m.astype(np.float32)


def _aug_shift(m, rng):
    offs = rng.randint(-15, 16, size=3)
    return (nd_shift(m, offs, order=0, mode="constant", cval=0.0) > 0.5).astype(np.float32)


def _aug_rotate(m, rng):
    angle = rng.uniform(-15, 15)
    axes = tuple(rng.choice([0, 1, 2], size=2, replace=False).tolist())
    return (nd_rotate(m, angle, axes=axes, reshape=False, order=0,
                      mode="constant", cval=0.0) > 0.5).astype(np.float32)


def _aug_erode(m, rng):
    return binary_erosion(m > 0.5, structure=STRUCT3D, iterations=rng.randint(1, 4)).astype(np.float32)


def _aug_dilate(m, rng):
    return binary_dilation(m > 0.5, structure=STRUCT3D, iterations=rng.randint(1, 4)).astype(np.float32)


def _aug_dropout(m, rng):
    """Zero out a random cuboid (simulating a dropped lobe / missing region)."""
    out = (m > 0.5).astype(np.float32).copy()
    coords = np.argwhere(out > 0.5)
    if len(coords) == 0:
        return out
    c = coords[rng.randint(len(coords))]
    shp = np.array(out.shape)
    half = (shp * rng.uniform(0.10, 0.25)).astype(int)
    lo = np.clip(c - half, 0, shp)
    hi = np.clip(c + half, 0, shp)
    out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = 0.0
    return out


def _aug_boundary_noise(m, rng):
    """Randomly flip voxels in the thin band around the boundary."""
    mb = m > 0.5
    band = np.logical_xor(binary_dilation(mb, STRUCT3D, 1), binary_erosion(mb, STRUCT3D, 1))
    flip = np.logical_and(band, rng.rand(*m.shape) < 0.5)
    out = mb.copy()
    out[flip] = ~out[flip]
    return out.astype(np.float32)


def _aug_elastic(m, rng, sigma=6.0, alpha=20.0):
    shp = m.shape
    disp = [gaussian_filter((rng.rand(*shp) * 2 - 1), sigma) * alpha for _ in range(3)]
    zz, yy, xx = np.meshgrid(np.arange(shp[0]), np.arange(shp[1]),
                             np.arange(shp[2]), indexing="ij")
    coords = [np.reshape(zz + disp[0], -1),
              np.reshape(yy + disp[1], -1),
              np.reshape(xx + disp[2], -1)]
    warped = map_coordinates(m, coords, order=0, mode="constant", cval=0.0).reshape(shp)
    return (warped > 0.5).astype(np.float32)


_AUG_FUNCS = {
    "shift": _aug_shift,
    "rotate": _aug_rotate,
    "erode": _aug_erode,
    "dilate": _aug_dilate,
    "dropout": _aug_dropout,
    "boundary_noise": _aug_boundary_noise,
    "elastic": _aug_elastic,
}


def random_augment(mask_native, rng):
    """Compose 1-3 random ops (from AUG_OPS) to produce a degraded mask."""
    n_ops = rng.randint(1, 4)
    ops = rng.choice(AUG_OPS, size=n_ops, replace=False)
    m = (mask_native > 0.5).astype(np.float32)
    for op in ops:
        m = _AUG_FUNCS[op](m, rng)
    return m, list(ops)


# ===========================================================================
# Plot builders
# ===========================================================================
def _ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)


def run_raw_scatter():
    model, input_size, grid, fit_mode = load_vjepa()
    patterns = precompute_mask_patterns(grid)
    seg_model = load_seg_model()
    files = list_scans(MODALITY)
    print(f"[raw_scatter] {MODALITY}: {len(files)} scans")

    gt_e, pred_e, pred_d = [], [], []
    for i, f in enumerate(files):
        img, gt = load_volume(f)
        gt = maybe_clean(gt)
        gt_e.append(compute_energy(model, gt, fit_mode, input_size, patterns))
        pred = maybe_clean(predict_mask(seg_model, img, MODALITY))
        pred_e.append(compute_energy(model, pred, fit_mode, input_size, patterns))
        pred_d.append(dice(pred, gt))
        print(f"  [{i+1}/{len(files)}] {os.path.basename(f)} "
              f"gt_E={gt_e[-1]:.4f} pred_E={pred_e[-1]:.4f} pred_Dice={pred_d[-1]:.3f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter([1.0] * len(gt_e), gt_e, c="tab:green", marker="*", s=120,
               label="GT mask (Dice=1)", zorder=3)
    ax.scatter(pred_d, pred_e, c="tab:red", marker="o", s=45,
               label="seg prediction", alpha=0.8, zorder=2)
    ax.set_xlabel("Dice vs GT  (mask quality, higher = better)")
    ax.set_ylabel("V-JEPA energy  (lower = more plausible shape)")
    ax.set_title(f"Phase 0 raw scatter | {MODALITY} | n={len(files)}\n"
                 f"mean GT energy={np.mean(gt_e):.4f}, mean pred energy={np.mean(pred_e):.4f}, "
                 f"mean pred Dice={np.mean(pred_d):.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = os.path.join(OUT_DIR, f"raw_scatter_{MODALITY}_{RUN_TAG}.png")
    # make the plot x axis range from 0 to 1
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # ax.set_xticks([0, 1]) 
    # ax.set_yticks([0, 1])
    # ax.set_xticklabels(["0", "1"]) # set the x axis labels to 0 and 1
    # ax.set_yticklabels(["0", "1"]) # set the y axis labels to 0 and 1
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[saved] {out}")


def run_morph_sweep():
    model, input_size, grid, fit_mode = load_vjepa()
    patterns = precompute_mask_patterns(grid)
    files = list_scans(MODALITY)[:MORPH_NUM_SCANS]
    print(f"[morph_sweep] {MODALITY}: {len(files)} scans, radii={MORPH_RADII}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.get_cmap("viridis")
    per_scan_rho = []

    for si, f in enumerate(files):
        _, gt = load_volume(f)
        gt = maybe_clean(gt)            # clean reference: morphs start from this
        dices, energies = [], []
        for r in MORPH_RADII:
            m = maybe_clean(morph_mask(gt, r))
            dices.append(dice(m, gt))
            energies.append(compute_energy(model, m, fit_mode, input_size, patterns))
        color = cmap(si / max(1, len(files) - 1))
        label = os.path.basename(f)[:18]
        axes[0].plot(dices, energies, "-o", color=color, label=label, alpha=0.8)
        axes[1].plot(MORPH_RADII, energies, "-o", color=color, label=label, alpha=0.8)
        rho, _ = spearmanr(dices, energies)
        per_scan_rho.append(rho)
        print(f"  {label}: per-scan Spearman(energy,Dice)={rho:.3f}")

    axes[0].set_xlabel("Dice vs GT")
    axes[0].set_ylabel("V-JEPA energy")
    axes[0].set_title("energy vs Dice (want: downhill)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)

    axes[1].axvline(0.0, color="k", ls="--", alpha=0.5)
    axes[1].set_xlabel("signed morph radius  (<0 erode, 0 = GT, >0 dilate)")
    axes[1].set_ylabel("V-JEPA energy")
    axes[1].set_title("energy vs radius (want: minimum at 0)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Phase 0 morph sweep | {MODALITY} | "
                 f"median per-scan Spearman={np.nanmedian(per_scan_rho):.3f} "
                 f"(want <= -0.7)")
    out = os.path.join(OUT_DIR, f"morph_sweep_{MODALITY}_{RUN_TAG}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[saved] {out}")


def run_random_aug_scatter():
    model, input_size, grid, fit_mode = load_vjepa()
    patterns = precompute_mask_patterns(grid)
    files = list_scans(MODALITY)
    print(f"[random_aug_scatter] {MODALITY}: {len(files)} scans x {RANDOM_AUG_PER_SCAN} augs")

    all_d, all_e = [], []
    gt_e = []
    per_scan_rho = []
    for si, f in enumerate(files):
        _, gt = load_volume(f)
        gt = maybe_clean(gt)           # clean reference: augmentations start from this
        gt_e.append(compute_energy(model, gt, fit_mode, input_size, patterns))
        rng = np.random.RandomState(RANDOM_AUG_SEED + si * 1000)
        sd, se = [], []
        for ai in range(RANDOM_AUG_PER_SCAN):
            m, _ops = random_augment(gt, rng)
            m = maybe_clean(m)
            sd.append(dice(m, gt))
            se.append(compute_energy(model, m, fit_mode, input_size, patterns))
        all_d.extend(sd)
        all_e.extend(se)
        if len(set(np.round(sd, 4))) > 1:
            rho, _ = spearmanr(sd, se)
            per_scan_rho.append(rho)
        print(f"  [{si+1}/{len(files)}] {os.path.basename(f)} gt_E={gt_e[-1]:.4f}")

    pooled_rho, _ = spearmanr(all_d, all_e)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(all_d, all_e, c="tab:blue", s=25, alpha=0.5, label="random degradations")
    ax.scatter([1.0] * len(gt_e), gt_e, c="tab:green", marker="*", s=120,
               label="GT mask (Dice=1)", zorder=3)
    ax.set_xlabel("Dice vs GT  (mask quality, higher = better)")
    ax.set_ylabel("V-JEPA energy  (lower = more plausible shape)")
    ax.set_title(f"Phase 0 random-aug scatter | {MODALITY} | "
                 f"n={len(files)}x{RANDOM_AUG_PER_SCAN}\n"
                 f"pooled Spearman={pooled_rho:.3f}, "
                 f"median per-scan Spearman={np.nanmedian(per_scan_rho):.3f} (want <= -0.7)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = os.path.join(OUT_DIR, f"random_aug_scatter_{MODALITY}_{RUN_TAG}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[saved] {out}")


# ===========================================================================
def main():
    _ensure_outdir()
    print(f"device={DEVICE} | MODE={MODE} | MODALITY={MODALITY}")
    if MODE == "raw_scatter":
        run_raw_scatter()
    elif MODE == "morph_sweep":
        run_morph_sweep()
    elif MODE == "random_aug_scatter":
        run_random_aug_scatter()
    else:
        raise ValueError(f"Unknown MODE={MODE!r}. "
                         f"Use 'raw_scatter', 'morph_sweep' or 'random_aug_scatter'.")


if __name__ == "__main__":
    main()
