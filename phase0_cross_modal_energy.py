"""
Phase 0 (DECISION-GRADE): does the CROSS-MODAL energy rank lung masks by quality?

This is the sibling of phase0_energy_vs_dice.py. That script screens the
MASK-ONLY (shape) energy from the Step-1 mask encoder; THIS script evaluates the
CROSS-MODAL "two-witnesses" energy built in Phase 1 step 2 -- the one Phase 2
actually steers. Keep the two scripts separate so the mask-only screen stays
available (just run phase0_energy_vs_dice.py).

WHAT 'ENERGY' IS HERE
---------------------
For a scan image x and a candidate mask m:
    E_x(x, m) = || P(E_img(x)) - E_mask(m) ||_1
where E_img + P are the trained image branch (train_img_jepa.py) and E_mask is the
frozen Step-1 mask encoder. The image side anchors *where / how big* the lungs
belong for THIS scan; the mask side says what shape THIS mask has.

Mechanical differences from the mask-only screen (see the project plan):
    * Energy takes TWO inputs (image + mask), not one.
    * DETERMINISTIC: no token hiding, no K-pattern averaging. E_img(x) -> P gives
      the full mask-token field in one shot; the mask gives the target field; L1.
    * Image side: real image, NO GIN (GIN is train-time only). This is also the
      first honest test of whether the GIN-trained image branch transfers to true
      UTE appearance -- the single biggest scientific risk.
    * Mask side: SAME preprocessing as Step 1 (clean -> fit 256^3 -> blur sigma 1).
    * Target LayerNorm matches training, so the evaluated energy == the trained
      objective == what Phase 2 will descend.

MODES
-----
  'raw_scatter'        : E_x vs Dice for each scan's GT mask and its seg
                         prediction. CT watch-scans (the emphysema cases that
                         inverted under the shape-only meter) are annotated.
  'morph_sweep'        : erode/dilate each scan's GT across radii; E_x vs Dice and
                         vs signed radius. Want minimum AT the GT (radius 0).
  'random_aug_scatter' : random degradations of GT spanning the quality range;
                         scatter E_x vs Dice + Spearman.
  'pred_morph_sweep'   : NEW. Start from each scan's PREDICTION m0 (Dice ~0.85-0.95
                         band) and walk a path toward GT (and a little away from
                         it); check E_x decreases monotonically toward GT. This is
                         the EXACT regime Phase 2 operates in (start from a real
                         prediction, not from GT).
  'gt_vs_pred'         : NEW. Per UTE scan, compare E_x(GT) vs E_x(seg prediction)
                         and report the % of scans where the prediction's energy is
                         NOT lower than GT's (i.e. GT is the per-scan minimum, so
                         descending E_x cannot pull a good mask away from GT). The
                         per-scan complement to raw_scatter / pred_morph_sweep: it
                         quantifies the "GT isn't always the energy minimum" risk
                         and tells Phase 2 how hard the early-stop/leash must work.

GO to Phase 2 (on UTE, with E_x): (a) morph minimum at radius 0; (b) median
per-scan Spearman <= -0.7; (c) pred_morph_sweep monotonically decreasing from m0
toward GT in the 0.85-0.95 band. NO-GO -> the fix lives in Phase 1 step 2 (image
branch not transferring), NOT in Phase 2.

No argparse by design: edit the CONFIG block, then run.
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

from model_img_jepa import ImgJEPA
from dataset_mask_jepa import _center_pad_crop, _clean_binary
from basic_unet_disentangled import BasicUNet


# ===========================================================================
# CONFIG  -- change these, then run.
# ===========================================================================
MODE = "gt_vs_pred"      # 'raw_scatter'|'morph_sweep'|'random_aug_scatter'|'pred_morph_sweep'|'gt_vs_pred'
MODALITY = "UTE"               # 'CT' | 'UTE'

# --- weights -----------------------------------------------------------------
# Phase 1 step 2 image-JEPA checkpoint (carries E_img + P + frozen E_mask).
IMG_JEPA_CKPT = "./save_models/best_img_jepa_ct256_new_aug.pth"
SEG_CKPT = ("./save_models/current_best_may_2026/"
            "best_bunet_causality_paper_ct_train_UTE_test_w_tversky_wo_kl_only_gin_roughness_enforced.pth")

# How to fit a candidate (image AND mask) to the encoder's 256^3 input. Step-2
# training used a centre pad/crop (preserve 1.25mm scale), so default 'pad_crop'.
# UTE is natively 256^3 -> both modes are a no-op for it.
FIT_MODE = "pad_crop"          # 'pad_crop' | 'resize'

# --- mask cleaning (match the mask encoder's training input distribution) ----
# Applied to EVERY candidate mask (GT / prediction / morphed / augmented) BEFORE
# both its Dice and its energy, so the quality label and the energy refer to the
# identical mask. None -> read from the checkpoint's saved 'aug' block.
CLEAN_MASKS = None
CLEAN_KEEP_FRAC = None

# Tag appended to output filenames.
RUN_TAG = "img_jepa_ct256"

# --- data --------------------------------------------------------------------
DATA_CSV = "./ids/only_ute_1.25mm.csv"
CT_SUBSTR = "COPDgene"         # filepath marker for CT scans
UTE_SUBSTR = "UTE_new_data"    # filepath marker for UTE scans
PATH_REPLACE = None            # optional (old, new) CSV path remap; None = verbatim
NUM_SCANS = None               # scans for raw_scatter / random_aug_scatter (None = all)

# CT emphysema cases that inverted under the shape-only meter -- annotated in
# raw_scatter so we can check whether image conditioning rescues them.
WATCH_SCANS = ["18065W_FRC", "10993X_RV"]

# --- energy parameters -------------------------------------------------------
GAUSSIAN_SIGMA = 1.0           # mask softening (match E_mask training input)
TARGET_NORM = True             # LayerNorm the E_mask target (match training)
PADVAL = -1.0                  # image pad value (images normalised to [-1, 1])

# --- segmentation prediction parameters (mirror datasets_causality.py) -------
NUM_DOUBLE_STRIDE = 4          # pad each dim up to a multiple of 2**this
SEG_THRESHOLD = 0.5

# --- morph_sweep parameters --------------------------------------------------
MORPH_RADII = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]  # <0 erode, 0 GT, >0 dilate
MORPH_NUM_SCANS = 15            # how many individual scans to draw (readability)

# --- random_aug_scatter parameters -------------------------------------------
RANDOM_AUG_PER_SCAN = 12
RANDOM_AUG_SEED = 0
AUG_OPS = ["shift", "rotate", "erode", "dilate", "dropout", "boundary_noise", "elastic"]

# --- pred_morph_sweep parameters ---------------------------------------------
# A path from the prediction m0 toward GT (t in [0, 1]) and a little away from it
# (t in [-AWAY_FRAC, 0), blending m0 toward a degraded copy of itself). The blend
# is a smooth interpolation of the two masks' soft fields, re-binarised.
PRED_MORPH_NUM_SCANS = 15
PRED_MORPH_STEPS_TOWARD = 6    # t = linspace(0, 1, this)
PRED_MORPH_STEPS_AWAY = 3      # extra points with t < 0 (m0 -> degraded m0)
PRED_MORPH_AWAY_FRAC = 0.5     # most-negative t
PRED_MORPH_BLEND_SIGMA = 2.0   # smoothing for the blend interpolation
PRED_MORPH_DEGRADE_RADIUS = 3  # dilation radius defining the "away" endpoint

# --- output ------------------------------------------------------------------
OUT_DIR = "./phase0_plots/cross_modal/final/"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STRUCT3D = generate_binary_structure(3, 1)  # 6-connectivity for morphology

# Cleaning settings resolved from the checkpoint inside load_model(); the CONFIG
# values above override them when not None.
_CLEAN_ENABLED = True
_CLEAN_KEEP_FRAC = 0.10


# ===========================================================================
# Small utilities
# ===========================================================================
def maybe_clean(mask):
    """Clean a candidate mask (keep major components + fill holes) iff enabled."""
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
    a = a > 0.5
    b = b > 0.5
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


# ===========================================================================
# Image normalisation (per modality)
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
# Cross-modal energy
# ===========================================================================

def get_bad_filepaths():
    """Get the filepaths of the scans with dice < 0.92."""
    results = pd.read_csv("test_result_csv/bunet_gin_only_with_roughness_enforced_td1_current_best_model_marissa_data.csv")
    filepaths = pd.read_csv("ids/only_ute_1.25mm.csv")["filepaths"].tolist()

    low_dice = results[results["dice"] < 0.90]
    indices = low_dice["sid"].str.split("_").str[-1].astype(int)

    bad_filepaths = [filepaths[i] for i in indices]
    return bad_filepaths
    
def load_model():
    """Load the Phase 1 step 2 image-JEPA (E_img + P + frozen E_mask)."""
    model, cfg = ImgJEPA.from_checkpoint(IMG_JEPA_CKPT, DEVICE)
    input_size = tuple(model.mask_cfg["input_size"])
    grid = model.encoder.grid

    # Resolve mask cleaning from the checkpoint's 'aug' provenance (CONFIG overrides).
    ckpt = torch.load(IMG_JEPA_CKPT, map_location="cpu")
    aug = ckpt.get("aug", {}) or {}
    global _CLEAN_ENABLED, _CLEAN_KEEP_FRAC
    _CLEAN_ENABLED = aug.get("clean_masks", True) if CLEAN_MASKS is None else CLEAN_MASKS
    _CLEAN_KEEP_FRAC = (aug.get("clean_keep_frac", 0.10)
                        if CLEAN_KEEP_FRAC is None else CLEAN_KEEP_FRAC)

    print(f"[img-jepa] loaded {IMG_JEPA_CKPT} | input_size={input_size} grid={grid} "
          f"tokens={model.encoder.num_patches} | fit_mode={FIT_MODE}")
    print(f"[img-jepa] mask cleaning: enabled={_CLEAN_ENABLED} keep_frac={_CLEAN_KEEP_FRAC}")
    return model, input_size


def _fit_to_input(vol, input_size, pad_value, order):
    """Fit a native volume to ``input_size`` (centre pad/crop or resize)."""
    if tuple(vol.shape) == tuple(input_size):
        return vol.astype(np.float32)
    if FIT_MODE == "pad_crop":
        return _center_pad_crop(vol, input_size, pad_value=pad_value).astype(np.float32)
    return sk_resize(vol, input_size, order=order, preserve_range=True,
                     anti_aliasing=False).astype(np.float32)


def preprocess_image(img_native, modality, input_size):
    """Normalise (per modality) then fit to 256^3 (pad with PADVAL). NO GIN."""
    img = normalise_image(img_native, modality)
    img = _fit_to_input(img, input_size, pad_value=PADVAL, order=1)
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)


def preprocess_mask(mask_native, input_size):
    """Mirror dataset_mask_jepa.py: binarise -> fit 256^3 -> Gaussian blur."""
    m = (mask_native > 0.5).astype(np.float32)
    m = _fit_to_input(m, input_size, pad_value=0.0, order=0)
    if GAUSSIAN_SIGMA > 0.0:
        m = gaussian_filter(m, sigma=GAUSSIAN_SIGMA).astype(np.float32)
        m = np.clip(m, 0.0, 1.0)
    return torch.from_numpy(m).unsqueeze(0).unsqueeze(0).to(DEVICE)


@torch.no_grad()
def image_pred_latent(model, img_native, modality, input_size):
    """P(E_img(x)) for a scan -- the image's predicted mask-token field (1, N, dim).

    Computed ONCE per scan and reused across every candidate mask (the image side
    of E_x does not depend on the mask), exactly mirroring how the mask-only
    screen ran its target encoder once per scan.
    """
    x = preprocess_image(img_native, modality, input_size)
    grid = model.encoder.grid_of(x)
    img_tokens = model.encoder(x, keep_idx=None)
    return model.predictor(img_tokens, grid)            # (1, N, dim)


@torch.no_grad()
def mask_target_latent(model, mask_native, input_size):
    """E_mask(m) (LayerNorm'd) -- the candidate mask's token field (1, N, dim)."""
    m = preprocess_mask(mask_native, input_size)
    target = model.mask_encoder(m, keep_idx=None)
    if TARGET_NORM:
        target = F.layer_norm(target, (target.shape[-1],))
    return target


def energy_from_latents(pred, target):
    """E_x = ||P(E_img(x)) - E_mask(m)||_1 (mean over tokens & features)."""
    return F.l1_loss(pred, target).item()


@torch.no_grad()
def compute_energy(model, img_pred, mask_native, input_size):
    """Cross-modal energy for one (precomputed image latent, candidate mask)."""
    target = mask_target_latent(model, mask_native, input_size)
    return energy_from_latents(img_pred, target)


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
    return pred[:h, :w, :d]


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


def blend_masks(a, b, t, sigma=2.0):
    """Smoothly interpolate binary masks a->b by t in [0,1], re-binarised.

    Blurring both masks and taking a convex combination of the soft fields gives
    a continuous morph whose Dice to ``b`` increases monotonically with t (t=0 is
    a, t=1 is b). Used to build a path from a prediction toward its GT.
    """
    fa = gaussian_filter((a > 0.5).astype(np.float32), sigma)
    fb = gaussian_filter((b > 0.5).astype(np.float32), sigma)
    return (((1.0 - t) * fa + t * fb) > 0.5).astype(np.float32)


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
    model, input_size = load_model()
    seg_model = load_seg_model()
    files = list_scans(MODALITY)
    print(f"[raw_scatter] {MODALITY}: {len(files)} scans")

    gt_e, pred_e, pred_d, names = [], [], [], []
    for i, f in enumerate(files):
        img, gt = load_volume(f)
        img_pred = image_pred_latent(model, img, MODALITY, input_size)  # per-scan, once
        gt = maybe_clean(gt)
        gt_e.append(compute_energy(model, img_pred, gt, input_size))
        pred = maybe_clean(predict_mask(seg_model, img, MODALITY))
        pred_e.append(compute_energy(model, img_pred, pred, input_size))
        pred_d.append(dice(pred, gt))
        names.append(os.path.basename(f).replace(".npy", ""))
        print(f"  [{i+1}/{len(files)}] {names[-1]} "
              f"gt_E={gt_e[-1]:.4f} pred_E={pred_e[-1]:.4f} pred_Dice={pred_d[-1]:.3f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter([1.0] * len(gt_e), gt_e, c="tab:green", marker="*", s=120,
               label="GT mask (Dice=1)", zorder=3)
    ax.scatter(pred_d, pred_e, c="tab:red", marker="o", s=45,
               label="seg prediction", alpha=0.8, zorder=2)
    # Annotate the watch-scans (emphysema cases that inverted under shape-only).
    for nm, dd, ee in zip(names, pred_d, pred_e):
        if any(w in nm for w in WATCH_SCANS):
            ax.annotate(nm, (dd, ee), fontsize=7, color="black",
                        xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Dice vs GT  (mask quality, higher = better)")
    ax.set_ylabel("Cross-modal energy E_x  (lower = mask fits the scan)")
    ax.set_title(f"Phase 0 cross-modal raw scatter | {MODALITY} | n={len(files)}\n"
                 f"mean GT E_x={np.mean(gt_e):.4f}, mean pred E_x={np.mean(pred_e):.4f}, "
                 f"mean pred Dice={np.mean(pred_d):.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = os.path.join(OUT_DIR, f"raw_scatter_{MODALITY}_{RUN_TAG}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[saved] {out}")


def run_morph_sweep():
    model, input_size = load_model()
    files = list_scans(MODALITY)[:MORPH_NUM_SCANS]
    print(f"[morph_sweep] {MODALITY}: {len(files)} scans, radii={MORPH_RADII}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.get_cmap("viridis")
    per_scan_rho = []

    for si, f in enumerate(files):
        img, gt = load_volume(f)
        img_pred = image_pred_latent(model, img, MODALITY, input_size)  # image fixed per scan
        gt = maybe_clean(gt)
        dices, energies = [], []
        for r in MORPH_RADII:
            m = maybe_clean(morph_mask(gt, r))
            dices.append(dice(m, gt))
            energies.append(compute_energy(model, img_pred, m, input_size))
        color = cmap(si / max(1, len(files) - 1))
        label = os.path.basename(f)[:18]
        axes[0].plot(dices, energies, "-o", color=color, label=label, alpha=0.8)
        axes[1].plot(MORPH_RADII, energies, "-o", color=color, label=label, alpha=0.8)
        rho, _ = spearmanr(dices, energies)
        per_scan_rho.append(rho)
        print(f"  {label}: per-scan Spearman(E_x,Dice)={rho:.3f}")

    axes[0].set_xlabel("Dice vs GT")
    axes[0].set_ylabel("Cross-modal energy E_x")
    axes[0].set_title("E_x vs Dice (want: downhill)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)

    axes[1].axvline(0.0, color="k", ls="--", alpha=0.5)
    axes[1].set_xlabel("signed morph radius  (<0 erode, 0 = GT, >0 dilate)")
    axes[1].set_ylabel("Cross-modal energy E_x")
    axes[1].set_title("E_x vs radius (want: minimum at 0)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Phase 0 cross-modal morph sweep | {MODALITY} | "
                 f"median per-scan Spearman={np.nanmedian(per_scan_rho):.3f} "
                 f"(want <= -0.7)")
    out = os.path.join(OUT_DIR, f"morph_sweep_{MODALITY}_{RUN_TAG}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[saved] {out}")


def run_random_aug_scatter():
    model, input_size = load_model()
    files = list_scans(MODALITY)
    print(f"[random_aug_scatter] {MODALITY}: {len(files)} scans x {RANDOM_AUG_PER_SCAN} augs")

    all_d, all_e = [], []
    gt_e = []
    per_scan_rho = []
    for si, f in enumerate(files):
        img, gt = load_volume(f)
        img_pred = image_pred_latent(model, img, MODALITY, input_size)
        gt = maybe_clean(gt)
        gt_e.append(compute_energy(model, img_pred, gt, input_size))
        rng = np.random.RandomState(RANDOM_AUG_SEED + si * 1000)
        sd, se = [], []
        for ai in range(RANDOM_AUG_PER_SCAN):
            m, _ops = random_augment(gt, rng)
            m = maybe_clean(m)
            sd.append(dice(m, gt))
            se.append(compute_energy(model, img_pred, m, input_size))
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
    ax.set_ylabel("Cross-modal energy E_x  (lower = mask fits the scan)")
    ax.set_title(f"Phase 0 cross-modal random-aug scatter | {MODALITY} | "
                 f"n={len(files)}x{RANDOM_AUG_PER_SCAN}\n"
                 f"pooled Spearman={pooled_rho:.3f}, "
                 f"median per-scan Spearman={np.nanmedian(per_scan_rho):.3f} (want <= -0.7)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = os.path.join(OUT_DIR, f"random_aug_scatter_{MODALITY}_{RUN_TAG}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[saved] {out}")


def run_pred_morph_sweep():
    """Start from each scan's PREDICTION m0 and walk toward (and a bit away from)
    GT; check E_x falls monotonically toward GT -- the exact Phase-2 regime."""
    model, input_size = load_model()
    seg_model = load_seg_model()
    files = get_bad_filepaths()[:PRED_MORPH_NUM_SCANS] #get_bad_filepaths()[:PRED_MORPH_NUM_SCANS] #list_scans(MODALITY)[:PRED_MORPH_NUM_SCANS]
    print(f"[pred_morph_sweep] {MODALITY}: {len(files)} scans")

    # t < 0 : m0 -> degraded m0 (away from GT). t in [0, 1] : m0 -> GT (toward GT).
    t_away = list(np.linspace(-PRED_MORPH_AWAY_FRAC, 0.0, PRED_MORPH_STEPS_AWAY + 1)[:-1])
    t_toward = list(np.linspace(0.0, 1.0, PRED_MORPH_STEPS_TOWARD))
    t_values = t_away + t_toward

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.get_cmap("viridis")
    per_scan_rho = []           # Spearman over the TOWARD-GT portion (the decisive band)

    for si, f in enumerate(files):
        img, gt = load_volume(f)
        img_pred = image_pred_latent(model, img, MODALITY, input_size)
        gt = maybe_clean(gt)
        m0 = maybe_clean(predict_mask(seg_model, img, MODALITY))
        m_bad = maybe_clean(morph_mask(m0, PRED_MORPH_DEGRADE_RADIUS))  # away endpoint

        dices, energies, d0 = [], [], dice(m0, gt)
        for t in t_values:
            if t >= 0.0:
                m = blend_masks(m0, gt, t, sigma=PRED_MORPH_BLEND_SIGMA)
            else:
                # map t in [-AWAY_FRAC, 0) to a blend m0 -> m_bad
                s = (-t) / PRED_MORPH_AWAY_FRAC
                m = blend_masks(m0, m_bad, s, sigma=PRED_MORPH_BLEND_SIGMA)
            m = maybe_clean(m)
            dices.append(dice(m, gt))
            energies.append(compute_energy(model, img_pred, m, input_size))

        color = cmap(si / max(1, len(files) - 1))
        label = os.path.basename(f)[:18]
        axes[0].plot(dices, energies, "-o", color=color, label=f"{label} (m0 D={d0:.2f})", alpha=0.8)
        axes[1].plot(t_values, energies, "-o", color=color, label=label, alpha=0.8)

        # Decisive test: monotone decrease along the toward-GT path (t >= 0).
        toward_d = dices[len(t_away):]
        toward_e = energies[len(t_away):]
        rho, _ = spearmanr(toward_d, toward_e)
        per_scan_rho.append(rho)
        print(f"  {label}: m0 Dice={d0:.3f} | toward-GT Spearman(E_x,Dice)={rho:.3f}")

    axes[0].set_xlabel("Dice vs GT")
    axes[0].set_ylabel("Cross-modal energy E_x")
    axes[0].set_title("E_x vs Dice along m0->GT path (want: downhill to Dice=1)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)

    axes[1].axvline(0.0, color="k", ls="--", alpha=0.5)
    axes[1].set_xlabel("path param t  (<0 away from GT, 0 = prediction m0, 1 = GT)")
    axes[1].set_ylabel("Cross-modal energy E_x")
    axes[1].set_title("E_x vs path param (want: decreasing toward t=1)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Phase 0 prediction-centered morph | {MODALITY} | "
                 f"median toward-GT Spearman={np.nanmedian(per_scan_rho):.3f} "
                 f"(want <= -0.7)")
    out = os.path.join(OUT_DIR, f"pred_morph_sweep_{MODALITY}_{RUN_TAG}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[saved] {out}")


def run_gt_vs_pred():
    """Per scan, compare E_x(GT) with E_x(seg prediction).

    Decision-relevant number: the % of scans where the prediction's energy is NOT
    lower than the GT's (E_pred >= E_GT), i.e. GT is the per-scan energy minimum.
    Where that holds, -grad E_x cannot drag a near-perfect mask off GT; where it
    fails (E_pred < E_GT) is exactly where Phase 2's early-stop/leash must hold.
    Points are coloured by prediction Dice so we can see whether the inversions
    concentrate in the high-Dice band (as pred_morph_sweep suggested).
    """
    model, input_size = load_model()
    seg_model = load_seg_model()
    files = list_scans(MODALITY)
    print(f"[gt_vs_pred] {MODALITY}: {len(files)} scans")

    gt_e, pred_e, pred_d, names = [], [], [], []
    for i, f in enumerate(files):
        img, gt = load_volume(f)
        img_pred = image_pred_latent(model, img, MODALITY, input_size)  # per-scan, once
        gt = maybe_clean(gt)
        e_gt = compute_energy(model, img_pred, gt, input_size)
        pred = maybe_clean(predict_mask(seg_model, img, MODALITY))
        e_pred = compute_energy(model, img_pred, pred, input_size)
        d = dice(pred, gt)
        gt_e.append(e_gt); pred_e.append(e_pred); pred_d.append(d)
        names.append(os.path.basename(f).replace(".npy", ""))
        flag = "GTmin" if e_pred >= e_gt else "INVERTED"
        print(f"  [{i+1}/{len(files)}] {names[-1]} E_gt={e_gt:.4f} "
              f"E_pred={e_pred:.4f} margin={e_pred-e_gt:+.4f} Dice={d:.3f} [{flag}]")

    gt_e = np.array(gt_e); pred_e = np.array(pred_e); pred_d = np.array(pred_d)
    margin = pred_e - gt_e                 # >0 => GT lower energy => GT is preferred
    n = len(gt_e)
    n_gt_min = int(np.sum(margin >= 0.0))  # pred energy NOT lower than GT
    pct_gt_min = 100.0 * n_gt_min / max(1, n)
    print(f"\n[gt_vs_pred] GT is the per-scan minimum (E_pred >= E_GT) in "
          f"{n_gt_min}/{n} scans = {pct_gt_min:.1f}%")
    print(f"[gt_vs_pred] inverted (E_pred < E_GT) in {n - n_gt_min}/{n} = "
          f"{100.0 - pct_gt_min:.1f}%  | mean margin={margin.mean():+.4f}")
    inv = margin < 0.0
    if inv.any():
        print(f"[gt_vs_pred] inverted-scan Dice: mean={pred_d[inv].mean():.3f} "
              f"(min={pred_d[inv].min():.3f}, max={pred_d[inv].max():.3f}) "
              f"vs GT-min-scan Dice mean={pred_d[~inv].mean():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Left: E_GT (x) vs E_pred (y); above the diagonal => GT is the lower-energy mask.
    lo = float(min(gt_e.min(), pred_e.min())); hi = float(max(gt_e.max(), pred_e.max()))
    axes[0].plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="E_pred = E_GT")
    sc = axes[0].scatter(gt_e, pred_e, c=pred_d, cmap="viridis", s=45,
                         alpha=0.85, zorder=3)
    fig.colorbar(sc, ax=axes[0], label="prediction Dice")
    axes[0].set_xlabel("E_x(GT)")
    axes[0].set_ylabel("E_x(prediction)")
    axes[0].set_title("above dashed = GT is the lower-energy mask (good)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Right: margin histogram; mass > 0 is the safe side.
    axes[1].hist(margin, bins=30, color="tab:blue", alpha=0.8)
    axes[1].axvline(0.0, color="k", ls="--", alpha=0.6)
    axes[1].set_xlabel("E_pred - E_GT  (>0 = GT preferred / safe)")
    axes[1].set_ylabel("# scans")
    axes[1].set_title(f"GT-min in {pct_gt_min:.1f}% of scans (want high)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Phase 0 cross-modal GT-vs-pred energy | {MODALITY} | n={n} | "
                 f"GT is per-scan min in {pct_gt_min:.1f}% (mean margin {margin.mean():+.4f})")
    out = os.path.join(OUT_DIR, f"gt_vs_pred_{MODALITY}_{RUN_TAG}.png")
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
    elif MODE == "pred_morph_sweep":
        run_pred_morph_sweep()
    elif MODE == "gt_vs_pred":
        run_gt_vs_pred()
    else:
        raise ValueError(f"Unknown MODE={MODE!r}. Use 'raw_scatter', 'morph_sweep', "
                         f"'random_aug_scatter', 'pred_morph_sweep' or 'gt_vs_pred'.")


if __name__ == "__main__":
    main()
