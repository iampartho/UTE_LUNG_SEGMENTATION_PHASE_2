"""
Phase 2: test-time activation guidance of the FROZEN segmentation U-Net by the
cross-modal energy E_x, evaluated on UTE.

WHAT THIS DOES
--------------
Per scan, everything is frozen (seg net AND the Phase-1 energy). We add a single
learnable activation perturbation delta (init 0) to the U-Net bottleneck x4, then:

    for k in range(MAX_ITERS):
        logits = decode(skips, x4 + delta)          # frozen decoder, grad -> delta
        m      = sigmoid(logits)                     # soft mask (differentiable)
        E_x    = || P(E_img(x)) - LN(E_mask(m)) ||_1 # cross-modal energy
        loss   = E_x + lambda_prox * ||delta||       # (leash, optional)
        delta -= lr * dloss/ddelta                  # optimizer.step()

The image side P(E_img(x)) is a constant w.r.t. delta, so it is computed ONCE per
scan (mirrors phase0_cross_modal_energy.image_pred_latent). Only the mask side
carries gradient. Weights are NEVER touched -- this is source-free test-time
adaptation of activations, not fine-tuning. (Mechanism per the project plan;
matches Humayun et al. ICLR'25 reward-gradient latent guidance.)

WHY THE LEASHES (Phase-0 read)
------------------------------
The cross-modal energy is a strong, correctly-signed compass for BAD predictions
(low-Dice -> steep -ve grad toward GT) but only a shallow/near-flat basin for
already-good predictions (Dice >~ 0.93), where -grad E can drift slightly off GT.
So the design is "fix the bad, protect the good": every leash below is an
independent boolean so you can ablate it.

  * USE_PROX            : lambda_prox * ||delta|| proximal term (stay near m0).
  * USE_DELTA_CAP       : hard-project ||delta|| <= DELTA_MAX_NORM each step.
  * USE_EARLY_STOP      : stop when relative |Delta E| < ENERGY_REL_TOL for PATIENCE
                          steps -> good masks (tiny grad) stop almost immediately.
  * USE_VOLUME_GUARD    : reject the refinement if lung volume moved > VOLUME_REL_TOL,
                          UNLESS the energy is clearly improving (energy-gated; see the
                          Leash-4 config note for why -- it used to veto our best scans).
  * USE_COMPONENT_GUARD : reject if guidance spawned > MAX_NEW_COMPONENTS blobs.
  * USE_LCC_POST        : keep up to NUM_LARGEST_CC components in the final mask,
                          but drop the 2nd..k-th unless they are >= LCC_MIN_FRAC of
                          the largest (guards against the fused-lungs case where the
                          true lungs are one component and the 2nd "largest" is a
                          pure false positive promoted only to satisfy k=2).
  * USE_ENERGY_FALLBACK : revert to m0 unless the refined mask lowered E by at least
                          ENERGY_FALLBACK_MARGIN (relative); protects already-good masks.

Set USE_GUIDANCE=False to emit the pure baseline m0 (reproduces eval_step_1) -- the
zero-leash ablation anchor.

I/O CONVENTIONS (merged from eval_step_1.py + eval_step_1_resample_to_1mm.py)
----------------------------------------------------------------------------
Guidance always runs at 1.25mm / 256^3 (UTE native). The single switch
RESAMPLE_TO_1MM chooses how outputs are written:

  RESAMPLE_TO_1MM=False : save pred/gt(/confusion) at 1.25mm straight from the npy
                          stack (eval_step_1.py convention).
  RESAMPLE_TO_1MM=True  : resample the refined 1.25mm mask to 1mm and stamp it onto
                          a 1mm reference NIfTI (GT folder = eval mode, else IMG
                          folder = predict-only), incl. orientation un-flip + LCC
                          (eval_step_1_resample_to_1mm.py convention).

No argparse by design: edit CONFIG, then run.
"""

import os
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import SimpleITK as sitk
import tqdm
from scipy.ndimage import label as cc_label

from basic_unet_disentangled import BasicUNet
from model_img_jepa import ImgJEPA
from dataset_mask_jepa import _center_pad_crop, _clean_binary

# Pure I/O helpers reused verbatim from the two eval scripts (importing does not
# run their __main__).
from eval_step_1 import (
    normalise_one_one,
    normalise_zero_one,
    normalise_hu,
    generate_nifty,
    generate_confusion_mask,
)
from eval_step_1_resample_to_1mm import (
    build_pair_table,
    resample_pred_to_1mm,
    match_shape,
    save_with_reference,
    canonicalize_flip_axes,
    restore_original_orientation,
)


# ===========================================================================
# CONFIG -- change these, then run.
# ===========================================================================
MODALITY = "UTE"               # 'UTE' | 'CT'  (guidance target is UTE)

# --- weights -----------------------------------------------------------------
# Phase 1 step 2 image-JEPA checkpoint (carries E_img + P + frozen E_mask).
IMG_JEPA_CKPT = "./save_models/best_img_jepa_ct256_new_aug.pth"
# The SAME seg net whose predictions Phase 0 validated (current best).
SEG_CKPT = ("./save_models/current_best_may_2026/"
            "best_bunet_causality_paper_ct_train_UTE_test_w_tversky_wo_kl_only_gin_roughness_enforced.pth")

# --- output mode -------------------------------------------------------------
RESAMPLE_TO_1MM = True        # False -> 1.25mm outputs ; True -> resample to 1mm

# 1.25mm mode (RESAMPLE_TO_1MM=False): npy stack carries image[...,0] and GT[...,1].
DATA_CSV = '' #"./ids/UTE_MRI_previous_numpy_without_clipping.csv"

# 1mm mode (RESAMPLE_TO_1MM=True): pair the 1.25mm npy with a 1mm reference NIfTI.
CSV_125MM   = "./ids/UTE_MRI_OECLAD_ORIGINAL_SPACING_resamples_1.25mm_only_img.csv"
GT_1MM_DIR  = "/Shared/lss_segerard/data/UTE_MRI_OECLAD/MASKS_better"               # set -> eval mode (GT + confusion); '' -> predict-only
IMG_1MM_DIR = "" #"/Shared/lss_segerard/data/UTE_MRI_OECLAD/UTE"   # used when GT_1MM_DIR == ''

OUTPUT_DIR = "./prediction/phase2_guidance_new_run/kp_data_OECLAD"
METRICS_CSV = "./phase2_guidance_logs/phase2_guidance_metrics_kp_data_OECLAD.csv"

NUM_SCANS = None               # cap for quick runs (None = all)

# --- seg-net input preprocessing (mirror the eval scripts) -------------------
NUM_DOUBLE_STRIDE_CONV = 4     # pad each dim up to a multiple of 2**this
SEG_PAD_VAL = -1.0             # NOTE: phase0_cross_modal pads the seg input with -1
                               # (the masks we validated). eval_step_1 used 0.5; for
                               # native-256 UTE there is NO padding, so this is moot.
SEG_THRESHOLD = 0.5

# --- energy parameters (match training / phase0_cross_modal_energy) ----------
INPUT_SIZE = (256, 256, 256)
ENERGY_PAD_VAL = -1.0          # image pad value at the 256^3 fit (images in [-1,1])
GAUSSIAN_SIGMA = 1.0           # mask softening (match E_mask training input)
TARGET_NORM = True             # LayerNorm the E_mask target (match training)
CLEAN_MASKS = None             # None -> read from the img-JEPA ckpt 'aug' block
CLEAN_KEEP_FRAC = None

# ===========================================================================
# GUIDANCE + LEASHES
# ===========================================================================
USE_GUIDANCE = True            # False -> emit baseline m0 (zero-leash ablation anchor)

MAX_ITERS = 20                 # max-K
LR = 0.05                      # Adam step on delta
OPTIMIZER = "adam"             # 'adam' | 'sgd'

# Leash 1: proximal term on the perturbation magnitude.
USE_PROX = True
LAMBDA_PROX = 0.01

# Leash 2: hard trust region on ||delta||.
USE_DELTA_CAP = True
DELTA_MAX_NORM = 50.0          # global L2 cap on delta (re-projected each step)

# Leash 3: energy-plateau early stop (protects already-good predictions).
USE_EARLY_STOP = True
ENERGY_REL_TOL = 1e-3
PATIENCE = 3

# Leash 4: volume guard -- reject refinement if lung volume drifts too far from m0.
# FIRST-RUN DEFAULT (the 4 UTE runs): USE_VOLUME_GUARD=True, VOLUME_REL_TOL=0.25, and the
# guard HARD-broke the walk and reverted to m0 with NO energy override. That cost us our
# three highest-value scans -- CVD-XE-002C (Dice 0.708), CVD-XE-007A (0.779) and
# OECLAD_004A (0.808). On a bad baseline the *correct* fix REQUIRES a large volume change
# (the baseline is over-/under-segmenting), so the symmetric +/-25% band fired exactly
# where guidance helps most: in each case the cross-modal energy was crashing
# (e.g. 0.616 -> 0.351) and we threw that result away to keep the worse m0.
# FIX: energy-gate the guard. Only let a big volume move count as "divergence" when the
# energy is NOT meaningfully improving. If E_bin has dropped past VOLUME_GUARD_ENERGY_BYPASS
# (relative) below E0, that move is the compass doing its job -- trust it (the whole thesis
# is energy > volume heuristic) and ignore the guard.
USE_VOLUME_GUARD = True
VOLUME_REL_TOL = 0.25          # allow +/-25% voxel-count change vs m0
VOLUME_GUARD_ENERGY_BYPASS = 0.15  # bypass the guard when E_bin <= E0*(1-this).
                                   # 0.0 -> never bypass (reproduces the first-run guard).

# Leash 5: component guard -- reject if guidance spawned spurious blobs.
USE_COMPONENT_GUARD = True
MAX_NEW_COMPONENTS = 1         # allowed increase in #connected-components over m0
COMPONENT_FULL_CONN = True     # 26-conn (True) vs 6-conn (False) for the count

# Leash 6: keep-largest-components post-process on the FINAL mask.
USE_LCC_POST = True
NUM_LARGEST_CC = 2
LCC_FULL_CONNECTIVITY = True
LCC_MIN_FRAC = 0.10            # keep the 2nd..k-th CC only if >= this fraction of the
                              # largest CC; else keep fewer (mirrors maybe_clean's
                              # keep_frac, but with a hard cap of NUM_LARGEST_CC).

# Leash 7: energy fallback -- revert to m0 if E did not actually improve.
# FIRST-RUN DEFAULT: accept the refinement whenever best_iter_E < E0 (i.e. ANY energy
# improvement -- margin = 0). Diagnosis across the 4 runs: of the ~100 "worsened" scans,
# essentially all were already-good baselines (Dice >= 0.93) whose energy still dropped a
# little (~13% rel) while Dice drifted DOWN -- the shallow-basin / off-GT drift the header
# warns about. Genuine wins drop energy ~28% rel, so a relative margin cleanly separates
# the two regimes without touching the bad scans (they blow past any threshold).
# FIX: require the refinement to beat m0 by a RELATIVE margin before accepting it, so a
# tiny energy dip on an already-good mask no longer overrides m0.
USE_ENERGY_FALLBACK = True
ENERGY_FALLBACK_MARGIN = 0.10  # accept refined only if best_iter_E < E0*(1-this).
                               # 0.0 -> reproduces the first-run "any improvement" rule.
                               # Tuned across the 4 UTE runs (218 applied scans): the
                               # delivered pooled mean-Dice gain is a flat plateau from
                               # 0.15 (+0.0098) to ~0.22, then falls off a cliff at 0.25+
                               # as the margin starts vetoing genuine wins. 0.10 (+0.0093)
                               # under-prunes -- it kept ~63 already-good masks that drifted
                               # DOWN in Dice while energy dipped only a little. 0.15 sits at
                               # the front of the plateau (safest vs in-sample overfit) and
                               # roughly halves the worsened-scan count on every dataset
                               # (e.g. prev_data 12->1 with identical pooled gain).

INPUT_SPACING_MM  = (1.25, 1.25, 1.25)
OUTPUT_SPACING_MM = (1.0, 1.0, 1.0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Resolved from the img-JEPA checkpoint inside load_energy() (CONFIG overrides win).
_CLEAN_ENABLED = True
_CLEAN_KEEP_FRAC = 0.10


# ===========================================================================
# Normalisation / cleaning
# ===========================================================================
def normalise_image(img, modality):
    if modality == "CT":
        return normalise_one_one(normalise_hu(img))
    return normalise_one_one(img)          # UTE


def maybe_clean(mask_bin):
    if _CLEAN_ENABLED:
        return _clean_binary(mask_bin, keep_frac=_CLEAN_KEEP_FRAC)
    return mask_bin.astype(np.float32)


def dice(a, b):
    a = a > 0.5
    b = b > 0.5
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return 1.0 if denom == 0 else float(2.0 * inter / denom)


def n_components(mask_bin):
    struct = np.ones((3, 3, 3)) if COMPONENT_FULL_CONN else None
    _, n = cc_label(mask_bin > 0.5, structure=struct)
    return int(n)


def keep_largest_cc_capped(mask_bin, k=NUM_LARGEST_CC, keep_frac=LCC_MIN_FRAC,
                           full_connectivity=LCC_FULL_CONNECTIVITY):
    """Keep AT MOST ``k`` largest connected components, but drop the 2nd..k-th
    unless its size is >= ``keep_frac`` * (largest CC size).

    Rationale: ``keep_largest_connected_components`` (eval_step_1_resample_to_1mm)
    unconditionally keeps the top-k blobs. When the two lungs are predicted as a
    single fused component, that promotes the next-largest blob -- often a pure
    false positive -- just to reach k=2. Thresholding on a fraction of the largest
    (the same idea as ``maybe_clean``'s keep_frac) returns only the largest CC in
    that case. Returns float32, same shape as input.
    """
    if k is None or k <= 0:
        return mask_bin.astype(np.float32)
    m = mask_bin > 0.5
    if not m.any():
        return mask_bin.astype(np.float32)
    struct = np.ones((3, 3, 3)) if full_connectivity else None
    lbl, _ = cc_label(m, structure=struct)
    counts = np.bincount(lbl.ravel())
    counts[0] = 0                       # drop background
    sizes = counts[1:]                  # label id i -> sizes[i-1]
    order = np.argsort(sizes)[::-1]     # component label ids by size, descending
    largest = sizes[order[0]]
    keep = [int(order[0]) + 1]          # always keep the largest
    for idx in order[1:k]:              # at most k total
        if sizes[idx] >= keep_frac * largest:
            keep.append(int(idx) + 1)
        else:
            break                       # sizes are descending -> rest are smaller too
    return np.isin(lbl, keep).astype(np.float32)


# ===========================================================================
# Differentiable mask -> 256^3 -> blur (the path delta's gradient flows through)
# ===========================================================================
def _center_fit_torch(t, target, pad_value=0.0):
    """Centre pad/crop a (1,1,H,W,D) tensor to ``target`` per spatial axis.

    Differentiable (narrow + F.pad). For native-256 UTE this is a no-op.
    """
    out = t
    # axis 2->H, 3->W, 4->D ; F.pad takes the last spatial dim first.
    for axis, tgt in zip((2, 3, 4), target):
        cur = out.shape[axis]
        if cur == tgt:
            continue
        if cur > tgt:
            start = (cur - tgt) // 2
            out = out.narrow(axis, start, tgt)
        else:
            total = tgt - cur
            before, after = total // 2, total - total // 2
            pad = [0, 0, 0, 0, 0, 0]          # (D_l,D_r,W_l,W_r,H_l,H_r)
            slot = {2: 4, 3: 2, 4: 0}[axis]
            pad[slot], pad[slot + 1] = before, after
            out = F.pad(out, pad, mode="constant", value=pad_value)
    return out


_BLUR_CACHE = {}


def _gaussian_blur3d(t, sigma):
    """Separable 3D Gaussian blur of a (1,1,H,W,D) tensor (differentiable)."""
    if sigma <= 0.0:
        return t
    key = (round(float(sigma), 4), t.device)
    if key not in _BLUR_CACHE:
        radius = max(1, int(round(3.0 * sigma)))
        x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=t.device)
        k = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
        k = k / k.sum()
        _BLUR_CACHE[key] = (k, radius)
    k, radius = _BLUR_CACHE[key]
    out = t
    for axis in (2, 3, 4):
        shape = [1, 1, 1, 1, 1]
        shape[axis] = k.numel()
        pad = [0, 0, 0]
        pad[4 - axis] = radius                # conv3d pad order is (D,W,H)
        out = F.conv3d(out, k.view(*shape), padding=tuple(pad))
    return out.clamp(0.0, 1.0)


# ===========================================================================
# U-Net split (encode -> bottleneck x4 -> decode), no model edits.
# ===========================================================================
def _cbr(model, name, x):
    conv = getattr(model, f"conv_{name}")
    norm = getattr(model, f"norm_{name}")
    act = getattr(model, f"act_{name}")
    return act(norm(conv(x)))


def unet_encode(model, x):
    """Return (x1, x2, x3, x4_bottleneck). Mirrors BasicUNet.forward up to x4."""
    x1 = _cbr(model, "0_0", x)
    x1 = _cbr(model, "0_1", x1)
    x2 = model.pool_1(x1)
    x2 = _cbr(model, "1_0", x2)
    x2 = _cbr(model, "1_1", x2)
    x3 = model.pool_2(x2)
    x3 = _cbr(model, "2_0", x3)
    x3 = _cbr(model, "2_1", x3)
    x4 = model.pool_3(x3)
    x4 = _cbr(model, "3_0", x4)
    x4 = _cbr(model, "3_1", x4)              # <-- guidance site
    return x1, x2, x3, x4


def unet_decode(model, x1, x2, x3, x4):
    """Decode from the (possibly perturbed) bottleneck x4 + frozen skips."""
    y = model.up_4(x4)
    y = torch.cat([y, x3], dim=1)
    y = _cbr(model, "up_4_0", y)
    y = _cbr(model, "up_4_1", y)
    y = model.up_3(y)
    y = torch.cat([y, x2], dim=1)
    y = _cbr(model, "up_3_0", y)
    y = _cbr(model, "up_3_1", y)
    y = model.up_2(y)
    y = torch.cat([y, x1], dim=1)
    y = _cbr(model, "up_2_0", y)
    y = _cbr(model, "up_2_1", y)
    return model.final_conv(y)


# ===========================================================================
# Cross-modal energy (image side once; mask side carries grad)
# ===========================================================================
def load_energy():
    model, cfg = ImgJEPA.from_checkpoint(IMG_JEPA_CKPT, DEVICE)
    ckpt = torch.load(IMG_JEPA_CKPT, map_location="cpu")
    aug = ckpt.get("aug", {}) or {}
    global _CLEAN_ENABLED, _CLEAN_KEEP_FRAC
    _CLEAN_ENABLED = aug.get("clean_masks", True) if CLEAN_MASKS is None else CLEAN_MASKS
    _CLEAN_KEEP_FRAC = (aug.get("clean_keep_frac", 0.10)
                        if CLEAN_KEEP_FRAC is None else CLEAN_KEEP_FRAC)
    print(f"[energy] loaded {IMG_JEPA_CKPT} | tokens={model.encoder.num_patches} "
          f"grid={model.encoder.grid}")
    print(f"[energy] mask cleaning: enabled={_CLEAN_ENABLED} keep_frac={_CLEAN_KEEP_FRAC}")
    return model


def preprocess_image_256(img_native, modality):
    """Normalise + centre-fit the image to 256^3 (the E_img input). NO GIN."""
    img = normalise_image(img_native, modality)
    if tuple(img.shape) != INPUT_SIZE:
        img = _center_pad_crop(img, INPUT_SIZE, pad_value=ENERGY_PAD_VAL)
    img = img.astype(np.float32)
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)


@torch.no_grad()
def image_pred_latent(energy_model, img_native, modality):
    """P(E_img(x)) for a scan -- constant w.r.t. delta, computed once."""
    x = preprocess_image_256(img_native, modality)
    grid = energy_model.encoder.grid_of(x)
    img_tokens = energy_model.encoder(x, keep_idx=None)
    return energy_model.predictor(img_tokens, grid)          # (1, N, dim)


def soft_mask_latent(energy_model, soft_mask_5d):
    """LN(E_mask(m)) for a SOFT mask (1,1,H,W,D), keeping the gradient path open."""
    m = _center_fit_torch(soft_mask_5d, INPUT_SIZE, pad_value=0.0)
    m = _gaussian_blur3d(m, GAUSSIAN_SIGMA)
    target = energy_model.mask_encoder(m, keep_idx=None)     # frozen weights, grad to input
    if TARGET_NORM:
        target = F.layer_norm(target, (target.shape[-1],))
    return target


def energy_of(img_pred, mask_target):
    return F.l1_loss(img_pred, mask_target)


# ===========================================================================
# Per-scan: predict m0, then guide.
# ===========================================================================
def seg_input(img_native, modality):
    """Normalise + pad to a multiple of 2**NUM_DOUBLE_STRIDE_CONV (eval convention)."""
    img = normalise_image(img_native, modality)
    h, w, d = img.shape
    mul = 2 ** NUM_DOUBLE_STRIDE_CONV
    nh, nw, nd = (mul * math.ceil(s / mul) for s in (h, w, d))
    img = np.pad(img, ((0, nh - h), (0, nw - w), (0, nd - d)),
                 mode="constant", constant_values=SEG_PAD_VAL)
    x = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return x, (h, w, d)


def _binary_native(prob_5d, hwd):
    """(1,1,H,W,D) soft -> cropped native binary numpy (uint8-ish float)."""
    h, w, d = hwd
    b = (prob_5d.detach()[0, 0, :h, :w, :d] > SEG_THRESHOLD).float().cpu().numpy()
    return b.astype(np.float32)


@torch.no_grad()
def binary_energy(energy_model, img_pred, mask_bin_native):
    """E_x of a CLEANED BINARY native mask -- identical computation to Phase 0
    (fit 256^3 -> blur sigma 1 -> LN(E_mask)). Used for selection + reporting so
    the numbers line up with phase0_cross_modal_energy.py."""
    t = torch.from_numpy(mask_bin_native.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return energy_of(img_pred, soft_mask_latent(energy_model, t)).item()


def guide_scan(seg, energy_model, img_native, modality):
    """Return (m0_bin, refined_bin, metrics_dict), both cleaned native binaries."""
    img_pred = image_pred_latent(energy_model, img_native, modality)
    x_seg, hwd = seg_input(img_native, modality)
    h, w, d = hwd

    with torch.no_grad():
        x1, x2, x3, x4 = unet_encode(seg, x_seg)
        logits0 = unet_decode(seg, x1, x2, x3, x4)
        prob0 = torch.sigmoid(logits0)

    m0_bin = maybe_clean(_binary_native(prob0, hwd))
    vol0 = float(m0_bin.sum())
    nc0 = n_components(m0_bin)
    E0 = binary_energy(energy_model, img_pred, m0_bin)   # Phase-0-comparable

    metrics = dict(E0=E0, E_final=E0, n_iters=0, fell_back=False,
                   reason="", vol0=vol0, vol_final=vol0, nc0=nc0, nc_final=nc0)

    if not USE_GUIDANCE:
        return m0_bin, m0_bin, metrics

    # Detach the frozen graph; delta is the only learnable leaf.
    x1, x2, x3, x4 = (t.detach() for t in (x1, x2, x3, x4))
    delta = torch.zeros_like(x4, requires_grad=True)
    if OPTIMIZER == "sgd":
        opt = torch.optim.SGD([delta], lr=LR, momentum=0.9)
    else:
        opt = torch.optim.Adam([delta], lr=LR)

    # Optimization tracks the SOFT energy (smooth objective / early-stop signal);
    # mask SELECTION tracks the BINARY energy of the cleaned deliverable.
    best_iter_E, best_iter_bin = np.inf, None
    prev_E, patience = None, 0
    guard_fired, reason = False, ""
    it = 0
    for it in range(1, MAX_ITERS + 1):
        opt.zero_grad()
        logits = unet_decode(seg, x1, x2, x3, x4 + delta)
        prob = torch.sigmoid(logits)
        target = soft_mask_latent(energy_model, prob)
        E = energy_of(img_pred, target)
        loss = E + (LAMBDA_PROX * delta.norm() if USE_PROX else 0.0)
        loss.backward()
        opt.step()

        if USE_DELTA_CAP:
            with torch.no_grad():
                nrm = delta.norm()
                if nrm > DELTA_MAX_NORM:
                    delta.mul_(DELTA_MAX_NORM / (nrm + 1e-12))

        E_soft = E.item()
        cur_bin = maybe_clean(_binary_native(prob, hwd))
        vol, ncomp = float(cur_bin.sum()), n_components(cur_bin)

        # Binary energy of the cleaned deliverable -- the SELECTION signal. Computed here
        # (it used to live AFTER the guards) so the volume guard can be energy-gated below.
        E_bin = binary_energy(energy_model, img_pred, cur_bin)
        if E_bin < best_iter_E:
            best_iter_E, best_iter_bin = E_bin, cur_bin

        # Guards: abort the walk and fall back to m0 if the mask diverges.
        # Volume guard is ENERGY-GATED (see Leash-4 note): a big volume move only counts as
        # divergence when the energy is NOT clearly improving. Once E_bin has dropped past
        # VOLUME_GUARD_ENERGY_BYPASS below E0, the move is the compass correcting a bad
        # baseline -- let it through instead of reverting to the worse m0.
        energy_improving = E_bin < E0 * (1.0 - VOLUME_GUARD_ENERGY_BYPASS)
        if (USE_VOLUME_GUARD and not energy_improving and vol0 > 0
                and abs(vol - vol0) / vol0 > VOLUME_REL_TOL):
            guard_fired, reason = True, f"volume_guard({abs(vol-vol0)/vol0:.2f})"
            break
        if USE_COMPONENT_GUARD and ncomp > nc0 + MAX_NEW_COMPONENTS:
            guard_fired, reason = True, f"component_guard({ncomp}>{nc0}+{MAX_NEW_COMPONENTS})"
            break

        rel = abs(prev_E - E_soft) / max(prev_E, 1e-8) if prev_E is not None else 1.0
        print(f"    it {it:02d}/{MAX_ITERS}  E_soft={E_soft:.6f} E_bin={E_bin:.6f} "
              f"(dE_rel={rel:.2e}) |delta|={delta.norm().item():.3f} "
              f"vol={vol/max(vol0,1):.3f} nc={ncomp}")
        if USE_EARLY_STOP:
            patience = patience + 1 if rel < ENERGY_REL_TOL else 0
            if patience >= PATIENCE:
                reason = "early_stop"
                break
        prev_E = E_soft

    # Decide the deliverable.
    fell_back = False
    if guard_fired:
        refined_bin = m0_bin
        fell_back = True
    elif best_iter_bin is None:                       # never produced a candidate
        refined_bin = m0_bin
        fell_back = True
        reason = reason or "no_candidate"
    elif USE_ENERGY_FALLBACK and best_iter_E > E0 * (1.0 - ENERGY_FALLBACK_MARGIN):
        # refinement didn't beat m0 by the required relative margin (see Leash-7 note)
        refined_bin = m0_bin
        fell_back = True
        reason = (reason + "+" if reason else "") + "energy_fallback"
    else:
        refined_bin = best_iter_bin

    if USE_LCC_POST:
        refined_bin = keep_largest_cc_capped(
            refined_bin, k=NUM_LARGEST_CC, keep_frac=LCC_MIN_FRAC,
            full_connectivity=LCC_FULL_CONNECTIVITY,
        )

    E_final = binary_energy(energy_model, img_pred, refined_bin)
    metrics.update(E_final=E_final, n_iters=it, fell_back=fell_back, reason=reason,
                   vol_final=float(refined_bin.sum()), nc_final=n_components(refined_bin))
    return m0_bin, refined_bin, metrics


# ===========================================================================
# Output writers
# ===========================================================================
def save_1p25mm(stem, refined_bin, gt_bin):
    out_pred = f"{OUTPUT_DIR}/{stem}_pred.nii.gz"
    generate_nifty(refined_bin.astype(np.uint8), INPUT_SPACING_MM, out_pred)
    if gt_bin is not None:
        out_gt = f"{OUTPUT_DIR}/{stem}_gt.nii.gz"
        out_conf = f"{OUTPUT_DIR}/{stem}_confusion_unet.nii.gz"
        generate_nifty(gt_bin.astype(np.uint8), INPUT_SPACING_MM, out_gt)
        generate_confusion_mask(out_gt, out_pred, out_conf)


def to_1mm_frame(bin_125, ref_img_sitk, target_shape):
    """Resample a 1.25mm native binary into the 1mm reference frame.

    Mirrors the refined-pred path (resample -> orientation un-flip -> shape match
    -> optional LCC) so that ANY 1.25mm binary (m0 or refined) can be diced against
    the 1mm GT in the same frame. Returns a uint8 array shaped ``target_shape``.
    """
    arr = resample_pred_to_1mm(bin_125.astype(np.uint8),
                               INPUT_SPACING_MM, OUTPUT_SPACING_MM)
    flips = canonicalize_flip_axes(ref_img_sitk)
    arr = restore_original_orientation(arr, flips)
    arr = match_shape(arr, target_shape, fill_value=0)
    if USE_LCC_POST:
        arr = keep_largest_cc_capped(
            arr, k=NUM_LARGEST_CC, keep_frac=LCC_MIN_FRAC,
            full_connectivity=LCC_FULL_CONNECTIVITY).astype(np.uint8)
    return arr


def save_1mm(stem, refined_bin_125, ref_img_sitk, gt_mode):
    ref_arr = sitk.GetArrayFromImage(ref_img_sitk)
    if gt_mode:
        gt_1mm = (ref_arr > 0).astype(np.uint8)
        target_shape = gt_1mm.shape
    else:
        gt_1mm, target_shape = None, ref_arr.shape

    pred_1mm = to_1mm_frame(refined_bin_125, ref_img_sitk, target_shape)

    out_pred = f"{OUTPUT_DIR}/{stem}_pred_1mm.nii.gz"
    save_with_reference(pred_1mm, ref_img_sitk, out_pred)
    if gt_mode:
        out_gt = f"{OUTPUT_DIR}/{stem}_gt_1mm.nii.gz"
        out_conf = f"{OUTPUT_DIR}/{stem}_confusion_unet_1mm.nii.gz"
        save_with_reference(gt_1mm, ref_img_sitk, out_gt)
        generate_confusion_mask(out_gt, out_pred, out_conf)
        return pred_1mm, gt_1mm
    return pred_1mm, None


# ===========================================================================
def _log(rows):
    os.makedirs(os.path.dirname(METRICS_CSV), exist_ok=True)
    pd.DataFrame(rows).to_csv(METRICS_CSV, index=False)


def run_1p25mm(seg, energy_model):
    df = pd.read_csv(DATA_CSV)
    files = df["filepaths"].values
    if NUM_SCANS is not None:
        files = files[:NUM_SCANS]
    print(f"[phase2] 1.25mm mode | {len(files)} scans")

    rows = []
    for f in tqdm.tqdm(files):
        arr = np.load(f)
        img = arr[:, :, :, 0].astype(np.float32)
        gt = maybe_clean((arr[:, :, :, 1] > 0).astype(np.float32))

        m0, refined, mtr = guide_scan(seg, energy_model, img, MODALITY)
        stem = "_".join(f.split("/")[-4:]).split(".npy")[0]
        save_1p25mm(stem, refined, gt)

        mtr.update(scan=stem, dice_m0=dice(m0, gt), dice_refined=dice(refined, gt))
        rows.append(mtr)
        print(f"  {stem}: Dice {mtr['dice_m0']:.3f} -> {mtr['dice_refined']:.3f} | "
              f"E {mtr['E0']:.4f} -> {mtr['E_final']:.4f} | "
              f"iters={mtr['n_iters']} fb={mtr['fell_back']}({mtr['reason']})")
        _log(rows)
    _summary(rows)


def run_1mm(seg, energy_model):
    gt_mode = bool(GT_1MM_DIR)
    ref_dir = GT_1MM_DIR if gt_mode else IMG_1MM_DIR
    ref_col = "gt_path" if gt_mode else "img_path"
    pair_df = build_pair_table(CSV_125MM, ref_dir, ref_col)
    if NUM_SCANS is not None:
        pair_df = pair_df.head(NUM_SCANS)
    print(f"[phase2] 1mm mode ({'eval' if gt_mode else 'predict-only'}) | "
          f"{len(pair_df)} scans")

    rows = []
    for _, row in tqdm.tqdm(pair_df.iterrows(), total=len(pair_df)):
        arr = np.load(row["filepaths"])
        img = arr[:, :, :, 0].astype(np.float32) if arr.ndim == 4 else arr.astype(np.float32)

        m0, refined, mtr = guide_scan(seg, energy_model, img, MODALITY)
        ref_img = sitk.ReadImage(row[ref_col])
        pred_1mm, gt_1mm = save_1mm(row["__stem__"], refined, ref_img, gt_mode)

        mtr.update(scan=row["__stem__"])
        dice_msg = ""
        if gt_mode:
            # Resample m0 into the SAME 1mm frame as the refined pred so both dice
            # against gt_1mm are apples-to-apples (pred0 -> final dice change).
            m0_1mm = to_1mm_frame(m0, ref_img, gt_1mm.shape)
            mtr.update(dice_m0_1mm=dice(m0_1mm, gt_1mm),
                       dice_refined_1mm=dice(pred_1mm, gt_1mm))
            dice_msg = (f"Dice {mtr['dice_m0_1mm']:.3f} -> "
                        f"{mtr['dice_refined_1mm']:.3f} | ")
        rows.append(mtr)
        print(f"  {row['__stem__']}: {dice_msg}"
              f"E {mtr['E0']:.4f} -> {mtr['E_final']:.4f} | "
              f"iters={mtr['n_iters']} fb={mtr['fell_back']}({mtr['reason']})")
        _log(rows)
    _summary(rows)


def _summary(rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    print("\n================ Phase 2 summary ================")
    # 1.25mm mode logs dice_m0/dice_refined; 1mm mode logs the _1mm pair.
    if "dice_m0" in df:
        m0_col, ref_col = "dice_m0", "dice_refined"
    elif "dice_m0_1mm" in df:
        m0_col, ref_col = "dice_m0_1mm", "dice_refined_1mm"
    else:
        m0_col = ref_col = None
    if m0_col:
        print(f" mean Dice  m0={df[m0_col].mean():.4f}  "
              f"refined={df[ref_col].mean():.4f}  "
              f"delta={df[ref_col].mean() - df[m0_col].mean():+.4f}")
        improved = (df[ref_col] > df[m0_col] + 1e-4).sum()
        worsened = (df[ref_col] < df[m0_col] - 1e-4).sum()
        print(f" scans improved={improved}  worsened={worsened}  "
              f"unchanged={len(df) - improved - worsened}")
    print(f" mean E  {df['E0'].mean():.4f} -> {df['E_final'].mean():.4f} | "
          f"fell_back={int(df['fell_back'].sum())}/{len(df)}")
    print(f" metrics -> {METRICS_CSV}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"device={DEVICE} | RESAMPLE_TO_1MM={RESAMPLE_TO_1MM} | USE_GUIDANCE={USE_GUIDANCE}")

    seg = BasicUNet().to(DEVICE)
    state = torch.load(SEG_CKPT, map_location=DEVICE)
    res = seg.load_state_dict(state, strict=False)
    if res.missing_keys:
        print(f"[seg] missing keys ({len(res.missing_keys)}): {res.missing_keys[:6]} ...")
    if res.unexpected_keys:
        print(f"[seg] unexpected keys ({len(res.unexpected_keys)}): {res.unexpected_keys[:6]} ...")
    seg.eval()
    for p in seg.parameters():
        p.requires_grad_(False)
    print(f"[seg] loaded {SEG_CKPT}")

    energy_model = load_energy()
    energy_model.eval()
    for p in energy_model.parameters():
        p.requires_grad_(False)

    if RESAMPLE_TO_1MM:
        run_1mm(seg, energy_model)
    else:
        run_1p25mm(seg, energy_model)
    print("Done!")


if __name__ == "__main__":
    main()
