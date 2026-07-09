"""
Phase 0 diagnostic: WHICH shape feature separates CT from UTE masks, and does it
drive the mask-only energy?

Motivation
----------
The size-attractor hypothesis was rejected: CT energy is essentially flat vs lung
volume (Spearman ~ -0.03), yet at *matched volume* UTE GT masks score ~0.04-0.06
higher energy than CT GT masks. Since the masks are binary (no intensity), the
only thing that can differ at fixed size is SHAPE / boundary morphology. The
CT-only mask encoder treats UTE lung *shapes* as off-distribution.

Before retraining with shape augmentations (which would otherwise be guesswork),
this script measures a panel of shape descriptors on every GT mask and asks, per
descriptor:

    (1) Does it separate CT from UTE?            -> Cohen's d (UTE minus CT)
    (2) Is it associated with the energy?        -> Spearman(descriptor, energy),
                                                    reported within CT, within UTE,
                                                    and pooled.

A descriptor that BOTH separates the modalities AND correlates with energy is the
augmentation target: making CT-trained representations invariant to *that* kind
of variation should close the modality gap (and is far better motivated than pure
scale augmentation).

Descriptors (most are scale-invariant on purpose, so we isolate SHAPE from SIZE,
which we already know differs):
    volume_L        lung volume in litres                      (size reference)
    sav_ratio       surface-area / volume  (1/mm)              (size-linked)
    sphericity      (pi^1/3 (6V)^2/3) / SA, 1=perfect sphere   (scale-inv shape)
    roughness       SA / SA(gaussian-smoothed mask)            (boundary, scale-inv)
    solidity_close  V / V(morphological closing)               (concavity, scale-inv)
    n_components    # 26-connected components                  (topology)
    extent          V / bounding-box volume                    (occupancy, scale-inv)
    elongation      sqrt(lambda_max / lambda_min) of coords    (anisotropy, scale-inv)

Energy is read from the cached volume_vs_energy CSV when available (so this can
run without a GPU); otherwise it is recomputed with the exact same machinery as
the other Phase-0 plots.

No argparse: edit the CONFIG block and run. Produces two figures + a results CSV.
"""

import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

# Reuse the exact energy machinery (only needed if the energy cache is missing).
from phase0_energy_vs_dice import (
    load_vjepa,
    precompute_mask_patterns,
    compute_energy,
    load_volume,
)


# ===========================================================================
# CONFIG -- change these, then run.
# ===========================================================================
# Full CT (~690) and UTE (~110) GT masks: we want the whole shape distribution.
CT_CSV = "./ids/only_copd_1.25mm.csv"
UTE_CSV = "./ids/only_ute_1.25mm.csv"

# Optional path remap if CSV roots differ on the run machine, e.g.
# ("/Shared/", "/Volumes/"). Leave None to use CSV paths verbatim.
PATH_REPLACE = None

# Both modalities are 1.25mm isotropic, so a voxel is the same physical size.
VOXEL_MM = 1.25

# Number of scans per modality (None = all). Set small for a quick smoke test.
NUM_CT = None
NUM_UTE = None

# Reuse already-computed energies keyed by filepath. The "both" run of
# phase0_volume_vs_energy.py writes exactly this CSV (columns: filepath, energy).
# If every scan is found here, the V-JEPA model is NOT loaded (no GPU needed).
ENERGY_CACHE_CSV = "./phase0_plots/volume_vs_energy_mask_jepa_ct256.csv"

# Cap coordinates used for the PCA-based elongation (memory guard on huge masks).
MAX_PCA_VOX = 200_000

# Smoothing sigma (voxels) for the boundary-roughness descriptor.
ROUGH_SIGMA = 2.0
# Closing radius (voxels) for the concavity/solidity descriptor.
CLOSE_RADIUS = 3

OUT_DIR = "./phase0_plots"
RUN_TAG = "mask_jepa_ct256"
RESULTS_CSV = os.path.join(OUT_DIR, f"shape_vs_energy_{RUN_TAG}.csv")
HIST_PNG = os.path.join(OUT_DIR, f"shape_separation_{RUN_TAG}.png")
SCATTER_PNG = os.path.join(OUT_DIR, f"shape_vs_energy_{RUN_TAG}.png")

# Descriptors to evaluate, in display order. Size-linked ones flagged for the eye.
DESCRIPTORS = [
    "volume_L", "sav_ratio", "sphericity", "roughness",
    "solidity_close", "n_components", "extent", "elongation",
]
SIZE_LINKED = {"volume_L", "sav_ratio"}  # shown but not augmentation targets


# ===========================================================================
def _resolve(p):
    if PATH_REPLACE is not None:
        p = p.replace(PATH_REPLACE[0], PATH_REPLACE[1])
    return p


def _scans(csv, limit):
    files = [_resolve(f) for f in pd.read_csv(csv)["filepaths"].values]
    return files[:limit] if limit is not None else files


def _ball(r):
    """Spherical binary structuring element of radius r voxels."""
    L = np.arange(-r, r + 1)
    x, y, z = np.meshgrid(L, L, L, indexing="ij")
    return (x * x + y * y + z * z) <= r * r


def _surface_voxfaces(m):
    """Count voxel faces exposed to background (fast surface-area proxy)."""
    m = m.astype(np.int8)
    total = 0
    for ax in range(3):
        pad = [(1, 1) if i == ax else (0, 0) for i in range(3)]
        d = np.abs(np.diff(np.pad(m, pad), axis=ax))
        total += int(d.sum())
    return total


def shape_descriptors(mask, voxel_mm):
    """Compute the descriptor panel for one binary mask (native 1.25mm grid)."""
    m = mask > 0.5
    vox = int(m.sum())
    out = {k: np.nan for k in DESCRIPTORS}
    if vox == 0:
        out["volume_L"] = 0.0
        out["n_components"] = 0
        return out

    vmm3 = voxel_mm ** 3
    amm2 = voxel_mm ** 2
    V = vox * vmm3
    out["volume_L"] = V / 1.0e6

    faces = _surface_voxfaces(m)
    SA = faces * amm2
    out["sav_ratio"] = SA / V
    out["sphericity"] = (np.pi ** (1.0 / 3.0) * (6.0 * V) ** (2.0 / 3.0)) / SA

    sm = gaussian_filter(m.astype(np.float32), sigma=ROUGH_SIGMA) > 0.5
    SA_sm = _surface_voxfaces(sm) * amm2
    out["roughness"] = (SA / SA_sm) if SA_sm > 0 else np.nan

    _, n = ndi.label(m, structure=np.ones((3, 3, 3)))
    out["n_components"] = int(n)

    mc = ndi.binary_closing(m, structure=_ball(CLOSE_RADIUS))
    Vc = int(mc.sum()) * vmm3
    out["solidity_close"] = (V / Vc) if Vc > 0 else np.nan

    nz = np.where(m)
    bbox = 1
    for i in range(3):
        bbox *= int(nz[i].max() - nz[i].min() + 1)
    out["extent"] = vox / float(bbox)

    coords = np.argwhere(m).astype(np.float32)
    if len(coords) > MAX_PCA_VOX:
        sel = np.random.default_rng(0).choice(len(coords), MAX_PCA_VOX, replace=False)
        coords = coords[sel]
    c = coords - coords.mean(axis=0, keepdims=True)
    cov = (c.T @ c) / max(len(c), 1)
    ev = np.clip(np.linalg.eigvalsh(cov), 1e-8, None)  # ascending
    out["elongation"] = float(np.sqrt(ev[-1] / ev[0]))
    return out


def _energy_lookup():
    cache = {}
    if ENERGY_CACHE_CSV and os.path.exists(ENERGY_CACHE_CSV):
        c = pd.read_csv(ENERGY_CACHE_CSV)
        if "filepath" in c.columns and "energy" in c.columns:
            for fp, e in zip(c["filepath"], c["energy"]):
                cache[str(fp)] = float(e)
        print(f"[cache] loaded {len(cache)} energies from {ENERGY_CACHE_CSV}")
    return cache


def _cohens_d(ct, ute):
    """Standardised mean difference, UTE minus CT (positive => UTE larger)."""
    ct = np.asarray(ct, float)
    ct = ct[np.isfinite(ct)]
    ute = np.asarray(ute, float)
    ute = ute[np.isfinite(ute)]
    if len(ct) < 2 or len(ute) < 2:
        return np.nan
    na, nb = len(ct), len(ute)
    sp = np.sqrt(((na - 1) * ct.var(ddof=1) + (nb - 1) * ute.var(ddof=1))
                 / (na + nb - 2))
    return (ute.mean() - ct.mean()) / sp if sp > 0 else np.nan


def _rho(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() > 2 and np.unique(x[ok]).size > 1:
        return spearmanr(x[ok], y[ok]).correlation
    return np.nan


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ct_files = _scans(CT_CSV, NUM_CT)
    ute_files = _scans(UTE_CSV, NUM_UTE)
    all_files = [(f, "CT") for f in ct_files] + [(f, "UTE") for f in ute_files]

    cache = _energy_lookup()
    need_model = any(f not in cache for f, _ in all_files)

    model = input_size = fit_mode = patterns = None
    if need_model:
        print("[model] energy cache incomplete -> loading V-JEPA")
        model, input_size, grid, fit_mode = load_vjepa()
        patterns = precompute_mask_patterns(grid)
    else:
        print("[model] all energies cached -> skipping V-JEPA load (no GPU needed)")

    rows = []
    for i, (f, modality) in enumerate(all_files):
        try:
            _img, gt = load_volume(f)
        except Exception as e:
            print(f"  [skip] {f}: {e}")
            continue

        if f in cache:
            energy = cache[f]
        else:
            energy = compute_energy(model, gt, fit_mode, input_size, patterns)

        d = shape_descriptors(gt, VOXEL_MM)
        d.update(dict(filepath=f, modality=modality, energy=energy))
        rows.append(d)
        print(f"  [{modality} {i+1}/{len(all_files)}] {os.path.basename(f)} "
              f"E={energy:.4f} sph={d['sphericity']:.3f} "
              f"rough={d['roughness']:.3f} sol={d['solidity_close']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"[saved] {RESULTS_CSV}")

    ct = df[df.modality == "CT"]
    ute = df[df.modality == "UTE"]

    # ---- Ranking table: modality separation + energy association per descriptor
    stats = []
    for name in DESCRIPTORS:
        stats.append(dict(
            descriptor=name,
            size_linked=name in SIZE_LINKED,
            cohens_d=_cohens_d(ct[name], ute[name]),
            rho_E_ct=_rho(ct[name], ct.energy),
            rho_E_ute=_rho(ute[name], ute.energy),
            rho_E_all=_rho(df[name], df.energy),
            ct_med=float(np.nanmedian(ct[name])) if len(ct) else np.nan,
            ute_med=float(np.nanmedian(ute[name])) if len(ute) else np.nan,
        ))
    sdf = pd.DataFrame(stats)
    # Rank shape (non-size) descriptors by |Cohen's d|.
    shape_rank = sdf[~sdf.size_linked].reindex(
        sdf[~sdf.size_linked].cohens_d.abs().sort_values(ascending=False).index
    )

    print("\n================ DESCRIPTOR RANKING ================")
    print(f"{'descriptor':<15}{'|d|':>7}{'d(UTE-CT)':>11}"
          f"{'rho_E_ct':>10}{'rho_E_ute':>11}{'CT med':>10}{'UTE med':>10}")
    for _, r in sdf.iterrows():
        tag = "  (size)" if r.size_linked else ""
        print(f"{r.descriptor:<15}{abs(r.cohens_d):>7.2f}{r.cohens_d:>11.2f}"
              f"{r.rho_E_ct:>10.3f}{r.rho_E_ute:>11.3f}"
              f"{r.ct_med:>10.3f}{r.ute_med:>10.3f}{tag}")
    print("---------------------------------------------------")
    top = shape_rank.iloc[0].descriptor if len(shape_rank) else "sphericity"
    print(f"Top modality-separating SHAPE descriptor: {top}")
    print("Augmentation target = a descriptor with large |d| AND non-trivial "
          "within-modality rho_E (so perturbing it moves the energy).")
    print("===================================================\n")

    # ---- Figure 1: per-descriptor CT-vs-UTE histograms ---------------------
    n = len(DESCRIPTORS)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes, DESCRIPTORS):
        cvals = ct[name].to_numpy()
        cvals = cvals[np.isfinite(cvals)]
        uvals = ute[name].to_numpy()
        uvals = uvals[np.isfinite(uvals)]
        lo = np.nanmin(np.concatenate([cvals, uvals])) if len(cvals) + len(uvals) else 0
        hi = np.nanmax(np.concatenate([cvals, uvals])) if len(cvals) + len(uvals) else 1
        bins = np.linspace(lo, hi, 30)
        ax.hist(cvals, bins=bins, color="tab:red", alpha=0.5, density=True, label="CT")
        ax.hist(uvals, bins=bins, color="tab:blue", alpha=0.6, density=True, label="UTE")
        row = sdf[sdf.descriptor == name].iloc[0]
        flag = " [size]" if name in SIZE_LINKED else ""
        ax.set_title(f"{name}{flag}\nd={row.cohens_d:.2f}  "
                     f"rho_E(ct/ute)={row.rho_E_ct:.2f}/{row.rho_E_ute:.2f}",
                     fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"Phase 0 shape-separation | CT vs UTE GT masks ({RUN_TAG})\n"
                 "d = Cohen's d (UTE-CT); large |d| = strong modality separation",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(HIST_PNG, dpi=150)
    print(f"[saved] {HIST_PNG}")

    # ---- Figure 2: top shape descriptor vs energy, coloured by modality ----
    fig2, ax = plt.subplots(figsize=(8, 6))
    if len(ct):
        ax.scatter(ct[top], ct.energy, c="tab:red", s=28, alpha=0.55,
                   label=f"CT (n={len(ct)})")
    if len(ute):
        ax.scatter(ute[top], ute.energy, c="tab:blue", s=34, alpha=0.7,
                   label=f"UTE (n={len(ute)})")
    ax.set_xlabel(f"{top}  (top modality-separating shape descriptor)")
    ax.set_ylabel("mask-only V-JEPA energy")
    row = sdf[sdf.descriptor == top].iloc[0]
    ax.set_title(f"Does '{top}' explain the energy gap?\n"
                 f"Cohen d(UTE-CT)={row.cohens_d:.2f}  "
                 f"rho_E all={row.rho_E_all:.3f}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(SCATTER_PNG, dpi=150)
    print(f"[saved] {SCATTER_PNG}")


if __name__ == "__main__":
    main()
