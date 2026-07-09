"""
Phase 0 verification: is the CT-vs-UTE separation on extent / elongation genuine
ANATOMY, or an artifact of fragmented/holey UTE masks?

Background
----------
phase0_shape_vs_energy.py found that UTE GT masks differ from CT GT masks most on
a "compactness/inflation" axis (UTE more elongated, lower bbox extent), and that
this axis is the strongest within-UTE correlate of the mask-only energy. BUT UTE
masks are also much more fragmented (connected components up to 18 vs 1-2 for CT),
and both `extent` (bbox-based) and `elongation` (PCA-based) are sensitive to stray
specks: one far speck enlarges the bounding box and the principal axis.

So before designing an anisotropic/elastic shape augmentation around that axis, we
must rule out the artifact explanation. The test:

    Recompute the descriptors on RAW masks and on CLEANED masks
    (keep the major connected component(s) + fill holes), then compare the
    CT-vs-UTE Cohen's d before and after cleaning.

    * If |d| SURVIVES cleaning           -> genuine inflation-state anatomy
                                            -> the augmentation is well motivated.
    * If |d| COLLAPSES after cleaning     -> it was the speck/hole artifact
                                            -> fix is mask cleaning, not augmentation.

This script is deliberately SELF-CONTAINED: it does NOT import the V-JEPA model or
compute any energy, so it needs only numpy / pandas / scipy / matplotlib and runs
quickly on CPU. (The separate energy re-check, which DOES need the model, is left
as the follow-up step described at the end of the printout.)

No argparse: edit the CONFIG block and run.
"""

import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi


# ===========================================================================
# CONFIG -- change these, then run.
# ===========================================================================
CT_CSV = "./ids/only_copd_1.25mm.csv"
UTE_CSV = "./ids/only_ute_1.25mm.csv"
PATH_REPLACE = None  # e.g. ("/Shared/", "/Volumes/"); None = use paths verbatim

VOXEL_MM = 1.25
NUM_CT = None   # None = all (~690). Set small (e.g. 20) for a quick smoke test.
NUM_UTE = None  # None = all (~110)

# --- Cleaning definition ---------------------------------------------------
# Keep every connected component whose voxel count is >= KEEP_FRAC * (largest
# component). For lungs this keeps both lungs (similar size) and drops specks.
KEEP_FRAC = 0.10
MIN_KEEP = 1          # always keep at least the largest component
CONNECTIVITY = 3      # 26-connectivity (3x3x3 structuring element)
FILL_HOLES_3D = True  # fill fully-enclosed cavities (binary_fill_holes, 3D)
FILL_HOLES_2D_AXIAL = True  # also fill per-axial-slice holes (vessel cross-sections)

MAX_PCA_VOX = 200_000  # subsample cap for the PCA-based elongation

OUT_DIR = "./phase0_plots"
RUN_TAG = "mask_jepa_ct256"
RESULTS_CSV = os.path.join(OUT_DIR, f"verify_anatomy_{RUN_TAG}.csv")
OUT_PNG = os.path.join(OUT_DIR, f"verify_anatomy_{RUN_TAG}.png")

# Descriptors we verify (the energy drivers first, then context).
DESCRIPTORS = ["extent", "elongation", "solidity_close", "n_components", "volume_L"]
# Decision is based on these two (the energy-driving compactness axis).
KEY_DESCRIPTORS = ["extent", "elongation"]


# ===========================================================================
def _resolve(p):
    if PATH_REPLACE is not None:
        p = p.replace(PATH_REPLACE[0], PATH_REPLACE[1])
    return p


def _scans(csv, limit):
    files = [_resolve(f) for f in pd.read_csv(csv)["filepaths"].values]
    return files[:limit] if limit is not None else files


def load_mask(path):
    """Binary GT lung mask from the stored (H,W,D,2) volume (channel 1)."""
    arr = np.load(path)
    return (arr[:, :, :, 1] > 0).astype(np.uint8)


def _ball(r):
    L = np.arange(-r, r + 1)
    x, y, z = np.meshgrid(L, L, L, indexing="ij")
    return (x * x + y * y + z * z) <= r * r


def clean_mask(m):
    """Keep major connected component(s) + fill holes. Returns (clean, n_before)."""
    struct = np.ones((3, 3, 3)) if CONNECTIVITY == 3 else None
    lbl, n_before = ndi.label(m, structure=struct)
    if n_before == 0:
        return m.copy(), 0
    sizes = ndi.sum(np.ones_like(m), lbl, index=np.arange(1, n_before + 1))
    largest = sizes.max()
    keep_ids = np.where(sizes >= KEEP_FRAC * largest)[0] + 1
    if len(keep_ids) < MIN_KEEP:  # always keep the biggest
        keep_ids = np.array([int(np.argmax(sizes)) + 1])
    clean = np.isin(lbl, keep_ids)

    if FILL_HOLES_3D:
        clean = ndi.binary_fill_holes(clean)
    if FILL_HOLES_2D_AXIAL:
        for k in range(clean.shape[2]):
            sl = clean[:, :, k]
            if sl.any():
                clean[:, :, k] = ndi.binary_fill_holes(sl)
    return clean.astype(np.uint8), int(n_before)


def descriptors(m):
    """extent, elongation, solidity_close, n_components, volume_L for binary m."""
    m = m > 0
    vox = int(m.sum())
    out = {k: np.nan for k in DESCRIPTORS}
    if vox == 0:
        out["volume_L"] = 0.0
        out["n_components"] = 0
        return out

    out["volume_L"] = vox * (VOXEL_MM ** 3) / 1.0e6

    _, n = ndi.label(m, structure=np.ones((3, 3, 3)))
    out["n_components"] = int(n)

    mc = ndi.binary_closing(m, structure=_ball(3))
    Vc = int(mc.sum())
    out["solidity_close"] = vox / float(Vc) if Vc > 0 else np.nan

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
    ev = np.clip(np.linalg.eigvalsh(cov), 1e-8, None)
    out["elongation"] = float(np.sqrt(ev[-1] / ev[0]))
    return out


def cohens_d(ct, ute):
    """Standardised mean difference, UTE minus CT."""
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


def _verdict(d_raw, d_clean):
    if not (np.isfinite(d_raw) and np.isfinite(d_clean)) or abs(d_raw) < 1e-9:
        return "n/a", np.nan
    ratio = abs(d_clean) / abs(d_raw)
    if ratio >= 0.70:
        v = "ANATOMY (survives)"
    elif ratio <= 0.40:
        v = "ARTIFACT (collapses)"
    else:
        v = "PARTIAL"
    return v, ratio


def _collect(files, modality):
    rows = []
    for i, f in enumerate(files):
        try:
            m = load_mask(f)
        except Exception as e:
            print(f"  [skip] {f}: {e}")
            continue
        clean, n_before = clean_mask(m)
        raw_d = descriptors(m)
        cln_d = descriptors(clean)
        vol_removed = 1.0 - (cln_d["volume_L"] / raw_d["volume_L"]) \
            if raw_d["volume_L"] > 0 else 0.0
        # Note: cleaning fills holes (adds) and drops specks (removes); net can be +/-.
        row = dict(filepath=f, modality=modality, n_before=n_before,
                   vol_removed_frac=vol_removed)
        row.update({f"{k}_raw": raw_d[k] for k in DESCRIPTORS})
        row.update({f"{k}_clean": cln_d[k] for k in DESCRIPTORS})
        rows.append(row)
        print(f"  [{modality} {i+1}/{len(files)}] {os.path.basename(f)} "
              f"ncomp {n_before}->{cln_d['n_components']}  "
              f"extent {raw_d['extent']:.3f}->{cln_d['extent']:.3f}  "
              f"elong {raw_d['elongation']:.3f}->{cln_d['elongation']:.3f}")
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ct_files = _scans(CT_CSV, NUM_CT)
    ute_files = _scans(UTE_CSV, NUM_UTE)
    print(f"[verify_anatomy] CT={len(ct_files)} UTE={len(ute_files)} "
          f"(keep_frac={KEEP_FRAC}, fill3D={FILL_HOLES_3D}, fill2D={FILL_HOLES_2D_AXIAL})")

    rows = _collect(ct_files, "CT") + _collect(ute_files, "UTE")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"[saved] {RESULTS_CSV}")

    ct = df[df.modality == "CT"]
    ute = df[df.modality == "UTE"]

    print("\n========================= ANATOMY-vs-ARTIFACT =========================")
    print(f"UTE components (median): {ute.n_before.median():.0f} raw  ->  "
          f"{ute['n_components_clean'].median():.0f} after cleaning")
    print(f"UTE volume change from cleaning (median): "
          f"{100*ute.vol_removed_frac.median():+.2f}%  "
          f"(near 0 => specks were tiny, holes minor)")
    print("-" * 70)
    print(f"{'descriptor':<16}{'d_raw':>9}{'d_clean':>10}{'|clean/raw|':>13}"
          f"   verdict")
    summary = []
    for name in DESCRIPTORS:
        d_raw = cohens_d(ct[f"{name}_raw"], ute[f"{name}_raw"])
        d_clean = cohens_d(ct[f"{name}_clean"], ute[f"{name}_clean"])
        verdict, ratio = _verdict(d_raw, d_clean)
        summary.append(dict(descriptor=name, d_raw=d_raw, d_clean=d_clean,
                            ratio=ratio, verdict=verdict))
        print(f"{name:<16}{d_raw:>9.2f}{d_clean:>10.2f}{ratio:>13.2f}   {verdict}")
    print("-" * 70)

    key = [s for s in summary if s["descriptor"] in KEY_DESCRIPTORS]
    survives = all(s["verdict"].startswith("ANATOMY") for s in key)
    collapses = any(s["verdict"].startswith("ARTIFACT") for s in key)
    if survives:
        concl = ("VERDICT: the compactness axis is GENUINE ANATOMY. The "
                 "extent/elongation gap survives cleaning -> proceed with the "
                 "anisotropic + elastic shape augmentation for the mask-encoder "
                 "retrain (and clean masks as hygiene).")
    elif collapses:
        concl = ("VERDICT: the gap is largely an ARTIFACT of fragmented/holey UTE "
                 "masks. Fix = clean masks (largest components + fill holes); the "
                 "shape augmentation is NOT the right lever.")
    else:
        concl = ("VERDICT: PARTIAL -- some real anatomy + some artifact. Clean the "
                 "masks first, then re-run phase0_shape_vs_energy.py to see how much "
                 "energy gap remains before committing to augmentation.")
    print(concl)
    print("Next (separate, needs the model): recompute the ENERGY on the CLEANED "
          "masks and re-check rho(extent/elongation, energy) to confirm the energy\n"
          "      association is not itself driven by the specks.")
    print("=" * 70 + "\n")

    # ---- Figure: KEY descriptors, raw (top) vs cleaned (bottom) -------------
    nkey = len(KEY_DESCRIPTORS)
    fig, axes = plt.subplots(2, nkey, figsize=(5.2 * nkey, 8))
    axes = np.atleast_2d(axes)
    for col, name in enumerate(KEY_DESCRIPTORS):
        for row_i, suffix in enumerate(["raw", "clean"]):
            ax = axes[row_i, col]
            cvals = ct[f"{name}_{suffix}"].to_numpy()
            cvals = cvals[np.isfinite(cvals)]
            uvals = ute[f"{name}_{suffix}"].to_numpy()
            uvals = uvals[np.isfinite(uvals)]
            both = np.concatenate([cvals, uvals])
            bins = np.linspace(both.min(), both.max(), 30)
            ax.hist(cvals, bins=bins, color="tab:red", alpha=0.5, density=True,
                    label="CT")
            ax.hist(uvals, bins=bins, color="tab:blue", alpha=0.6, density=True,
                    label="UTE")
            d = cohens_d(cvals, uvals)
            ax.set_title(f"{name} ({suffix})   d={d:.2f}", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    fig.suptitle(f"Anatomy check: does CT-vs-UTE separation survive mask cleaning? "
                 f"({RUN_TAG})\ntop = raw masks, bottom = cleaned "
                 f"(largest comps + filled holes)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_PNG, dpi=150)
    print(f"[saved] {OUT_PNG}")


if __name__ == "__main__":
    main()
