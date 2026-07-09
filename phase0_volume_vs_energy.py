"""
Phase 0 diagnostic: does the mask-only energy have an absolute-SIZE attractor?

Motivation
----------
The morph sweeps showed opposite biases by modality: on CT the energy is lowest
for *eroded* (smaller) masks, on UTE it is lowest for *slightly dilated* (larger)
masks. The hypothesis is that the energy is minimised near a fixed absolute lung
size/occupancy learned from the (CT-only, scale-preserving pad_crop) training,
rather than at the per-scan correct boundary. CT GT sits above that preferred
size (wants to shrink); UTE GT sits below it (wants to grow).

This script tests that hypothesis directly by plotting, for the GROUND-TRUTH mask
of every scan (CT and UTE, ~800 total):

    x = lung volume   (physical, in litres at 1.25mm isotropic)
    y = mask-only V-JEPA energy of that GT mask

If the hypothesis holds we expect:
    * UTE GT volumes clearly smaller than CT GT volumes (the size gap), and
    * a pooled energy-vs-volume trend with a minimum at some fixed "preferred"
      volume, with UTE on the small/low-volume side and CT on the large side.

It reuses the energy definition from phase0_energy_vs_dice.py verbatim
(same checkpoint, same fit_mode, same fixed hide-patterns) so the numbers are
directly comparable to the other Phase-0 plots.

No argparse: edit the CONFIG block and run. Produces ONE figure + a results CSV.
"""

import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Reuse the exact energy machinery (model load, preprocessing, hide-patterns,
# energy reduction) so this diagnostic matches the other Phase-0 plots exactly.
from phase0_energy_vs_dice import (
    load_vjepa,
    precompute_mask_patterns,
    compute_energy,
    load_volume,
)


# ===========================================================================
# CONFIG -- change these, then run.
# ===========================================================================
# Which modality/modalities to plot: "CT", "UTE", or "both".
PLOT_MODALITY = "UTE"

# All CT (~690) and all UTE (~110) -> ~800 scans. These are the broad CSVs, not
# the held-out split: for a *size -> energy relationship* we want the full range
# of volumes, not an unbiased generalisation estimate.
CT_CSV = "./ids/only_copd_1.25mm.csv"
UTE_CSV = "./ids/only_ute_1.25mm.csv"

# Optional path remap if the CSV roots differ on the run machine, e.g.
# ("/Shared/", "/Volumes/"). Leave None to use the CSV paths verbatim (matches
# phase0_energy_vs_dice.py, which the user runs on the cluster as-is).
PATH_REPLACE = None

# Both modalities are resampled to 1.25mm isotropic, so a voxel is the same
# physical volume in CT and UTE and the counts are directly comparable.
VOXEL_MM = 1.25

# Number of scans per modality (None = all). Set small for a quick smoke test.
NUM_CT = None
NUM_UTE = None

# Binned-mean trend (to reveal the energy-vs-volume minimum) -- pooled bins.
NUM_BINS = 12
MIN_PER_BIN = 3

OUT_DIR = "./phase0_plots"
RUN_TAG = "mask_jepa_ct256"
# Output filenames carry the modality selection so CT-only / UTE-only / both runs
# do not overwrite each other.
RESULTS_CSV = os.path.join(OUT_DIR, f"volume_vs_energy_{PLOT_MODALITY}_{RUN_TAG}.csv")
OUT_PNG = os.path.join(OUT_DIR, f"volume_vs_energy_{PLOT_MODALITY}_{RUN_TAG}.png")


# ===========================================================================
def _resolve(p):
    if PATH_REPLACE is not None:
        p = p.replace(PATH_REPLACE[0], PATH_REPLACE[1])
    return p


def _scans(csv, limit):
    files = [_resolve(f) for f in pd.read_csv(csv)["filepaths"].values]
    return files[:limit] if limit is not None else files


def _collect(model, input_size, fit_mode, patterns, files, modality):
    """Per scan: GT lung volume (voxels + litres) and the GT mask energy."""
    rows = []
    for i, f in enumerate(files):
        try:
            _img, gt = load_volume(f)
        except Exception as e:  # missing file / bad path -> skip, keep going
            print(f"  [skip] {f}: {e}")
            continue
        vox = float((gt > 0.5).sum())
        vol_L = vox * (VOXEL_MM ** 3) / 1.0e6  # mm^3 -> litres
        energy = compute_energy(model, gt, fit_mode, input_size, patterns)
        rows.append(
            dict(filepath=f, modality=modality, volume_vox=vox,
                 volume_L=vol_L, energy=energy)
        )
        print(f"  [{modality} {i+1}/{len(files)}] {os.path.basename(f)} "
              f"vol={vol_L:.3f} L  energy={energy:.4f}")
    return rows


def _binned_trend(vols, energies, num_bins, min_per_bin):
    """Pooled mean energy per equal-width volume bin (reveals any minimum)."""
    if len(vols) == 0:
        return np.array([]), np.array([])
    edges = np.linspace(vols.min(), vols.max(), num_bins + 1)
    idx = np.digitize(vols, edges)
    centers, means = [], []
    for b in range(1, num_bins + 1):
        m = idx == b
        if m.sum() >= min_per_bin:
            centers.append(vols[m].mean())
            means.append(energies[m].mean())
    return np.array(centers), np.array(means)


def main():
    if PLOT_MODALITY not in ("CT", "UTE", "both"):
        raise ValueError(f"PLOT_MODALITY must be 'CT', 'UTE' or 'both', got {PLOT_MODALITY!r}")
    do_ct = PLOT_MODALITY in ("CT", "both")
    do_ute = PLOT_MODALITY in ("UTE", "both")

    os.makedirs(OUT_DIR, exist_ok=True)

    model, input_size, grid, fit_mode = load_vjepa()
    patterns = precompute_mask_patterns(grid)

    ct_files = _scans(CT_CSV, NUM_CT) if do_ct else []
    ute_files = _scans(UTE_CSV, NUM_UTE) if do_ute else []
    print(f"[volume_vs_energy] modality={PLOT_MODALITY} "
          f"CT={len(ct_files)} UTE={len(ute_files)} fit_mode={fit_mode}")

    rows = []
    if do_ct:
        rows += _collect(model, input_size, fit_mode, patterns, ct_files, "CT")
    if do_ute:
        rows += _collect(model, input_size, fit_mode, patterns, ute_files, "UTE")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"[saved] {RESULTS_CSV}")

    ct = df[df.modality == "CT"]
    ute = df[df.modality == "UTE"]

    # Correlations between volume and energy (a strong relationship of EITHER sign
    # means the energy is reading size rather than per-scan correctness).
    def _rho(d):
        if len(d) > 2 and d.volume_L.nunique() > 1:
            return spearmanr(d.volume_L, d.energy).correlation
        return float("nan")

    rho_ct, rho_ute, rho_all = _rho(ct), _rho(ute), _rho(df)

    # Pooled binned trend + the "preferred" volume (where pooled energy is lowest).
    vols = df.volume_L.to_numpy()
    energies = df.energy.to_numpy()
    bc, bm = _binned_trend(vols, energies, NUM_BINS, MIN_PER_BIN)
    pref_vol = bc[int(np.argmin(bm))] if len(bm) else float("nan")

    print("\n================ SUMMARY ================")
    if do_ct and len(ct):
        print(f"CT  : n={len(ct)}  vol med={ct.volume_L.median():.3f} L  "
              f"energy med={ct.energy.median():.4f}  Spearman(vol,E)={rho_ct:.3f}")
    if do_ute and len(ute):
        print(f"UTE : n={len(ute)} vol med={ute.volume_L.median():.3f} L  "
              f"energy med={ute.energy.median():.4f}  Spearman(vol,E)={rho_ute:.3f}")
    if do_ct and do_ute:
        print(f"ALL : Spearman(vol,E)={rho_all:.3f}")
    print(f"Pooled energy minimised near volume = {pref_vol:.3f} L "
          f"(the 'preferred size' if the attractor hypothesis holds)")
    print("=========================================\n")

    # ---- Figure: (left) energy vs volume scatter + trend, (right) volume hist --
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    if do_ct and len(ct):
        ax.scatter(ct.volume_L, ct.energy, c="tab:red", s=28, alpha=0.55,
                   label=f"CT GT (n={len(ct)})")
    if do_ute and len(ute):
        ax.scatter(ute.volume_L, ute.energy, c="tab:blue", s=34, alpha=0.7,
                   label=f"UTE GT (n={len(ute)})")
    if len(bc):
        ax.plot(bc, bm, "-k", lw=2.2, marker="s", ms=5,
                label="pooled binned-mean energy")
        ax.axvline(pref_vol, color="k", ls="--", alpha=0.5,
                   label=f"preferred vol ~{pref_vol:.2f} L")
    ax.set_xlabel("lung volume (litres, 1.25mm isotropic)")
    ax.set_ylabel("mask-only V-JEPA energy")
    ax.set_title("Energy vs lung volume (GT masks)\n"
                 f"Spearman vol-vs-E: CT={rho_ct:.3f}  UTE={rho_ute:.3f}  "
                 f"all={rho_all:.3f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    lo, hi = df.volume_L.min(), df.volume_L.max()
    bins = np.linspace(lo, hi, 30)
    if do_ct and len(ct):
        ax.hist(ct.volume_L, bins=bins, color="tab:red", alpha=0.5,
                label=f"CT (med {ct.volume_L.median():.2f} L)")
        ax.axvline(ct.volume_L.median(), color="tab:red", ls="--", alpha=0.8)
    if do_ute and len(ute):
        ax.hist(ute.volume_L, bins=bins, color="tab:blue", alpha=0.6,
                label=f"UTE (med {ute.volume_L.median():.2f} L)")
        ax.axvline(ute.volume_L.median(), color="tab:blue", ls="--", alpha=0.8)
    if len(bc):
        ax.axvline(pref_vol, color="k", ls="--", alpha=0.7,
                   label=f"preferred vol ~{pref_vol:.2f} L")
    ax.set_xlabel("lung volume (litres)")
    ax.set_ylabel("scan count")
    ax.set_title("Lung-volume distribution by modality\n"
                 "(does UTE sit below CT, around the preferred volume?)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Phase 0 size-attractor diagnostic | {PLOT_MODALITY} | "
                 f"mask-only energy ({RUN_TAG})", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"[saved] {OUT_PNG}")


if __name__ == "__main__":
    main()
