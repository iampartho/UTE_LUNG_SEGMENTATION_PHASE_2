"""
Visualise how much the Phase-2 cross-modal guidance helps the *hard* scans.

Takes one (or several) metrics CSV(s) written by phase2_guidance.py and a Dice
THRESHOLD, keeps every scan whose BASELINE (m0) Dice is below the threshold, and
draws a side-by-side dumbbell figure:

    LEFT  subplot : baseline Dice   ->  refined Dice    (higher is better, --->)
    RIGHT subplot : baseline energy ->  refined energy  (lower  is better, <---)

Each scan is one row, shared across both subplots, so you can read "this scan's
Dice jumped AND its energy dropped" straight across. Rows are sorted worst-Dice
first so the cases the method is meant to rescue sit together at the top.

The colour of every connector encodes whether the move went the *good* way
(green) or the *bad* way (red); scans the pipeline declined to refine
(fell_back == True, refined == m0) are drawn hollow/grey so they read as "no
action taken". Per-scan deltas are annotated.

CSV columns used (auto-detected):
    dice_m0 / dice_refined            (1.25mm mode)   -- OR --
    dice_m0_1mm / dice_refined_1mm    (1mm mode)
    E0, E_final, scan, fell_back, reason

USAGE
    python plot_phase2_low_dice.py METRICS.csv --threshold 0.90
    python plot_phase2_low_dice.py a.csv b.csv c.csv -t 0.88 -o low_dice.png
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # headless / cluster-safe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Colours: good move (Dice up / energy down) vs bad move, plus "no action".
C_GOOD = "#2ca02c"      # green
C_BAD = "#d62728"       # red
C_NONE = "#9e9e9e"      # grey  (fell back -> refined == m0)
C_M0 = "#5a5a5a"        # baseline marker
C_REF_GOOD = "#1a7d1a"
C_REF_BAD = "#a31515"


def _unify_dice_columns(df):
    """Return df with canonical 'dice_m0' / 'dice_refined' columns regardless of
    whether the CSV was written in 1mm or 1.25mm mode."""
    if "dice_m0" in df and "dice_refined" in df:
        pass
    elif "dice_m0_1mm" in df and "dice_refined_1mm" in df:
        df = df.rename(columns={"dice_m0_1mm": "dice_m0",
                                "dice_refined_1mm": "dice_refined"})
    else:
        raise ValueError("CSV has no recognised Dice columns "
                         "(need dice_m0/dice_refined or *_1mm variants).")
    return df


def load_csvs(paths):
    frames = []
    for p in paths:
        d = _unify_dice_columns(pd.read_csv(p))
        if "fell_back" not in d:
            d["fell_back"] = False
        if "reason" not in d:
            d["reason"] = ""
        d["__source__"] = os.path.basename(p)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def short_label(scan, source, multi_source):
    """A compact, readable y-tick label for a scan."""
    s = str(scan)
    # Strip the long shared prefixes the pipeline bakes into the stem.
    for junk in ("parthghosh_data_UTE_previous_data_numpy_without_clipping_",
                 "UTE_new_data_numpy_"):
        s = s.replace(junk, "")
    s = s.replace("_AnatCorrLungs", "")
    if len(s) > 28:
        s = s[:27] + "…"
    return f"{s}" if not multi_source else f"{s}  [{source.split('.')[0][-8:]}]"


def make_figure(df, threshold, out_path):
    df = _unify_dice_columns(df).copy()
    low = df[df["dice_m0"] < threshold].copy()
    if low.empty:
        print(f"No scans with baseline Dice < {threshold:.3f}. Nothing to plot.")
        return

    # Worst baseline at the TOP: sort ascending, then reverse for matplotlib's
    # bottom-up y-axis so row 0 (worst) ends up at the top.
    low = low.sort_values("dice_m0", ascending=False).reset_index(drop=True)
    n = len(low)
    y = np.arange(n)

    multi = low["__source__"].nunique() > 1
    labels = [short_label(s, src, multi)
              for s, src in zip(low["scan"], low["__source__"])]

    dice_up = low["dice_refined"].values - low["dice_m0"].values
    e_down = low["E0"].values - low["E_final"].values       # +ve = energy dropped (good)
    took_action = ~low["fell_back"].values

    fig, (axd, axe) = plt.subplots(
        1, 2, figsize=(13, max(3.0, 0.55 * n + 1.5)), sharey=True)

    # ---- LEFT: Dice (higher = better, arrow points right when improved) -------
    for i in range(n):
        m0, rf = low["dice_m0"].values[i], low["dice_refined"].values[i]
        if not took_action[i]:
            col, refcol = C_NONE, C_NONE
        else:
            good = rf > m0
            col = C_GOOD if good else C_BAD
            refcol = C_REF_GOOD if good else C_REF_BAD
        axd.annotate(
            "", xy=(rf, y[i]), xytext=(m0, y[i]),
            arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2,
                            shrinkA=0, shrinkB=0))
        axd.scatter(m0, y[i], s=55, color=C_M0, zorder=3,
                    edgecolor="white", linewidth=0.6)
        axd.scatter(rf, y[i], s=70, color=refcol, zorder=4,
                    edgecolor="white", linewidth=0.6,
                    facecolor=refcol if took_action[i] else "white")
        # Delta label at the refined end.
        dx = 0.004 if rf >= m0 else -0.004
        axd.text(rf + dx, y[i], f"{dice_up[i]:+.3f}", va="center",
                 ha="left" if rf >= m0 else "right", fontsize=8,
                 color=refcol)

    axd.set_yticks(y)
    axd.set_yticklabels(labels, fontsize=8)
    axd.set_xlabel("Dice  (baseline ●  →  refined)")
    axd.set_title(f"Dice on hard scans (baseline < {threshold:.2f})", fontsize=11)
    axd.grid(axis="x", ls=":", alpha=0.5)
    axd.axvline(threshold, color="k", ls="--", lw=0.8, alpha=0.5)

    # ---- RIGHT: Energy (lower = better, arrow points left when improved) -------
    for i in range(n):
        e0, ef = low["E0"].values[i], low["E_final"].values[i]
        if not took_action[i]:
            col, refcol = C_NONE, C_NONE
        else:
            good = ef < e0
            col = C_GOOD if good else C_BAD
            refcol = C_REF_GOOD if good else C_REF_BAD
        axe.annotate(
            "", xy=(ef, y[i]), xytext=(e0, y[i]),
            arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2,
                            shrinkA=0, shrinkB=0))
        axe.scatter(e0, y[i], s=55, color=C_M0, zorder=3,
                    edgecolor="white", linewidth=0.6)
        axe.scatter(ef, y[i], s=70, color=refcol, zorder=4,
                    edgecolor="white", linewidth=0.6,
                    facecolor=refcol if took_action[i] else "white")
        dx = -0.004 if ef <= e0 else 0.004
        axe.text(ef + dx, y[i], f"{-e_down[i]:+.3f}", va="center",
                 ha="right" if ef <= e0 else "left", fontsize=8,
                 color=refcol)

    axe.set_xlabel("Cross-modal energy  (baseline ●  →  refined)")
    axe.set_title("Energy on the same scans (lower = better)", fontsize=11)
    axe.grid(axis="x", ls=":", alpha=0.5)
    axe.invert_yaxis()                       # worst-Dice scan on top

    # ---- shared legend + headline summary -------------------------------------
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_M0,
               markersize=9, label="baseline (m0)"),
        Line2D([0], [0], color=C_GOOD, lw=2.5, label="moved the good way"),
        Line2D([0], [0], color=C_BAD, lw=2.5, label="moved the bad way"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor=C_NONE, markersize=9,
               label="no action (fell back to m0)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    n_imp = int((dice_up > 1e-4).sum())
    n_wor = int((dice_up < -1e-4).sum())
    fig.suptitle(
        f"Phase-2 guidance on hard scans  |  N={n} (baseline Dice < {threshold:.2f})  "
        f"|  mean Dice {low['dice_m0'].mean():.3f} → {low['dice_refined'].mean():.3f} "
        f"({dice_up.mean():+.3f})  |  improved {n_imp}, worsened {n_wor}  "
        f"|  mean energy {low['E0'].mean():.3f} → {low['E_final'].mean():.3f}",
        fontsize=11, y=1.00)

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved -> {out_path}")

    # Console echo so the figure is also explainable from the terminal.
    print(f"\n{n} scans with baseline Dice < {threshold:.2f} "
          f"(worst first):")
    for i in range(n):
        flag = "" if took_action[i] else "  [no action]"
        print(f"  {labels[i]:32s}  Dice {low['dice_m0'].values[i]:.3f} -> "
              f"{low['dice_refined'].values[i]:.3f} ({dice_up[i]:+.3f})   "
              f"E {low['E0'].values[i]:.3f} -> {low['E_final'].values[i]:.3f} "
              f"({-e_down[i]:+.3f}){flag}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="phase2_guidance metrics CSV(s)")
    ap.add_argument("-t", "--threshold", type=float, default=0.90,
                    help="keep scans whose BASELINE Dice is below this (default 0.90)")
    ap.add_argument("-o", "--out", default=None,
                    help="output PNG path (default: phase2_low_dice_<thr>.png "
                         "next to the first CSV)")
    args = ap.parse_args()

    df = load_csvs(args.csv)
    if args.out is None:
        base = os.path.dirname(os.path.abspath(args.csv[0]))
        args.out = os.path.join(
            base, f"phase2_low_dice_thr{args.threshold:.2f}.png")
    make_figure(df, args.threshold, args.out)


if __name__ == "__main__":
    main()
