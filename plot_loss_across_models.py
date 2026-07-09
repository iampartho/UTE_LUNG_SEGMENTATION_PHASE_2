"""
Plot training/test loss curves across multiple models.

Each model is described by a dict with:
    - "name":      Display label for the legend.
    - "train_log": Path to a train metrics CSV (must have an `epoch`/`step`
                   column and a `loss` column). Optional; can be None.
    - "test_log":  Path to a test  metrics CSV (must have an `epoch`/`step`
                   column and a `loss` column). Optional; can be None.

The script supports three layouts:
    - "overlay":  all train + test curves drawn on a single axis (one PNG).
                  Train is solid, test is dashed; one color per model.
    - "split":    two side-by-side subplots in one figure (one PNG).
    - "separate": train and test written as TWO separate PNG files. Filenames
                  are derived from `output_path` by inserting "_train" and
                  "_test" before the extension.
"""

import argparse
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Column inference helpers (kept self-contained so this file is standalone).
# ---------------------------------------------------------------------------

def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _infer_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    norm = {c: _normalize(c) for c in df.columns}
    cand_norm = [_normalize(c) for c in candidates]
    for col, ncol in norm.items():
        if ncol in cand_norm:
            return col
    for col, ncol in norm.items():
        if any(c in ncol for c in cand_norm if len(c) >= 4):
            return col
    return None


def _infer_x(df: pd.DataFrame) -> Optional[str]:
    return _infer_column(df, ["epoch", "step", "iteration", "iter", "batch", "round"])


def _infer_loss(df: pd.DataFrame) -> Optional[str]:
    col = _infer_column(df, ["loss"])
    if col:
        return col
    for c in df.columns:
        if "loss" in _normalize(c):
            return c
    return None


def _read_loss_curve(
    csv_path: Optional[str],
    smoothing: Optional[int],
) -> Tuple[Optional[List[float]], Optional[List[float]], Optional[str]]:
    """Return (x, y, x_label) from a CSV, or (None, None, None) if not usable."""
    if not csv_path:
        return None, None, None
    if not os.path.isfile(csv_path):
        print(f"[warn] file not found, skipping: {csv_path}")
        return None, None, None

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[warn] empty CSV, skipping: {csv_path}")
        return None, None, None

    x_col = _infer_x(df)
    y_col = _infer_loss(df)
    if y_col is None:
        print(f"[warn] no loss column found in {csv_path}; columns={list(df.columns)}")
        return None, None, None

    y = pd.to_numeric(df[y_col], errors="coerce")
    if x_col is not None:
        x = pd.to_numeric(df[x_col], errors="coerce")
        if x.isna().any():
            x = pd.Series(range(len(df)))
    else:
        x = pd.Series(range(len(df)))
        x_col = "index"

    mask = ~(x.isna() | y.isna())
    x = x[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    if smoothing and smoothing > 1 and len(y) >= smoothing:
        y = y.rolling(window=smoothing, min_periods=1, center=False).mean()

    return x.tolist(), y.tolist(), x_col


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def _draw_split_curves(
    ax,
    resolved: List[Dict],
    split: str,
    linestyle: str,
    show_split_in_label: bool,
) -> None:
    """Plot one of {"train", "test"} on `ax` for every resolved model entry."""
    for entry in resolved:
        x, y = entry[split]
        if x is None:
            continue
        label = f"{entry['name']} - {split}" if show_split_in_label else entry["name"]
        ax.plot(x, y, color=entry["color"], linestyle=linestyle, linewidth=1.6, label=label)


def _decorate_axis(
    ax,
    x_label: str,
    subtitle: Optional[str],
    log_y: bool,
) -> None:
    if subtitle:
        ax.set_title(subtitle)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    if log_y:
        ax.set_yscale("log")
    if ax.has_data():
        ax.legend(fontsize=9, loc="best")


def _split_output_path(output_path: str) -> Tuple[str, str]:
    """Insert _train / _test before the extension of output_path."""
    base, ext = os.path.splitext(output_path)
    if not ext:
        ext = ".png"
    return f"{base}_train{ext}", f"{base}_test{ext}"


def plot_loss_across_models(
    models: List[Dict[str, Optional[str]]],
    output_path: str = "results_plots/loss_across_models.png",
    layout: str = "overlay",
    smoothing: Optional[int] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    log_y: bool = False,
    colors: Optional[Sequence[str]] = None,
):
    """
    Plot loss curves across multiple models.

    Parameters
    ----------
    models : list of dict
        Each entry: {"name": str, "train_log": path or None, "test_log": path or None}.
    output_path : str
        Output PNG path. For layout="separate", two files are written and the
        filenames are derived by inserting "_train" / "_test" before the
        extension.
    layout : {"overlay", "split", "separate"}
        "overlay"  - one axis with train (solid) + test (dashed) per model.
        "split"    - two side-by-side axes in a single figure (one PNG).
        "separate" - train and test as two distinct PNG files.
    smoothing : int or None
        Rolling-mean window size applied to each curve. None or <=1 disables it.
    title : str or None
        Figure title; auto-generated if None.
    figsize : (w, h) or None
        Figure size in inches. Sensible defaults are picked per layout.
    log_y : bool
        If True, use log scale on the loss axis.
    colors : sequence of color specs or None
        Override the per-model color cycle.

    Returns
    -------
    str or List[str]
        Path(s) of the PNG file(s) written. A single string for "overlay"/"split",
        a list of two strings for "separate".
    """
    if layout not in {"overlay", "split", "separate"}:
        raise ValueError(
            f"layout must be 'overlay', 'split', or 'separate', got {layout!r}"
        )
    if not models:
        raise ValueError("`models` must be a non-empty list.")

    if layout == "separate":
        out_train, out_test = _split_output_path(output_path)
        for p in (out_train, out_test):
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    else:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Build a stable color list, one per model.
    if colors is None:
        prop = plt.rcParams.get("axes.prop_cycle")
        cycle = prop.by_key().get("color", []) if prop is not None else []
        if not cycle:
            cycle = [f"C{i}" for i in range(10)]
        colors = [cycle[i % len(cycle)] for i in range(len(models))]

    # Resolve all curves first so we can pick a sensible x-axis label.
    resolved: List[Dict] = []
    x_labels = set()
    for model, color in zip(models, colors):
        name = model.get("name") or "model"
        tr_x, tr_y, tr_xlab = _read_loss_curve(model.get("train_log"), smoothing)
        te_x, te_y, te_xlab = _read_loss_curve(model.get("test_log"), smoothing)
        if tr_xlab:
            x_labels.add(tr_xlab)
        if te_xlab:
            x_labels.add(te_xlab)
        resolved.append({
            "name": name, "color": color,
            "train": (tr_x, tr_y), "test": (te_x, te_y),
        })

    x_label = (next(iter(x_labels)) if len(x_labels) == 1 else "epoch / step").capitalize()

    smoothed_suffix = f" (smoothed window={smoothing})" if smoothing and smoothing > 1 else ""
    base_title = title if title is not None else f"Loss across models{smoothed_suffix}"

    if layout == "overlay":
        fig, ax = plt.subplots(figsize=figsize or (10, 6))
        _draw_split_curves(ax, resolved, "train", "-",  show_split_in_label=True)
        _draw_split_curves(ax, resolved, "test",  "--", show_split_in_label=True)
        _decorate_axis(ax, x_label, None, log_y)
        fig.suptitle(base_title, y=0.995)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return output_path

    if layout == "split":
        fig, (train_ax, test_ax) = plt.subplots(
            1, 2, figsize=figsize or (14, 6), sharey=True
        )
        _draw_split_curves(train_ax, resolved, "train", "-", show_split_in_label=False)
        _draw_split_curves(test_ax,  resolved, "test",  "-", show_split_in_label=False)
        _decorate_axis(train_ax, x_label, "Train loss", log_y)
        _decorate_axis(test_ax,  x_label, "Test loss",  log_y)
        fig.suptitle(base_title, y=0.995)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return output_path

    # layout == "separate": two PNGs.
    out_train, out_test = _split_output_path(output_path)
    written: List[str] = []
    for split, out_p, suffix in (
        ("train", out_train, "Train loss"),
        ("test",  out_test,  "Test loss"),
    ):
        fig, ax = plt.subplots(figsize=figsize or (10, 6))
        _draw_split_curves(ax, resolved, split, "-", show_split_in_label=False)
        _decorate_axis(ax, x_label, None, log_y)
        fig.suptitle(f"{suffix} across models{smoothed_suffix}" if title is None else f"{base_title} - {suffix}", y=0.995)
        fig.tight_layout()
        fig.savefig(out_p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        written.append(out_p)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_cli(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot loss curves across multiple models.")
    p.add_argument(
        "--model", "-m",
        action="append",
        nargs=3,
        metavar=("NAME", "TRAIN_LOG", "TEST_LOG"),
        help="Add a model. Pass '-' for either log path to skip it. Repeatable.",
    )
    p.add_argument("--output", "-o", default="results_plots/loss_across_models.png",
                   help="Output PNG path.")
    p.add_argument("--layout", choices=["overlay", "split", "separate"], default="overlay",
                   help="overlay = single axis; split = two subplots in one PNG; "
                        "separate = two PNG files (train, test).")
    p.add_argument("--smoothing", type=int, default=0,
                   help="Rolling mean window (<=1 disables smoothing).")
    p.add_argument("--title", default=None)
    p.add_argument("--log-y", action="store_true", help="Use log scale on the loss axis.")
    return p.parse_args(argv)


def _main(argv: Optional[List[str]] = None) -> None:
    args = _parse_cli(argv)
    if not args.model:
        raise SystemExit(
            "No models passed. Use --model NAME TRAIN_LOG TEST_LOG (repeatable). "
            "Pass '-' to omit train_log or test_log."
        )

    models: List[Dict[str, Optional[str]]] = []
    for name, train_log, test_log in args.model:
        models.append({
            "name": name,
            "train_log": None if train_log == "-" else train_log,
            "test_log":  None if test_log  == "-" else test_log,
        })

    out = plot_loss_across_models(
        models=models,
        output_path=args.output,
        layout=args.layout,
        smoothing=args.smoothing if args.smoothing and args.smoothing > 1 else None,
        title=args.title,
        log_y=args.log_y,
    )
    if isinstance(out, list):
        for p in out:
            print(f"Saved: {p}")
    else:
        print(f"Saved: {out}")


if __name__ == "__main__":
    # If invoked with CLI args, use the CLI; otherwise run a small example
    # against the existing logs in `log/` to make this script self-demoing.
    import sys

    if len(sys.argv) > 1:
        _main()
    else:
        example_models = [
            {
                "name": "65 RE 50 GIN",
                "train_log": "log/train_metrics_unet_td1_lc_monitor_65_RE_50_GIN.csv",
                "test_log":  "log/test_metrics_unet_td1_lc_monitor_65_RE_50_GIN.csv",
            },
            {
                "name": "65 RE 50 GIN GroupNorm",
                "train_log": "log/train_metrics_unet_td1_lc_monitor_65_RE_50_GIN_groupnorm.csv",
                "test_log":  "log/test_metrics_unet_td1_lc_monitor_65_RE_50_GIN_groupnorm.csv",
            },
            {
                "name": "Current Best Model(70 RE 80 GIN)",
                "train_log": "log/train_metrics_unet_causality_paper_ct_train_UTE_test_w_tversky_wo_kl_only_gin_td1_roughness_enforced.csv",
                "test_log":  "log/test_metrics_unet_causality_papery_ct_train_UTE_test_w_tversky_wo_kl_only_gin_td1_roughness_enforced.csv",
            },
            {
                "name": "70 RE 80 GIN GroupNorm",
                "train_log": "log/train_metrics_unet_td1_lc_monitor_70_RE_80_GIN_groupnorm.csv",
                "test_log":  "log/test_metrics_unet_td1_lc_monitor_70_RE_80_GIN_groupnorm.csv",
            },
            # {
            #     "name": "70 RE 80 GIN GroupNorm",
            #     "train_log": "log/train_metrics_unet_td1_lc_monitor_70_RE_80_GIN_groupnorm.csv",
            #     "test_log":  "log/test_metrics_unet_td1_lc_monitor_70_RE_80_GIN_groupnorm.csv",
            # },
            # {
            #     "name": "70 RE 80 GIN GroupNorm with Varying Groups",
            #     "train_log": "log/train_metrics_unet_td1_lc_monitor_70_RE_80_GIN_groupnorm_with_varying_groups.csv",
            #     "test_log":  "log/test_metrics_unet_td1_lc_monitor_70_RE_80_GIN_groupnorm_with_varying_groups.csv",
            # }
        ]
        out = plot_loss_across_models(
            models=example_models,
            output_path="results_plots/loss_across_models_example.png",
            layout="separate",
            smoothing=5,
        )
        if isinstance(out, list):
            for p in out:
                print(f"Saved: {p}")
        else:
            print(f"Saved: {out}")
