import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Helpers
# =========================

def _normalize_colname(name: str) -> str:
    """Lowercase and remove non-alphanumeric for robust matching."""
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


def _find_first_matching_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Return the first column in df whose name matches any candidate.
    Strategy:
      1) Exact match on normalized names.
      2) Substring fallback ONLY for candidates with length >= 4
         (prevents short tokens like 'rec' matching 'precision').
    """
    if df is None or df.empty:
        return None

    # Original and normalized maps
    norm_map = {col: _normalize_colname(col) for col in df.columns}
    cand_norm = [_normalize_colname(c) for c in candidates]

    # Exact normalized match
    for col, ncol in norm_map.items():
        if ncol in cand_norm:
            return col

    # Substring fallback (only for reasonably long aliases to avoid collisions)
    long_cands = [c for c in cand_norm if len(c) >= 4]
    if long_cands:
        for col, ncol in norm_map.items():
            if any(c in ncol for c in long_cands):
                return col

    return None


def _infer_x_axis(df: Optional[pd.DataFrame]) -> Optional[str]:
    """Infer x-axis column (epoch/step/iteration/iter/batch/round)."""
    if df is None:
        return None
    return _find_first_matching_column(df, ["epoch", "step", "iteration", "iter", "batch", "round"])


def _infer_loss_column(df: Optional[pd.DataFrame]) -> Optional[str]:
    """Infer a loss column; prefer 'loss' or any column containing 'loss'."""
    if df is None:
        return None
    exact = _find_first_matching_column(df, ["loss"])
    if exact is not None:
        return exact
    for col in df.columns:
        if "loss" in _normalize_colname(col):
            return col
    return None


def _prepare_xy(df: pd.DataFrame, x_col: Optional[str], y_col: str) -> Tuple[List[float], List[float]]:
    """Return numeric x and y for plotting; fallback to simple index if needed."""
    y = pd.to_numeric(df[y_col], errors="coerce")
    if x_col and x_col in df.columns:
        x = pd.to_numeric(df[x_col], errors="coerce")
        if x.isna().any():
            x = list(range(len(df)))
        else:
            x = x.tolist()
    else:
        x = list(range(len(df)))
    return x, y.tolist()


def _detect_metrics(
    df: Optional[pd.DataFrame],
    extra_aliases: Optional[Dict[str, List[str]]] = None
) -> Dict[str, str]:
    """
    Detect common metric columns. Returns a map {canonical_metric_name -> actual_column_name}.
    Canonical set includes: accuracy, precision, recall, f1, iou, dice.
    """
    if df is None or df.empty:
        return {}

    # IMPORTANT: No short 'rec' token to avoid matching 'precision'
    aliases = {
        "accuracy": ["accuracy", "acc", "top1"],
        "precision": ["precision", "prec", "ppv"],
        "recall": ["recall", "tpr", "sensitivity"],
        "f1": ["f1", "f1score", "f1_score", "f1macro", "f1micro"],
        "iou": ["iou", "miou", "jaccard", "jaccardindex", "intersectionoverunion"],
        "dice": ["dice", "dsc", "sorensen", "f1_dice"],
    }
    if extra_aliases:
        for k, v in extra_aliases.items():
            k = k.lower()
            aliases.setdefault(k, [])
            # Deduplicate while preserving
            aliases[k] = list(dict.fromkeys(aliases[k] + v))

    found = {}
    for canon, cands in aliases.items():
        col = _find_first_matching_column(df, cands)
        if col:
            found[canon] = col
    return found


# =========================
# 1) Single Figure with Two Subplots: Loss + Metrics
# =========================

def plot_training_and_test_logs(
    train_csv: Optional[str],
    test_csv: Optional[str] = None,
    output_dir: str = "results_plots",
    figure_prefix: Optional[str] = None,
    extra_metric_aliases: Optional[Dict[str, List[str]]] = None,
    desired_metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> str:
    """
    Create a single figure with TWO SUBPLOTS:
      - Subplot(2,1,1): Loss curves (Train & Test if available)
      - Subplot(2,1,2): Metrics curves (auto-detected; e.g., accuracy/precision/recall/f1/iou/dice)
        * Only plots metrics that exist in either train or test CSV.
    """
    os.makedirs(output_dir, exist_ok=True)

    df_train = pd.read_csv(train_csv) if train_csv else None
    df_test = pd.read_csv(test_csv) if test_csv else None

    # Figure filename
    if figure_prefix:
        prefix = figure_prefix
    else:
        parts = []
        if train_csv:
            parts.append(os.path.splitext(os.path.basename(train_csv))[0])
        if test_csv:
            parts.append(os.path.splitext(os.path.basename(test_csv))[0])
        prefix = "__vs__".join(parts) if parts else "training_progress"
    out_path = os.path.join(output_dir, f"{prefix}_loss_metrics.png")

    # X axes
    x_train_col = _infer_x_axis(df_train)
    x_test_col = _infer_x_axis(df_test)
    x_label = x_train_col or x_test_col or "Index"

    # Loss columns
    loss_train_col = _infer_loss_column(df_train)
    loss_test_col = _infer_loss_column(df_test)

    # Metrics present
    metrics_train = _detect_metrics(df_train, extra_aliases=extra_metric_aliases)
    metrics_test = _detect_metrics(df_test, extra_aliases=extra_metric_aliases)

    # Canonical plotting order
    if desired_metrics is None:
        desired_metrics = ["accuracy", "precision", "recall", "f1", "iou", "dice"]

    metrics_to_plot = [
        m for m in desired_metrics
        if (m in metrics_train) or (m in metrics_test)
    ]

    # ------------- Plotting -------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=False)

    # Subplot 1: Loss
    if df_train is not None and loss_train_col:
        x, y = _prepare_xy(df_train, x_train_col, loss_train_col)
        ax1.plot(x, y, label=f"Train ({loss_train_col})")
    if df_test is not None and loss_test_col:
        x, y = _prepare_xy(df_test, x_test_col, loss_test_col)
        ax1.plot(x, y, label=f"Test ({loss_test_col})")
    ax1.set_title("Loss")
    ax1.set_xlabel(x_label.capitalize())
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax1.legend()

    # Subplot 2: Metrics (auto-detected)
    if metrics_to_plot:
        for m in metrics_to_plot:
            if df_train is not None and m in metrics_train:
                x, y = _prepare_xy(df_train, x_train_col, metrics_train[m])
                ax2.plot(x, y, label=f"Train {m.capitalize()}")
            if df_test is not None and m in metrics_test:
                x, y = _prepare_xy(df_test, x_test_col, metrics_test[m])
                ax2.plot(x, y, label=f"Test {m.capitalize()}")
    else:
        # Fallback: plot any other numeric columns (excluding loss & x)
        def numeric_cols(df):
            if df is None:
                return []
            out = []
            for col in df.columns:
                if col in {x_train_col, x_test_col, loss_train_col, loss_test_col}:
                    continue
                ser = pd.to_numeric(df[col], errors="coerce")
                if ser.notna().any():
                    out.append(col)
            return out

        for col in numeric_cols(df_train)[:6]:
            x, y = _prepare_xy(df_train, x_train_col, col)
            ax2.plot(x, y, label=f"Train {col}")
        for col in numeric_cols(df_test)[:6]:
            x, y = _prepare_xy(df_test, x_test_col, col)
            ax2.plot(x, y, label=f"Test {col}")

    ax2.set_title("Metrics")
    ax2.set_xlabel(x_label.capitalize())
    ax2.set_ylabel("Metric value")
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax2.legend(ncol=2)

    fig.suptitle("Training/Test Progress", y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    return out_path


# =========================
# 2) Flexible Bar Plot for Manually Given Best Test Metrics
# =========================

def plot_best_metrics_bar(
    models: List[Dict[str, Dict[str, float]]],
    output_path: str = "plots/best_metrics_bar.png",
    title: str = "Best Test Metrics",
    value_fmt: str = ".3f",
    bar_width: float = 0.14,
    metric_colors: Optional[Dict[str, str]] = None,  # optional override: {"IoU": "#1f77b4", ...}
) -> str:
    """
    Plot a flexible bar chart of best test metrics for one or more models.
    - Colors are consistent per METRIC across all models.
    - Legend shows one entry per metric with matching color.

    Parameters
    ----------
    models : list of dict
        Each item must be {"name": str, "metrics": {metric_name: value, ...}}
    output_path : str
        Where to save the PNG.
    title : str
        Figure title.
    value_fmt : str
        Format for value labels on bars (e.g., ".3f").
    bar_width : float
        Width of each bar within a model group.
    metric_colors : dict or None
        Optional mapping metric name -> color hex/string for full control.
    """
    import os
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    assert isinstance(models, list) and len(models) > 0, "Provide at least one model dict."

    # Ensure folder exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Collect all metric names in order of first appearance (preserve order)
    all_metrics = []
    for model in models:
        for m_name in model.get("metrics", {}).keys():
            if m_name not in all_metrics:
                all_metrics.append(m_name)

    # Build a stable color mapping: metric -> color
    default_cycle = plt.rcParams.get("axes.prop_cycle", None)
    cycle_colors = (default_cycle.by_key().get("color", [])
                    if default_cycle is not None else [])
    if not cycle_colors:
        # Reasonable fallback palette
        cycle_colors = [f"C{i}" for i in range(10)]

    color_map = {}
    for i, m in enumerate(all_metrics):
        if metric_colors and m in metric_colors:
            color_map[m] = metric_colors[m]
        else:
            color_map[m] = cycle_colors[i % len(cycle_colors)]

    # Start plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group centers: 0..N-1
    group_centers = list(range(len(models)))
    xtick_positions, xtick_labels = [], []

    for g_idx, model in enumerate(models):
        name = model.get("name", f"Model {g_idx+1}")
        metrics = model.get("metrics", {})
        if not isinstance(metrics, dict) or len(metrics) == 0:
            continue

        metric_items = list(metrics.items())  # [(metric_name, value), ...]
        num_bars = len(metric_items)

        # Center bars for this model group
        total_width = num_bars * bar_width
        start_x = group_centers[g_idx] - (total_width / 2.0) + (bar_width / 2.0)

        for i, (m_name, m_val) in enumerate(metric_items):
            x = start_x + i * bar_width
            bar = ax.bar(
                x, m_val, width=bar_width,
                color=color_map.get(m_name, cycle_colors[0])
            )
            # Add numeric labels
            ax.bar_label(bar, labels=[format(m_val, value_fmt)])

        xtick_positions.append(group_centers[g_idx])
        xtick_labels.append(name)

    # Legend: one handle per metric with the mapped color
    handles = [Patch(facecolor=color_map[m], label=m) for m in all_metrics]
    ax.legend(handles=handles, title="Metric", ncol=3, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    ax.set_title(title)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels)
    ax.set_ylabel("Metric value")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path




# =========================
# Usage Examples
# =========================
if __name__ == "__main__":
    # Example 1: One figure with two subplots (loss + metrics), using separate train/test CSVs
    # (Replace with your actual paths)
    fp = plot_training_and_test_logs(
        train_csv="./log/train_metrics_bunet_joint_train_random_crop_separate_norm_manual.csv",
        test_csv="./log/test_metrics_bunet_joint_train_random_crop_sliding_window_separate_norm_manual.csv",
        output_dir="results_plots",
        desired_metrics = ["precision", "iou"],
        figure_prefix="bunet_joint_train_random_crop_separate_norm_manual"
    )
    print("Saved:", fp)

    # Example 2: Only train CSV (no test) — still produces the same figure with available curves
    # fp2 = plot_training_and_test_logs(
    #     train_csv="train_log.csv",
    #     test_csv=None,
    #     output_dir="plots"
    # )
    # print("Saved:", fp2)

    # Example 3: Flexible bar chart for best test metrics
    # You can pass one model:
    # out_bar = plot_best_metrics_bar(
    #     models=[{"name": "Swin Unet Transformer", "metrics": {"IoU": 0.44722, "Precision": 0.61493}}],
    #     output_path="plots/best_metrics_swin_unetr_airway_seg_ground_up.png",
    #     title="Best Test Metrics (Single Model)"
    # )
    # print("Saved:", out_bar)

    # Or multiple models with different metric sets:
    # out_bar2 = plot_best_metrics_bar(
    #     models=[
    #         {"name": "Swin Unetr Grd. Up", "metrics": {"IoU": 0.49031, "Precision": 0.66053}},
    #         {"name": "Swin Unetr Pre. Enc. Freeze", "metrics": {"IoU": 0.48847, "Precision": 0.69783}},
    #         {"name": "Swin Unetr Pre. Enc. Finetune", "metrics": {"IoU": 0.50249, "Precision": 0.68923}}
    #     ],
    #     output_path="plots/best_metrics_models.png",
    #     title="Best Test Metrics (Model-Grouped)"
    # )
    # print("Saved:", out_bar2)
