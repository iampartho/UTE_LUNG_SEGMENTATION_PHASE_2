"""
Compare Local Complexity (LC) across multiple models.

For every LC column shared across the provided models (the total LC and
each per-layer column), this script writes TWO figures:

    <col>_train.png : train-only curves, one per model (legend = model)
    <col>_test.png  : test-only curves,  one per model (legend = model)

X-axis = opt_step on a log scale.
Y-axis = LC value for the given column.

Configure the models in MODELS below: a dict mapping a display label to a
directory that contains `train_lc.csv` and `test_lc.csv` (the same files
produced by causality_train.py).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================ Configuration ============================
# label -> directory containing train_lc.csv and test_lc.csv
MODELS = {
    '70_RE_80_GIN_instance_norm':      './log/local_complexity_monitoring_during_training_70_RE_80_GIN_instance_norm',
    '70_RE_80_GIN_instance_norm_ACNN':   './log/local_complexity_monitoring_during_training_70_RE_80_GIN_instance_norm_ACNN',
    # '70_RE_80_GIN':    './log/local_complexity_monitoring_during_training_70_RE_80_GIN_instance_norm_unfinished',
    # '70_RE_80_GIN_groupnorm':    './log/local_complexity_monitoring_during_training_70_RE_80_GIN_groupnorm',
}

OUTPUT_DIR = './log/local_complexity_monitoring_during_training_70_RE_80_GIN_instance_norm/plots_compare_models'

X_COL = 'opt_step'
SKIP_COLS = ('opt_step', 'epoch')

FIG_SIZE = (8, 5)
DPI = 150

# Optional fixed colors per model. If a label isn't here we fall back to
# matplotlib's default cycler.
MODEL_COLORS = {}

TRAIN_FILENAME = 'train_lc.csv'
TEST_FILENAME = 'test_lc.csv'


# ============================ Helpers ============================
def _humanize(col):
    """Turn 'avg_lc_enc_0_0' into 'Encoder layer 0_0', 'avg_total_lc' into
    'Total LC'."""
    if col == 'avg_total_lc':
        return 'Total LC'
    if col.startswith('avg_lc_enc_'):
        return f"Encoder layer {col[len('avg_lc_enc_'):]}"
    if col.startswith('avg_lc_dec_'):
        return f"Decoder layer {col[len('avg_lc_dec_'):]}"
    return col


def _load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing CSV: {path}")
    df = pd.read_csv(path).sort_values(X_COL).reset_index(drop=True)
    return df


def load_models(models):
    """Return {label: {'train': df, 'test': df}} for each model."""
    out = {}
    for label, lc_dir in models.items():
        train_path = os.path.join(lc_dir, TRAIN_FILENAME)
        test_path = os.path.join(lc_dir, TEST_FILENAME)
        out[label] = {
            'train': _load_csv(train_path),
            'test': _load_csv(test_path),
        }
    return out


def _shared_lc_columns(model_data):
    """Intersect LC columns across all models (and across train/test for each)."""
    col_sets = []
    for split_dfs in model_data.values():
        for df in split_dfs.values():
            col_sets.append({c for c in df.columns if c not in SKIP_COLS})
    if not col_sets:
        return []
    common = set.intersection(*col_sets)
    if not common:
        return []
    cols = sorted(common)
    if 'avg_total_lc' in cols:
        cols = ['avg_total_lc'] + [c for c in cols if c != 'avg_total_lc']
    return cols


def plot_split(model_data, col, split, out_path):
    """Plot one figure containing `col` for the given `split` (train|test),
    drawing one curve per model with model labels in the legend."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for label, dfs in model_data.items():
        df = dfs[split]
        if col not in df.columns:
            continue
        color = MODEL_COLORS.get(label)
        ax.plot(df[X_COL], df[col], label=label, linewidth=1.6, color=color)

    ax.set_xscale('log')
    ax.set_xlabel('Optimization step (log scale)')
    ax.set_ylabel('Local Complexity')
    ax.set_title(f"{_humanize(col)}  ({split})")
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.legend(loc='best', title='Model')

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


# ============================ Main ============================
def main():
    if not MODELS:
        raise ValueError("MODELS is empty; add at least one entry.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_data = load_models(MODELS)
    cols = _shared_lc_columns(model_data)
    if not cols:
        raise ValueError("No LC columns shared across the provided models.")

    print(f"[plot] Comparing {len(MODELS)} model(s) across {len(cols)} LC columns -> {OUTPUT_DIR}")
    for col in cols:
        for split in ('train', 'test'):
            out_path = os.path.join(OUTPUT_DIR, f"{col}_{split}.png")
            plot_split(model_data, col, split, out_path)
            print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
