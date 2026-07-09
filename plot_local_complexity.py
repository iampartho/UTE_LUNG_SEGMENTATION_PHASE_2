"""
Plot Local Complexity (LC) curves vs. optimization steps.

Inputs:
    log/local_complexity_monitoring_during_training/train_lc.csv
    log/local_complexity_monitoring_during_training/test_lc.csv

Output:
    log/local_complexity_monitoring_during_training/plots/<col>.png
    where <col> is `avg_total_lc` and every per-layer column
    (avg_lc_enc_0_0, avg_lc_enc_0_1, ..., avg_lc_dec_2_1).

Each plot has:
    - X-axis: opt_step on a log scale
    - Y-axis: LC value for the given column
    - Two curves: train (blue) and test (orange) with legends
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================ Configuration ============================
LC_DIR = './log/local_complexity_monitoring_during_training_70_RE_80_GIN_groupnorm'
TRAIN_CSV = os.path.join(LC_DIR, 'train_lc.csv')
TEST_CSV = os.path.join(LC_DIR, 'test_lc.csv')
PLOT_DIR = os.path.join(LC_DIR, 'plots')

X_COL = 'opt_step'
SKIP_COLS = ('opt_step', 'epoch')

FIG_SIZE = (8, 5)
DPI = 150


def _humanize(col):
    """Turn 'avg_lc_enc_0_0' into 'Encoder 0_0', 'avg_total_lc' into 'Total LC'."""
    if col == 'avg_total_lc':
        return 'Total LC'
    if col.startswith('avg_lc_enc_'):
        return f"Encoder layer {col[len('avg_lc_enc_'):]}"
    if col.startswith('avg_lc_dec_'):
        return f"Decoder layer {col[len('avg_lc_dec_'):]}"
    return col


def plot_lc_column(train_df, test_df, col, out_path):
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(train_df[X_COL], train_df[col], label='train', color='tab:blue', linewidth=1.6)
    ax.plot(test_df[X_COL], test_df[col], label='test', color='tab:orange', linewidth=1.6)

    ax.set_xscale('log')
    ax.set_xlabel('Optimization step (log scale)')
    ax.set_ylabel('Local Complexity')
    ax.set_title(_humanize(col))
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.legend(loc='best')

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def main():
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(f"Missing train CSV: {TRAIN_CSV}")
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"Missing test CSV: {TEST_CSV}")

    os.makedirs(PLOT_DIR, exist_ok=True)

    train_df = pd.read_csv(TRAIN_CSV).sort_values(X_COL).reset_index(drop=True)
    test_df = pd.read_csv(TEST_CSV).sort_values(X_COL).reset_index(drop=True)

    common_cols = [c for c in train_df.columns
                   if c in test_df.columns and c not in SKIP_COLS]
    if not common_cols:
        raise ValueError("No LC columns shared between train and test CSVs.")

    # Make sure 'avg_total_lc' is plotted first if present.
    if 'avg_total_lc' in common_cols:
        common_cols = ['avg_total_lc'] + [c for c in common_cols if c != 'avg_total_lc']

    print(f"[plot] Plotting {len(common_cols)} LC columns to {PLOT_DIR}")
    for col in common_cols:
        out_path = os.path.join(PLOT_DIR, f"{col}.png")
        plot_lc_column(train_df, test_df, col, out_path)
        print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
