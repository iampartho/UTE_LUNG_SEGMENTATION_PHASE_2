import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import itertools
from scipy.fftpack import fftn
from collections import Counter


def get_scan_type(filename):
    if "AnatCorrLungs" in filename:
        return "UTE"
    elif "TLC" in filename:
        return "TLC"
    elif "FRC" in filename:
        return "FRC"
    elif "RV" in filename:
        return "RV"
    else:
        return "Unknown"


def parse_kernel_string(k_str):
    return eval(k_str)


def compute_roughness(flat_weights, layer_idx):
    """
    Ratio of High Frequency Energy to Total Energy.
    ~0.0 = Smooth (Low Frequency), ~1.0 = Rough (High Frequency)
    """
    n_params = len(flat_weights)

    if layer_idx == 0:
        channels = 4 * 1
    elif layer_idx == 3:
        channels = 1 * 4
    else:
        channels = 4 * 4

    vol = n_params / channels
    k = int(round(vol ** (1 / 3)))

    if k < 2:
        return 0.0

    try:
        w_reshaped = np.array(flat_weights).reshape(channels, k, k, k)
        w_spatial = np.mean(w_reshaped, axis=0)
        fft_vals = np.abs(fftn(w_spatial))
        total_energy = np.sum(fft_vals)
        low_freq_energy = fft_vals[0, 0, 0]
        ratio_high = 1.0 - (low_freq_energy / (total_energy + 1e-9))
        return ratio_high
    except Exception as e:
        raise RuntimeError(f"Error in compute_roughness: {e}")


def classify_roughness(roughness_val, threshold=0.5):
    return "L" if roughness_val < threshold else "H"


def build_combo_label(combo_tuple):
    """(H, L, L, H) -> 'H-L-L-H'"""
    return "-".join(combo_tuple)


def main():
    # ---- Configuration ----
    csv_path = './log/augmentation_weights_log_causality_train_td1_roughness_enforced_2_normalised_gin_saved_on_best_test.csv'
    output_dir = './results_plots'
    roughness_threshold = 0.5

    if not os.path.exists(csv_path):
        print(f"Error: Log file not found at {csv_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading log data...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows.")

    all_combos = list(itertools.product(["L", "H"], repeat=4))
    all_combo_labels = [build_combo_label(c) for c in all_combos]

    scan_combo_data = {}

    print("Computing roughness for each scan...")
    for idx, row in df.iterrows():
        fname = row['filename']
        scan_type = get_scan_type(fname)

        if scan_type == "Unknown":
            continue

        roughness_classes = []
        valid = True
        for layer_i in range(4):
            col_name = f'kernel_{layer_i}'
            if col_name not in row or pd.isna(row[col_name]):
                valid = False
                break

            weights = parse_kernel_string(row[col_name])
            roughness = compute_roughness(weights, layer_i)
            roughness_classes.append(classify_roughness(roughness, roughness_threshold))

        if not valid or len(roughness_classes) != 4:
            continue

        combo_label = build_combo_label(tuple(roughness_classes))

        if scan_type not in scan_combo_data:
            scan_combo_data[scan_type] = []
        scan_combo_data[scan_type].append(combo_label)

    if not scan_combo_data:
        print("No valid scan data found. Check CSV path and filename patterns.")
        return

    scan_types_found = sorted(scan_combo_data.keys(),
                              key=lambda x: ["UTE", "TLC", "FRC", "RV"].index(x)
                              if x in ["UTE", "TLC", "FRC", "RV"] else 99)

    print(f"Scan types found: {scan_types_found}")
    for st in scan_types_found:
        print(f"  {st}: {len(scan_combo_data[st])} scans")

    # ---- Plotting ----
    n_types = len(scan_types_found)
    fig, axes = plt.subplots(n_types, 1, figsize=(18, 5 * n_types), squeeze=False)

    colors = {
        "UTE": "#2196F3",
        "TLC": "#4CAF50",
        "FRC": "#FF9800",
        "RV": "#E91E63",
    }

    for i, scan_type in enumerate(scan_types_found):
        ax = axes[i, 0]
        combo_list = scan_combo_data[scan_type]
        counts = Counter(combo_list)

        bar_heights = [counts.get(label, 0) for label in all_combo_labels]
        total = sum(bar_heights)

        color = colors.get(scan_type, "#9E9E9E")
        bars = ax.bar(range(len(all_combo_labels)), bar_heights,
                      color=color, edgecolor='black', linewidth=0.5, alpha=0.85)

        for bar_rect, h in zip(bars, bar_heights):
            if h > 0:
                pct = h / total * 100
                ax.text(bar_rect.get_x() + bar_rect.get_width() / 2, h + 0.3,
                        f'{h}\n({pct:.0f}%)', ha='center', va='bottom',
                        fontsize=7, fontweight='bold')

        ax.set_xticks(range(len(all_combo_labels)))
        ax.set_xticklabels(all_combo_labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Count', fontsize=13)
        ax.set_title(f'{scan_type}  (n={total})', fontsize=15, fontweight='bold')
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.set_xlim(-0.6, len(all_combo_labels) - 0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(
        'Distribution of Kernel Roughness Combinations (L < 0.5, H >= 0.5)\n'
        'Layer order: L0-L1-L2-L3',
        fontsize=17, fontweight='bold', y=1.01
    )
    fig.tight_layout()

    save_path = os.path.join(output_dir, 'kernel_roughness_combination_histogram_roughness_enforced_2_normalised_gin_train_with_ct_only_saved_on_best_test.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
