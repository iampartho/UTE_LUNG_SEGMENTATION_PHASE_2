import pandas as pd
import matplotlib.pyplot as plt
# # Load the results csv

# df = pd.read_csv("./test_result_csv/td1_roughness_enforced_5_normalised_gin_prev_data.csv")

# # Sort the csv by the iou column
# df = df.sort_values(by='iou', ascending=False)

# # select rows by jumping every 5 rows
# # df = df.iloc[:45]

# # get the filepaths and iou values
# filepaths = df["sid"].tolist()
# iou_values = df["iou"].tolist()

# # extract the index number from the filepaths
# index_numbers = [int(filepath.split('/')[-1].split('_')[-1]) for filepath in filepaths]

# # Get the original UTE filepaths from the original UTE csv
# ute_df = pd.read_csv("./ids/UTE_MRI_previous_numpy_without_clipping.csv")
# ute_filepaths = ute_df["filepaths"].tolist()

# # Based on the index numbers, get the corresponding UTE filepaths
# ute_filepaths = [ute_filepaths[index] for index in index_numbers]

# # Now based on the ute filepaths, get the corresponding total_lc from the log_local_complexity

# csv_paths = [f"./log_local_complexity/roughness_enforced_td1_app_5/roughness_enforced_model_UTE_MRI_previous_data/{'-'.join(scan_path.split('/')[1:])[:-4]}.csv" for scan_path in ute_filepaths]
# lc_dfs = [pd.read_csv(csv_path, comment='#', skip_blank_lines=True) for csv_path in csv_paths]

# layer_columns = [
#     "total_lc",
#     "enc_0_0", "enc_0_1", "enc_1_0", "enc_1_1",
#     "enc_2_0", "enc_2_1", "enc_3_0", "enc_3_1",
#     "dec_4_0", "dec_4_1", "dec_3_0", "dec_3_1",
#     "dec_2_0", "dec_2_1",
# ]

# n_cols = 3
# n_rows = (len(layer_columns) + n_cols - 1) // n_cols
# fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
# axes = axes.flatten()

# for i, col in enumerate(layer_columns):
#     lc_values = [lc_df[col].iloc[-1] for lc_df in lc_dfs]
#     axes[i].scatter(lc_values, iou_values, s=10)
#     axes[i].set_xlabel(f"{col} LC")
#     axes[i].set_ylabel("IOU")
#     axes[i].set_title(f"{col} LC vs IOU")

# for j in range(len(layer_columns), len(axes)):
#     axes[j].set_visible(False)

# fig.tight_layout()
# fig.savefig("./log_local_complexity/all_lc_vs_iou_for_all_scans_previous_data.png", dpi=150)
# plt.close(fig)
# print("Saved combined scatter plot")

# Histogram of LC values across ALL csv files in the folder
import glob
# all_csv_paths = sorted(glob.glob("./log_local_complexity/roughness_enforced_td1_app_5/roughness_enforced_model_UTE_MRI_previous_data/*.csv"))
# all_lc_dfs = [pd.read_csv(p, comment='#', skip_blank_lines=True) for p in all_csv_paths]

# layer_columns = [
#     "total_lc",
#     "enc_0_0", "enc_0_1", "enc_1_0", "enc_1_1",
#     "enc_2_0", "enc_2_1", "enc_3_0", "enc_3_1",
#     "dec_4_0", "dec_4_1", "dec_3_0", "dec_3_1",
#     "dec_2_0", "dec_2_1",
# ]

# n_cols = 3
# n_rows = (len(layer_columns) + n_cols - 1) // n_cols

# fig_hist, axes_hist = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
# axes_hist = axes_hist.flatten()

# for i, col in enumerate(layer_columns):
#     values = [lc_df[col].iloc[-1] for lc_df in all_lc_dfs]
#     axes_hist[i].hist(values, bins=20, edgecolor='black')
#     axes_hist[i].set_xlabel(f"{col} LC")
#     axes_hist[i].set_ylabel("Count")
#     axes_hist[i].set_title(f"Distribution of {col}")

# for j in range(len(layer_columns), len(axes_hist)):
#     axes_hist[j].set_visible(False)

# fig_hist.tight_layout()
# fig_hist.savefig("./log_local_complexity/all_lc_causality_paper_model_histograms_UTE_MRI_previous_data.png", dpi=150)
# plt.close(fig_hist)
# print("Saved combined histogram plot")

# Overlapping histograms: UTE vs CT with histogram intersection
import numpy as np

ute_csv_paths = sorted(glob.glob("./log_local_complexity/roughness_enforced_td1_app_5/roughness_enforced_model_UTE_MRI_previous_data/*.csv"))
ct_csv_paths = sorted(glob.glob("./log_local_complexity/roughness_enforced_td1_app_5/roughness_enforced_latest_model_COPD_CT_110/*.csv"))

ute_lc_dfs = [pd.read_csv(p, comment='#', skip_blank_lines=True) for p in ute_csv_paths]
ct_lc_dfs = [pd.read_csv(p, comment='#', skip_blank_lines=True) for p in ct_csv_paths]

layer_columns = [
    "total_lc",
    "enc_0_0", "enc_0_1", "enc_1_0", "enc_1_1",
    "enc_2_0", "enc_2_1", "enc_3_0", "enc_3_1",
    "dec_4_0", "dec_4_1", "dec_3_0", "dec_3_1",
    "dec_2_0", "dec_2_1",
]

n_cols = 3
n_rows = (len(layer_columns) + n_cols - 1) // n_cols

fig_overlap, axes_overlap = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes_overlap = axes_overlap.flatten()

n_bins = 30

for i, col in enumerate(layer_columns):
    ute_vals = np.array([lc_df[col].iloc[-1] for lc_df in ute_lc_dfs])
    ct_vals = np.array([lc_df[col].iloc[-1] for lc_df in ct_lc_dfs])

    bin_min = min(ute_vals.min(), ct_vals.min())
    bin_max = max(ute_vals.max(), ct_vals.max())
    bins = np.linspace(bin_min, bin_max, n_bins + 1)

    ute_hist, _ = np.histogram(ute_vals, bins=bins)
    ct_hist, _ = np.histogram(ct_vals, bins=bins)

    ute_norm = ute_hist / ute_hist.sum()
    ct_norm = ct_hist / ct_hist.sum()
    intersection = np.sum(np.minimum(ute_norm, ct_norm))

    eps = 1e-10
    ute_smooth = ute_norm + eps
    ct_smooth = ct_norm + eps
    kl_div = 0.5 * (np.sum(ute_smooth * np.log(ute_smooth / ct_smooth)) +
                     np.sum(ct_smooth * np.log(ct_smooth / ute_smooth)))

    axes_overlap[i].hist(ute_vals, bins=bins, alpha=0.5, label='UTE', edgecolor='black')
    axes_overlap[i].hist(ct_vals, bins=bins, alpha=0.5, label='CT', edgecolor='black')
    axes_overlap[i].set_xlabel(f"{col} LC")
    axes_overlap[i].set_ylabel("Count")
    axes_overlap[i].set_title(f"{col}")
    axes_overlap[i].tick_params(axis='x', rotation=45)
    axes_overlap[i].legend(fontsize=8, loc='upper right')
    axes_overlap[i].text(
        0.95, 0.55, f"Intersect: {intersection:.3f}\nKL Div: {kl_div:.3f}",
        transform=axes_overlap[i].transAxes,
        fontsize=8, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )

for j in range(len(layer_columns), len(axes_overlap)):
    axes_overlap[j].set_visible(False)

fig_overlap.suptitle("UTE vs CT Local Complexity Distribution Overlap", fontsize=14, y=1.01)
fig_overlap.tight_layout()
fig_overlap.savefig("./log_local_complexity/previous_ute_vs_ct_lc_histogram_overlap.png", dpi=150, bbox_inches='tight')
plt.close(fig_overlap)
print("Saved UTE vs CT overlapping histogram plot")