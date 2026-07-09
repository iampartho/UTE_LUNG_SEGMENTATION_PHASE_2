import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pandas as pd
import os

def generate_data_for_box_plot(df_list):
    def extract_col_values(df):
        df.drop(columns=["sid", "fold", "sdsd", "hd"], inplace=True)  # Dropping the "batch" column
        iou, dice, assd = None, None, None

        for i, col in enumerate(df.columns):
            if i == 0:
                iou = df[col].values
            elif i == 1:
                dice = df[col].values
            elif i == 2:
                assd = df[col].values
            
        return iou, dice, assd

    IOU, DICE, ASSD = [], [], []
    
    for each_df in df_list:
        iou, dice, assd = extract_col_values(each_df)
        IOU.append(iou)
        DICE.append(dice)
        ASSD.append(assd)
        
    return [IOU, DICE], [ASSD]


# Example data loading
df = pd.read_csv('./test_result_csv/unet_td1.csv')
df_1 = pd.read_csv('./test_result_csv/mri_to_ct_cyclegan_no_mri_leakage_crop_192_volmatched_on_basicunet_td1_marissa_data.csv')
# df_2 = pd.read_csv('./test_result_csv/energy_based_model_KP_data_OECLAD.csv')
# df_3 = pd.read_csv('./test_result_csv/td1_roughness_enforced_5_normalised_gin.csv')
# df_4 = pd.read_csv('./test_result_csv/bunet_disintangled_1.25mm.csv')

df_list = [df, df_1 ]#, df_2]#, df_3]#, df_4]

# Generate the data for box plots
data1, data2 = generate_data_for_box_plot(df_list)

# Create a figure with two subplots
plt.rcParams.update({'font.size': 16})
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Define colors and labels for box plots
box_colors = ['gold', 'black']#, 'green']#, 'gray']# 'green', 'gray']# 'gray']# 'green', 'black']
handles = []

# Common boxplot styling
medianprops = dict(color='red', linewidth=4.5)   # bold median
meanprops = dict(marker='o', markerfacecolor='magenta', markeredgecolor='black', markersize=10, alpha=0.9)  # bold mean
boxprops_template = lambda color: dict(facecolor='none', edgecolor=color, linewidth=4)  # transparent box

# Plotting IOU and DICE metrics
for i, group_data in enumerate(data1, start=1):
    positions = np.linspace(i - 0.2, i + 0.2, len(group_data))
    
    for j, d in enumerate(group_data, start=1):
        d = np.array(d)
        
        # Boxplot
        box = axes[0].boxplot(
            d, positions=[positions[j - 1]], widths=0.08, patch_artist=True,
            boxprops=boxprops_template(box_colors[j - 1]), whis=100,
            showmeans=True, meanprops=meanprops, medianprops=medianprops
        )
        
        # Scatter points
        scatter_positions = np.random.normal(positions[j - 1], 0.02, size=len(d))
        axes[0].scatter(scatter_positions, d, alpha=0.4, color=box_colors[j - 1], edgecolors='black')
        
        # Legend entries
        if i == 1:
            if j == 1:
                handles.append(Patch(facecolor='none', edgecolor=box_colors[j - 1], linewidth=2,
                                     label='Basic Unet (No Aug/No GIN) Trained on COPD-gene CT'))
            if j == 2:
                handles.append(Patch(facecolor='none', edgecolor=box_colors[j - 1], linewidth=2,
                                     label='CycleGAN + Same Unet'))
            # if j == 3:
            #     handles.append(Patch(facecolor='none', edgecolor=box_colors[j - 1], linewidth=2,
            #                          label='Test Time Computed Energy Model'))
            # if j == 4:
            #     handles.append(Patch(facecolor='none', edgecolor=box_colors[j - 1], linewidth=2,
                                    #  label='GIN with Roughness Enforced'))
            # if j == 5:
            #     handles.append(Patch(facecolor='none', edgecolor=box_colors[j - 1], linewidth=2,
            #                          label='Basic Unet(Disintangled) Resolution 1.25mm Isotropic'))

metrics = ["IOU\u2191", "DICE\u2191"]

axes[0].set_xticks(range(1, len(metrics) + 1))
axes[0].set_xticklabels(metrics)
axes[0].set_yticks(np.arange(0, 1.1, 0.1))
axes[0].grid(True)

# Add legend with extra mean & median info
extra_handles = [
    Line2D([0], [0], color='red', linewidth=4.5, label='Median'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='magenta', markeredgecolor='black', markersize=10, label='Mean')
]
axes[0].legend(handles=handles + extra_handles, loc='lower left')


# Plotting ASSD metric
for i, group_data in enumerate(data2, start=1):
    positions = np.linspace(i - 0.2, i + 0.2, len(group_data))
    
    for j, d in enumerate(group_data, start=1):
        d = np.array(d)
        
        box = axes[1].boxplot(
            d, positions=[positions[j - 1]], widths=0.08, patch_artist=True,
            boxprops=boxprops_template(box_colors[j - 1]), whis=100,
            showmeans=True, meanprops=meanprops, medianprops=medianprops
        )
        
        scatter_positions = np.random.normal(positions[j - 1], 0.02, size=len(d))
        axes[1].scatter(scatter_positions, d, alpha=0.4, color=box_colors[j - 1], edgecolors='black')

axes[1].set_xticks([1])
axes[1].set_xticklabels(["ASSD\u2193"])
axes[1].grid(True)

axes[1].legend(handles=handles + extra_handles, loc='upper right')

os.makedirs("./results_plots", exist_ok=True)
plt.suptitle('Results on Marissa Data Set')
plt.savefig('./results_plots/temp.png', bbox_inches='tight')
plt.cla()
