import os
import ast
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import exposure

# ==========================================
# Configuration
# ==========================================

CSV_PATH = './log/augmentation_weights_log_causality_train_td1_roughness_enforced.csv'
OUTPUT_DIR = './results_plots'
NUM_SAMPLES = 5

MRI_IMAGE_PATHS = [
    '/Shared/lss_segerard/parthghosh/data/UTE_new_data_numpy/103-033/20180412/AnatCorrLungs.npy',
    '/Shared/lss_segerard/parthghosh/data/UTE_new_data_numpy/103-028/20190819/AnatCorrLungs.npy',
    '/Shared/lss_segerard/parthghosh/data/UTE_new_data_numpy/103-028/20170725/AnatCorrLungs.npy',
    '/Shared/lss_segerard/parthghosh/data/UTE_new_data_numpy/103-002/20160119/AnatCorrLungs.npy',
    '/Shared/lss_segerard/parthghosh/data/UTE_new_data_numpy/103-027/20170821/AnatCorrLungs.npy',
]

# ==========================================
# Preprocessing (same as training pipeline)
# ==========================================

def normalise_hu(image, hu_range=[-1024.0, 200.0]):
    return np.clip(image, hu_range[0], hu_range[1]).astype(np.float32)

def normalise_zero_one(image):
    image = image.astype(np.float32)
    minimum = np.min(image)
    maximum = np.max(image)
    if maximum > minimum:
        return (image - minimum) / (maximum - minimum)
    return image * 0.

def normalise_one_one(image):
    ret = normalise_zero_one(image)
    return ret * 2. - 1.

def improve_contrast(image_array):
    """Improve the contrast of the image by rescaling intensity."""
    p1, p90 = np.percentile(image_array, (1, 95))
    image_array_rescaled = exposure.rescale_intensity(image_array, in_range=(p1, p90))
    return image_array_rescaled

def pad_to_multiple(image, num_of_double_stride_conv=4, padval=-1):
    mul = 2 ** num_of_double_stride_conv
    h, w, d = image.shape
    new_h = mul * math.ceil(h / mul)
    new_w = mul * math.ceil(w / mul)
    new_d = mul * math.ceil(d / mul)
    return np.pad(image, ((0, new_h - h), (0, new_w - w), (0, new_d - d)),
                  'constant', constant_values=padval)


# ==========================================
# Replayable GIN (from evaluate_gin_weights)
# ==========================================

class ReplayableBlock(nn.Module):
    def __init__(self, out_channel=32, in_channel=3, use_act=True):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.use_act = use_act
        self.forced_kernel = None
        self.forced_k = None

    def set_weights(self, kernel_tensor, k_size):
        self.forced_kernel = kernel_tensor
        self.forced_k = k_size

    def forward(self, x_in):
        nb, nc, nx, ny, nz = x_in.shape
        k = self.forced_k
        ker = self.forced_kernel.to(x_in.device)
        shift = torch.zeros([self.out_channel * nb, 1, 1, 1], device=x_in.device)

        x_in = x_in.view(1, nb * nc, nx, ny, nz)

        pad = math.ceil(k / 2) - 1
        if k == 2:
            x_in = nn.ZeroPad3d(padding=(0, 1, 0, 1, 0, 1))(x_in)

        x_conv = F.conv3d(x_in, ker, stride=1, padding=pad, dilation=1, groups=nb)
        x_conv = x_conv + shift

        if self.use_act:
            x_conv = F.leaky_relu(x_conv)

        x_conv = x_conv.view(nb, self.out_channel, nx, ny, nz)
        return x_conv


class ReplayableGIN3D(nn.Module):
    def __init__(self, out_channel=1, in_channel=1, interm_channel=4, n_layer=4, out_norm='frob'):
        super().__init__()
        self.out_channel = out_channel
        self.out_norm = out_norm
        self.layers = nn.ModuleList()
        self.layers.append(ReplayableBlock(out_channel=interm_channel, in_channel=in_channel))
        for _ in range(n_layer - 2):
            self.layers.append(ReplayableBlock(out_channel=interm_channel, in_channel=interm_channel))
        self.layers.append(ReplayableBlock(out_channel=out_channel, in_channel=interm_channel, use_act=False))

    def set_gin_weights(self, parsed_kernels):
        for i, layer in enumerate(self.layers):
            layer.set_weights(parsed_kernels[i]['w'], parsed_kernels[i]['k'])

    def forward(self, x_in):
        nb, nc, nx, ny, nz = x_in.shape
        device = x_in.device
        alphas = torch.ones(nb, 1, 1, 1, 1, device=device) * 0.5
        alphas = alphas.repeat(1, nc, 1, 1, 1)

        x = self.layers[0](x_in)
        for blk in self.layers[1:]:
            x = blk(x)

        mixed = alphas * x + (1.0 - alphas) * x_in

        if self.out_norm == 'frob':
            _in_frob = torch.norm(x_in.view(nb, nc, -1), dim=(-1, -2), p='fro', keepdim=False)
            _in_frob = _in_frob[:, None, None, None, None].repeat(1, nc, 1, 1, 1)
            _self_frob = torch.norm(mixed.view(nb, self.out_channel, -1), dim=(-1, -2), p='fro', keepdim=False)
            _self_frob = _self_frob[:, None, None, None, None].repeat(1, self.out_channel, 1, 1, 1)
            mixed = mixed * (1.0 / (_self_frob + 1e-5)) * _in_frob
        

        return mixed


# ==========================================
# CSV Parsing Helpers
# ==========================================

def parse_kernel_string(k_str):
    try:
        return ast.literal_eval(k_str)
    except:
        return [float(x) for x in k_str.strip('[]').split(',')]

def reshape_kernel(flat_list, layer_idx, batch_size=1):
    if layer_idx == 0:
        in_c, out_c = 1, 4
    elif layer_idx == 3:
        in_c, out_c = 4, 1
    else:
        in_c, out_c = 4, 4

    arr = np.array(flat_list, dtype=np.float32)
    vol = arr.shape[0] / (out_c * batch_size * in_c)
    k = int(round(vol ** (1 / 3)))
    target_shape = (out_c * batch_size, in_c, k, k, k)
    return torch.from_numpy(arr.reshape(target_shape)), k

def load_params_for_file(df, filename):
    row = df[df['filename'] == filename]
    if len(row) == 0:
        return None
    row = row.iloc[-1]
    params = []
    for i in range(4):
        w_col = f'kernel_{i}'
        k_col = f'kernel_size_{i}'
        if w_col not in row:
            break
        flat_w = parse_kernel_string(row[w_col])
        k_val_log = int(row[k_col])
        w_tensor, k_calc = reshape_kernel(flat_w, i)
        params.append({'k': k_calc, 'w': w_tensor})
    return params


# ==========================================
# Plotting
# ==========================================

def plot_histogram(ax, data, title, color, bins=200):
    flat = data.flatten()
    ax.hist(flat, bins=bins, color=color, alpha=0.8, edgecolor='none', density=True)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xlabel('Intensity', fontsize=8)
    ax.set_ylabel('Density', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    all_filenames = df['filename'].unique()
    print(f"Total unique files in log: {len(all_filenames)}")

    chosen = np.random.choice(all_filenames, size=min(NUM_SAMPLES, len(all_filenames)), replace=False)
    print("Randomly selected files:")
    for f in chosen:
        print(f"  {f}")

    augmentor = ReplayableGIN3D(out_channel=1, in_channel=1, interm_channel=4).to(device)
    augmentor.eval()

    fig, axes = plt.subplots(3, NUM_SAMPLES, figsize=(4 * NUM_SAMPLES, 10))
    if NUM_SAMPLES == 1:
        axes = axes.reshape(3, 1)

    # ---- Row 1 & 2: CT scans from CSV ----
    for col, fpath in enumerate(chosen):
        short_name = os.path.basename(fpath).replace('.npy', '')

        if not os.path.exists(fpath):
            print(f"  File not found: {fpath}, skipping columns {col}")
            for r in range(2):
                axes[r, col].text(0.5, 0.5, 'File not found', ha='center', va='center',
                                  transform=axes[r, col].transAxes, fontsize=9)
            continue

        arr = np.load(fpath)
        img_raw = arr[:, :, :, 0]

        # Row 1: clipped + normalized
        img_clipped = normalise_hu(img_raw)
        img_norm = normalise_one_one(img_clipped)
        plot_histogram(axes[0, col], img_norm,
                       f'CT Norm: {short_name}', color='#2196F3')

        # Row 2: after GIN transform
        img_padded = pad_to_multiple(img_norm)
        img_tensor = torch.from_numpy(img_padded).unsqueeze(0).unsqueeze(0).float().to(device)

        params = load_params_for_file(df, fpath)
        if params is None:
            axes[1, col].text(0.5, 0.5, 'No GIN params', ha='center', va='center',
                              transform=axes[1, col].transAxes, fontsize=9)
            continue

        augmentor.set_gin_weights(params)
        with torch.no_grad():
            gin_output = augmentor(img_tensor)

        gin_np = gin_output.squeeze().cpu().numpy()
        # gin_np = improve_contrast(gin_np)
        # gin_np = normalise_one_one(gin_np)
        h, w, d = img_norm.shape
        gin_np = gin_np[:h, :w, :d]
        plot_histogram(axes[1, col], gin_np,
                       f'After GIN: {short_name}', color='#E91E63')

    # ---- Row 3: MRI images ----
    for col in range(NUM_SAMPLES):
        if col >= len(MRI_IMAGE_PATHS):
            axes[2, col].text(0.5, 0.5, 'No MRI path', ha='center', va='center',
                              transform=axes[2, col].transAxes, fontsize=9)
            continue

        mri_path = MRI_IMAGE_PATHS[col]
        short_name = os.path.basename(mri_path).replace('.npy', '')

        if not os.path.exists(mri_path):
            axes[2, col].text(0.5, 0.5, 'File not found', ha='center', va='center',
                              transform=axes[2, col].transAxes, fontsize=9)
            continue

        mri_arr = np.load(mri_path)
        mri_img = mri_arr[:, :, :, 0]
        mri_norm = normalise_one_one(mri_img)
        plot_histogram(axes[2, col], mri_norm,
                       f'MRI: {short_name}', color='#4CAF50')

    # Row labels
    row_labels = ['CT (Clipped + Normalized)', 'CT After GIN Transform', 'MRI (Normalized)']
    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(f'{label}\nDensity', fontsize=10, fontweight='bold')

    fig.suptitle('Intensity Distributions: CT vs GIN-Augmented CT vs MRI',
                 fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'intensity_distributions_ct_gin_mri.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
