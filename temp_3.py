import os
import ast
import math
import numpy as np
import pandas as pd
from scipy.fftpack import fftn
import torch
import torch.nn.functional as F
import torch.nn as nn
import SimpleITK as sitk
import matplotlib.pyplot as plt

# ===================== CONFIGURATION =====================

NPY_PATH = '/Shared/lss_segerard/parthghosh/data/COPDgene_CT_1.25mm_numpy/13370C_TLC.npy'
CSV_PATH = './log/augmentation_weights_log_causality_train_td1_roughness_enforced.csv'
OUTPUT_DIR = './generated_combinations'
SPACING = (1.25, 1.25, 1.25)
ROUGHNESS_THRESHOLD = 0.7

DESIRED_COMBOS = [
    'H-H-H-H', 'H-H-H-L', 'H-H-L-H', 'H-H-L-L', 'H-L-H-H', 'H-L-H-L', 'H-L-L-H', 'H-L-L-L', 'L-H-H-H', 'L-H-H-L', 'L-H-L-H', 'L-H-L-L', 'L-L-H-H', 'L-L-H-L', 'L-L-L-H', 'L-L-L-L'
]

LAYER_SHAPES = [
    (4, 1),   # Layer 0: out=interm(4), in=1
    (4, 4),   # Layer 1: out=4,         in=4
    (4, 4),   # Layer 2: out=4,         in=4
    (1, 4),   # Layer 3: out=1,         in=4
]

# ===================== HELPERS =====================

def compute_roughness(weights_flat, kernel_size, out_ch, in_ch):
    if kernel_size < 2:
        return 0.0
    k = kernel_size
    w = np.array(weights_flat).reshape(out_ch, in_ch, k, k, k)
    w_spatial = np.mean(w, axis=(0, 1))
    fft_vals = np.abs(fftn(w_spatial))
    total_energy = np.sum(fft_vals)
    low_freq_energy = fft_vals[0, 0, 0]
    return 1.0 - (low_freq_energy / (total_energy + 1e-9))


def classify(roughness):
    return 'H' if roughness >= ROUGHNESS_THRESHOLD else 'L'


def row_combo(row):
    labels = []
    for i in range(4):
        k = int(row[f'kernel_size_{i}'])
        weights = ast.literal_eval(row[f'kernel_{i}'])
        out_ch, in_ch = LAYER_SHAPES[i]
        r = compute_roughness(weights, k, out_ch, in_ch)
        labels.append(classify(r))
    return '-'.join(labels)


def normalise_zero_one(image):
    mn, mx = image.min(), image.max()
    if mx > mn:
        return (image - mn) / (mx - mn)
    return image * 0.0


def normalise_one_one(image):
    return normalise_zero_one(image) * 2.0 - 1.0


def generate_nifty(arr_np, spacing, output_path):
    nifty = sitk.GetImageFromArray(arr_np)
    nifty.SetSpacing(spacing)
    sitk.WriteImage(nifty, output_path)


def save_mid_coronal_png(volume, output_path):
    mid = volume.shape[1] // 2
    sl = volume[:, mid, :]
    sl = np.rot90(sl.T, k=3)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(sl, cmap='gray', origin='lower')
    ax.axis('off')
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close(fig)


# ===================== GIN REPLAY =====================

def replay_gin_layer(x_in, ker, k, use_act=True):
    """Replay a single GradlessGCReplayNonlinBlock3D forward pass."""
    nb, nc, nx, ny, nz = x_in.shape
    out_ch = ker.shape[0] // nb

    shift = torch.randn([out_ch * nb, 1, 1, 1], device=x_in.device)

    x = x_in.reshape(1, nb * nc, nx, ny, nz)

    pad = math.ceil(k / 2) - 1
    if k == 2:
        x = nn.ZeroPad3d(padding=(0, 1, 0, 1, 0, 1))(x)

    x = F.conv3d(x, ker, stride=1, padding=pad, dilation=1, groups=nb)
    x = x + shift
    if use_act:
        x = F.leaky_relu(x)
    x = x.reshape(nb, out_ch, nx, ny, nz)
    return x


def replay_gin(x_in, layer_kernels, layer_ks):
    """
    Replay full GIN3D forward pass with specific kernels.
    layer_kernels: list of 4 torch tensors (the conv kernels per layer)
    layer_ks:      list of 4 ints (kernel sizes)
    """
    nb, nc, nx, ny, nz = x_in.shape
    out_channel = 1

    alpha = torch.rand(nb, 1, 1, 1, 1, device=x_in.device)
    alpha = alpha.repeat(1, nc, 1, 1, 1)

    x = x_in
    for i, (ker, k) in enumerate(zip(layer_kernels, layer_ks)):
        use_act = (i < len(layer_kernels) - 1)
        x = replay_gin_layer(x, ker, k, use_act=use_act)

    mixed = alpha * x + (1.0 - alpha) * x_in

    _in_frob = torch.norm(x_in.reshape(nb, nc, -1), dim=(-1, -2), p='fro', keepdim=False)
    _in_frob = _in_frob[:, None, None, None, None].repeat(1, nc, 1, 1, 1)
    _self_frob = torch.norm(mixed.reshape(nb, out_channel, -1), dim=(-1, -2), p='fro', keepdim=False)
    _self_frob = _self_frob[:, None, None, None, None].repeat(1, out_channel, 1, 1, 1)
    mixed = mixed * (1.0 / (_self_frob + 1e-5)) * _in_frob

    return mixed


def parse_row_kernels(row, device='cpu'):
    """Parse a CSV row into kernel tensors and kernel sizes."""
    kernels = []
    ks = []
    for i in range(4):
        k = int(row[f'kernel_size_{i}'])
        weights = ast.literal_eval(row[f'kernel_{i}'])
        out_ch, in_ch = LAYER_SHAPES[i]
        ker = torch.tensor(weights, dtype=torch.float32, device=device).reshape(out_ch, in_ch, k, k, k)
        kernels.append(ker)
        ks.append(k)
    return kernels, ks


# ===================== MAIN =====================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load image ---
    arr = np.load(NPY_PATH)
    img_np = arr[:, :, :, 0].astype(np.float32)
    img_np = normalise_one_one(img_np)

    basename = os.path.basename(NPY_PATH).replace('.npy', '')

    # --- Save raw ---
    raw_nifti_path = os.path.join(OUTPUT_DIR, f'{basename}_raw.nii.gz')
    raw_png_path = os.path.join(OUTPUT_DIR, f'{basename}_raw_midcoronal.png')
    generate_nifty(img_np, SPACING, raw_nifti_path)
    save_mid_coronal_png(img_np, raw_png_path)
    print(f'Saved raw: {raw_nifti_path}')

    # --- Parse CSV and index by combo ---
    df = pd.read_csv(CSV_PATH)
    print(f'Total CSV rows: {len(df)}')

    combo_to_rows = {}
    for _, row in df.iterrows():
        combo = row_combo(row)
        combo_to_rows.setdefault(combo, []).append(row)

    import random
    combo_to_row = {combo: random.choice(rows) for combo, rows in combo_to_rows.items()}

    print(f'Unique combos found in CSV: {sorted(combo_to_row.keys())}')

    # --- Prepare tensor ---
    img_tensor = torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0).to(device)

    # --- Generate augmented images ---
    for combo in DESIRED_COMBOS:
        if combo not in combo_to_row:
            print(f'WARNING: combo {combo} not found in CSV, skipping.')
            continue

        row = combo_to_row[combo]
        kernels, ks = parse_row_kernels(row, device=device)

        with torch.no_grad():
            aug = replay_gin(img_tensor, kernels, ks)
            aug_np = aug.squeeze().cpu().numpy()

        aug_np = normalise_one_one(aug_np)

        tag = combo.replace('-', '')
        nifti_path = os.path.join(OUTPUT_DIR, f'{basename}_GIN_{tag}.nii.gz')
        png_path = os.path.join(OUTPUT_DIR, f'{basename}_GIN_{tag}_midcoronal.png')

        generate_nifty(aug_np, SPACING, nifti_path)
        save_mid_coronal_png(aug_np, png_path)
        print(f'Saved {combo}: {nifti_path}')


if __name__ == '__main__':
    main()
