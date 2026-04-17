# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import ast
# import os
# from scipy.fftpack import fftn

# # ==========================================
# # 1. Helper Functions
# # ==========================================

# def get_scan_type(filename):
#     """
#     Categorizes the filename into UTE, TLC, FRC, or RV.
#     """
#     if "AnatCorrLungs" in filename:
#         return "UTE"
#     # elif "TLC" in filename:
#     #     return "TLC"
#     # elif "FRC" in filename:
#     #     return "FRC"
#     # elif "RV" in filename:
#     #     return "RV"
#     else:
#         return "Unknown"

# def parse_kernel_string(k_str):
#     """
#     Parses the string representation of the kernel list.
#     """
#     return eval(k_str)
#     # try:
#     #     # Handles standard list strings "[1, 2, ...]"
#     #     return ast.literal_eval(k_str)
#     # except:
#     #     # Fallback for simpler comma-separated strings
#     #     return [float(x) for x in k_str.strip('[]').split(',')]

# def compute_roughness(flat_weights, layer_idx):
#     """
#     Computes 'Roughness': Ratio of High Frequency Energy to Total Energy.
#     Roughness ~ 0.0: Smooth (Low Frequency / Bias Field)
#     Roughness ~ 1.0: Rough (High Frequency / Noise)
#     """
#     n_params = len(flat_weights)
    
#     # Determine input/output channels to find k (assuming GIN structure)
#     if layer_idx == 0: channels = 4 * 1
#     elif layer_idx == 3: channels = 1 * 4
#     else: channels = 4 * 4
        
#     vol = n_params / channels
#     # print(f"Vol: {vol}")
#     k = int(round(vol**(1/3)))
#     print(f"Kernel size K: {k}")
    
#     # 1x1x1 kernels are pure DC (Smooth)
#     if k < 2: return 0.0 
    
#     try:
#         # Reshape to generic 3D block for FFT
#         # We reshape to (channels, k, k, k)
#         w_reshaped = np.array(flat_weights).reshape(channels, k, k, k)
        
#         # Average over channels to get a single "spatial kernel" representative
#         w_spatial = np.mean(w_reshaped, axis=0) 
        
#         # Compute 3D FFT
#         fft_vals = np.abs(fftn(w_spatial))
        
#         # Calculate Energy Ratio
#         total_energy = np.sum(fft_vals)
#         low_freq_energy = fft_vals[0, 0, 0] # DC component (Zero Frequency)
        
#         # Ratio of non-DC energy to total energy
#         ratio_high = 1.0 - (low_freq_energy / (total_energy + 1e-9))
#         # print(ratio_high)
#         return ratio_high
        
#     except Exception as e:
#         raise RuntimeError(f"Error in compute_roughness: {e}")
        
#         # return 0.0

# # ==========================================
# # 2. Main Execution
# # ==========================================

# def main():
#     # Configuration
#     csv_path = './log/augmentation_weights_log_on_ute_with_roughness_enforced_saved_on_best_test.csv'
#     output_dir = './results_plots'
    
#     if not os.path.exists(csv_path):
#         print(f"Error: Log file not found at {csv_path}")
#         return

#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     print("Loading log data...")
#     df = pd.read_csv(csv_path)
    
#     plot_data = []

#     print("Computing roughness scores...")
#     counter = 0
#     # Iterate through each row in the log
#     for idx, row in df.iterrows():
#         fname = row['filename']
#         scan_type = get_scan_type(fname)
        
#         if scan_type == "Unknown":
#             continue

#         # Check all 4 layers
#         for layer_i in range(4):
#             col_name = f'kernel_{layer_i}'
#             if col_name not in row: continue
            
#             # Parse weights
#             raw_str = row[col_name]
#             # print(f"Raw String: {raw_str}")
#             weights = parse_kernel_string(raw_str)
            
#             # Compute Metric
#             roughness = compute_roughness(weights, layer_i)
#             print(f"Counter: {counter}")
#             print(f"Layer: {layer_i}")
#             print(f"Roughness: {roughness}")
#             counter += 1
            
#             # Append to data list
#             plot_data.append({
#                 'Scan Type': scan_type,
#                 'Layer': f'Layer {layer_i}',
#                 'Roughness': roughness
#             })
    
#     # print("len(plot_data):", len(plot_data))

#     # Convert to DataFrame
#     df_plot = pd.DataFrame(plot_data)
    
#     # Sort Scan Types for consistent plotting order
#     scan_order = ['UTE'] #['UTE'] #['UTE', 'TLC', 'FRC', 'RV']
    
#     # ==========================================
#     # 3. Plotting
#     # ==========================================
#     print("Generating Violin Plot...")
    
#     # Set plot style
#     sns.set_theme(style="whitegrid", context="talk")
    
#     plt.figure(figsize=(14, 8))
    
#     # Create Violin Plot
#     # split=False ensures we see full distribution for each category side-by-side
#     # inner="quartile" draws lines for median and quartiles inside the violin
#     sns.violinplot(
#         x='Layer', y='Roughness', hue='Scan Type', 
#         data=df_plot, hue_order=scan_order, 
#         palette="viridis", inner="quartile",
#         linewidth=1.2
#     )
    
#     plt.title("Distribution of Kernel Roughness per Layer", fontsize=20, fontweight='bold')
#     plt.ylabel("Roughness Score\n(0.0 = Smooth Bias, 1.0 = Noisy/Rough)", fontsize=16)
#     plt.xlabel("GIN Network Layer", fontsize=16)
#     plt.legend(title='Scan Type', loc='lower right', frameon=True)
    
#     # Set y-axis limits to logical range (with slight padding)
#     plt.ylim(-0.1, 1.1)
    
#     # Save Plot
#     save_path = os.path.join(output_dir, 'kernel_roughness_distribution_gin_on_ute_with_roughness_enforced_test_model.png')
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"Plot saved to {save_path}")

# if __name__ == "__main__":
#     main()


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import SimpleITK as sitk
import ast
import math
import itertools
from scipy.fftpack import fftn

# ==========================================
# 1. Replayable GIN & Helper Classes
# ==========================================

class ReplayableBlock(nn.Module):
    def __init__(self, out_channel=32, in_channel=3, scale_pool=[1, 3], use_act=True):
        super(ReplayableBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.use_act = use_act
        self.forced_kernel = None
        self.forced_k = None

    def set_weights(self, kernel_tensor, k_size):
        self.forced_kernel = kernel_tensor
        self.forced_k = k_size

    def forward(self, x_in):
        if self.forced_kernel is None:
            raise ValueError("Weights must be set before forward pass.")
            
        nb, nc, nx, ny, nz = x_in.shape
        k = self.forced_k
        ker = self.forced_kernel.to(x_in.device)
        shift = torch.zeros([self.out_channel * nb, 1, 1, 1], device=x_in.device)

        # Reshape for 3D Conv (Batched)
        x_in = x_in.reshape(1, nb * nc, nx, ny, nz)
        pad = math.ceil(k / 2) - 1  # works for k = 1, 2, 3
        # print(f"Kernel size: {k} and Padding: {pad}")
        
        if k == 2:
            # print("Before zero padding x_in shape: ", x_in.shape)
            x_in = nn.ZeroPad3d(padding=(0, 1, 0, 1, 0, 1))(x_in)
            # print("After zero padding x_in shape: ", x_in.shape)
        x_conv = F.conv3d(x_in, ker, stride=1, padding=pad, dilation=1, groups=nb)
        x_conv = x_conv + shift
        
        if self.use_act:
            x_conv = F.leaky_relu(x_conv)

        x_conv = x_conv.reshape(nb, self.out_channel, nx, ny, nz)
        return x_conv

class ReplayableGIN3D(nn.Module):
    def __init__(self, out_channel=1, in_channel=1, interm_channel=4, n_layer=4, out_norm='frob'):
        super(ReplayableGIN3D, self).__init__()
        self.out_channel = out_channel
        self.out_norm = out_norm
        self.layers = nn.ModuleList()
        # Layer 0
        self.layers.append(ReplayableBlock(out_channel=interm_channel, in_channel=in_channel))
        # Intermediate
        for ii in range(n_layer - 2):
            self.layers.append(ReplayableBlock(out_channel=interm_channel, in_channel=interm_channel))
        # Output
        self.layers.append(ReplayableBlock(out_channel=out_channel, in_channel=interm_channel, use_act=False))

    def set_gin_weights(self, kernel_list):
        for i, layer in enumerate(self.layers):
            layer.set_weights(kernel_list[i]['w'], kernel_list[i]['k'])

    def forward(self, x_in):
        nb, nc, nx, ny, nz = x_in.shape
        device = x_in.device
        
        # Use alpha = 0.5 for mixing
        alphas = torch.ones(nb, 1, 1, 1, 1, device=device) * 0.5
        
        x = self.layers[0](x_in)
        for blk in self.layers[1:]:
            x = blk(x)
        
        mixed = alphas * x + (1.0 - alphas) * x_in

        # Normalization
        if self.out_norm == 'frob':
            _in_frob = torch.norm(x_in.reshape(nb, nc, -1), dim=(-1, -2), p='fro', keepdim=False)
            _in_frob = _in_frob[:, None, None, None, None].repeat(1, nc, 1, 1, 1)
            
            _self_frob = torch.norm(mixed.reshape(nb, self.out_channel, -1), dim=(-1, -2), p='fro', keepdim=False)
            _self_frob = _self_frob[:, None, None, None, None].repeat(1, self.out_channel, 1, 1, 1)
            
            mixed = mixed * (1.0 / (_self_frob + 1e-5)) * _in_frob

        return mixed

# ==========================================
# 2. Kernel Library & Parsing
# ==========================================

def parse_kernel_string(k_str):
    try: return ast.literal_eval(k_str)
    except: return [float(x) for x in k_str.strip('[]').split(',')]

def reshape_kernel(flat_list, layer_idx):
    if layer_idx == 0: in_c, out_c = 1, 4
    elif layer_idx == 3: in_c, out_c = 4, 1
    else: in_c, out_c = 4, 4
    
    arr = np.array(flat_list, dtype=np.float32)
    # Calculate k
    vol = arr.shape[0] / (out_c * in_c)
    k = int(round(vol ** (1/3)))
    
    # Reshape to [Out, In, k, k, k]
    tensor = torch.from_numpy(arr.reshape(out_c, in_c, k, k, k))
    return tensor, k

def compute_roughness(tensor_w):
    w_spatial = tensor_w.mean(dim=(0, 1)).numpy()
    fft_vals = np.abs(fftn(w_spatial))
    total_energy = np.sum(fft_vals)
    low_freq_energy = fft_vals[0, 0, 0]
    return 1.0 - (low_freq_energy / (total_energy + 1e-9))

class KernelLibrary:
    def __init__(self, log_path):
        self.df = pd.read_csv(log_path)
        self.high_roughness = {0: [], 1: [], 2: [], 3: []} # >= 0.9
        self.low_roughness = {0: [], 1: [], 2: [], 3: []}  # <= 0.1
        self._populate()
        
    def _populate(self):
        print("Parsing Log for UTE kernels...")
        for idx, row in self.df.iterrows():
            if "AnatCorrLungs" not in row['filename']:
                continue
            
            for layer in range(4):
                if f'kernel_{layer}' not in row: continue
                
                flat = parse_kernel_string(row[f'kernel_{layer}'])
                tens, k = reshape_kernel(flat, layer)
                r_score = compute_roughness(tens)
                
                kernel_obj = {'k': k, 'w': tens, 'roughness': r_score}
                
                if r_score >= 0.9:
                    self.high_roughness[layer].append(kernel_obj)
                elif r_score <= 0.1:
                    self.low_roughness[layer].append(kernel_obj)

    def get_kernel(self, layer, roughness_type):
        """Returns a single kernel dict."""
        source = self.high_roughness if roughness_type == 'high' else self.low_roughness
        candidates = source[layer]
        
        if not candidates:
            # IMPORTANT: Handle missing kernels gracefully (e.g., if no 'low' roughness found for layer X)
            # We try to fallback to the other type just to keep script running, or error out.
            # Here we print a warning and try to use *any* kernel from that layer.
            print(f"Warning: No {roughness_type} kernel for L{layer}. Fallback to random.")
            # Fallback logic: check the other list
            fallback_source = self.low_roughness if roughness_type == 'high' else self.high_roughness
            if fallback_source[layer]:
                 return fallback_source[layer][np.random.randint(len(fallback_source[layer]))]
            return None # Total failure
            
        return candidates[np.random.randint(len(candidates))]

# ==========================================
# 3. Normalization & Saving
# ==========================================

def normalise_one_one(image):
    image = image.astype(np.float32)
    minimum = np.min(image)
    maximum = np.max(image)
    if maximum > minimum:
        ret = (image - minimum) / (maximum - minimum)
    else:
        ret = image * 0.
    ret = ret * 2. - 1.
    return ret

def save_nifti(tensor_data, original_shape, save_path):
    # tensor_data: (1, 1, H, W, D) -> because we removed permute
    arr = tensor_data.squeeze().cpu().numpy() # Becomes (H, W, D)
    
    # Crop padding
    oh, ow, od = original_shape
    arr_cropped = arr[0:oh, 0:ow, 0:od]
    
    # Save
    img_sitk = sitk.GetImageFromArray(arr_cropped)
    img_sitk.SetSpacing([1.25, 1.25, 1.25])
    sitk.WriteImage(img_sitk, save_path)
    print(f"Saved: {save_path}")

# ==========================================
# 4. Main
# ==========================================

def main():
    # --- Configuration ---
    npy_file = "/Shared/lss_segerard/parthghosh/data/UTE_new_data_numpy/103-041/20181011/AnatCorrLungs.npy" # <--- INPUT FILE
    log_path = "./log/augmentation_weights_log_on_ute_with_roughness_enforced_saved_on_best_test.csv"
    output_dir = "./generated_combinations"
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    # 1. Setup Kernel Library
    lib = KernelLibrary(log_path)
    
    # 2. Load & Process Image
    if not os.path.exists(npy_file):
        print(f"File not found: {npy_file}")
        return

    raw_arr = np.load(npy_file)
    img_arr = raw_arr[:, :, :, 0] # Extract image channel
    original_shape = img_arr.shape # (H, W, D)
    
    # Normalize
    img_norm = normalise_one_one(img_arr)
    
    # Pad (VariableSpatialFix logic)
    num_of_double_stride_conv = 4
    mul_factor = 2 ** num_of_double_stride_conv 
    h, w, d = img_norm.shape
    new_h = mul_factor * math.ceil(h/mul_factor)
    new_w = mul_factor * math.ceil(w/mul_factor)
    new_d = mul_factor * math.ceil(d/mul_factor)
    
    img_padded = np.pad(img_norm, 
                        ((0, new_h - h), (0, new_w - w), (0, new_d - d)), 
                        'constant', constant_values=-1)
    
    # Convert to Tensor WITHOUT permute
    # Shape becomes (1, 1, H, W, D)
    img_tensor = torch.from_numpy(img_padded).unsqueeze(0).unsqueeze(0).float()
    
    # Save Original
    save_nifti(img_tensor, original_shape, os.path.join(output_dir, "original.nii.gz"))
    
    # 3. Generate All Combinations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = img_tensor.to(device)
    
    augmentor = ReplayableGIN3D(out_channel=1, in_channel=1, interm_channel=4).to(device)
    augmentor.eval()
    
    # Define possibilities
    options = ['high', 'low']
    # itertools.product generates tuples like ('high', 'high', 'low', 'low')
    combinations = list(itertools.product(options, repeat=4))
    
    print(f"Generating {len(combinations)} augmented variations...")
    print(f"Combinations: {combinations}")
    
    with torch.no_grad():
        for combo in combinations:
            # combo is a tuple e.g. ('high', 'low', 'high', 'low')
            
            # Construct config for this combo
            config = []
            valid_combo = True
            
            for layer_idx, r_type in enumerate(combo):
                kernel = lib.get_kernel(layer_idx, r_type)
                if kernel is None:
                    valid_combo = False
                    break
                config.append(kernel)
            
            if not valid_combo:
                print(f"Skipping combo {combo} due to missing kernels.")
                continue
                
            # Set weights
            augmentor.set_gin_weights(config)
            
            # Forward Pass
            output = augmentor(img_tensor)
            
            # Construct Filename
            # e.g., aug_high_low_high_low.nii.gz
            fname_str = "_".join(combo)
            save_path = os.path.join(output_dir, f"aug_{fname_str}.nii.gz")
            
            save_nifti(output, original_shape, save_path)

if __name__ == "__main__":
    main()



# import torch
# from train_argon_ute_copd_join_with_gin_with_log_with_high_roughness import compute_roughness

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ker = torch.randn([1, 1, 3, 3, 3], requires_grad=False, device=device)

# ker_0 = ker - ker.mean()

# spatial_ker = ker_0.mean(dim=(0, 1))



# fft_vals = torch.abs(torch.fft.fftn(spatial_ker))

# s_ac = torch.sum(fft_vals)
# N = spatial_ker.numel()

# desired_roughness = 0.05

# c = (s_ac * (1-desired_roughness)) / (N * desired_roughness)

# print(c)

# final_ker = ker + c
        
# roughness = compute_roughness(final_ker)
# print(final_ker.shape)
# print(roughness)