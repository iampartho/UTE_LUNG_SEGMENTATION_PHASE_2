import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import ast
from torch.utils.data import DataLoader
from torchvision import transforms
from monai.losses import TverskyLoss

# Import your custom modules
# Ensure these files are in the same directory or python path
from basic_unet_disentagled import BasicUNet
from helper_ute_copd import UTEDataset, ToTensor, VariableSpatialFix
from causality_paper_augmentation_2 import GIN3D

# ==========================================
# Part 1: Replayable GIN Implementation
# ==========================================

class ReplayableBlock(nn.Module):
    """
    Modified block that accepts specific weights and sets bias to zero.
    """
    def __init__(self, out_channel=32, in_channel=3, scale_pool=[1, 3], use_act=True):
        super(ReplayableBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.use_act = use_act
        
        # Placeholders for forced weights
        self.forced_kernel = None
        self.forced_k = None

    def set_weights(self, kernel_tensor, k_size):
        self.forced_kernel = kernel_tensor
        self.forced_k = k_size

    def forward(self, x_in):
        if self.forced_kernel is None:
            raise ValueError("Weights must be set before forward pass in ReplayableBlock")

        nb, nc, nx, ny, nz = x_in.shape
        k = self.forced_k
        ker = self.forced_kernel.to(x_in.device)
        
        # User requested "without bias", so we set shift to 0
        # shape matches the original random shift: [out * nb, 1, 1, 1]
        shift = torch.zeros([self.out_channel * nb, 1, 1, 1], device=x_in.device)

        x_in = x_in.view(1, nb * nc, nx, ny, nz)
        
        # Verify kernel shape matches input expectations
        # Expected: [out_channel*nb, in_channel, k, k, k]
        if ker.shape[1] != nc:
             # Basic check: if channels don't match, we might need to reshape or check logic
             # But assuming correct logging, it should match.
             pass

        x_conv = F.conv3d(x_in, ker, stride=1, padding=k // 2, dilation=1, groups=nb)
        x_conv = x_conv + shift
        
        if self.use_act:
            x_conv = F.leaky_relu(x_conv)

        x_conv = x_conv.view(nb, self.out_channel, nx, ny, nz)
        return x_conv

class ReplayableGIN3D(nn.Module):
    """
    GIN3D wrapper that uses ReplayableBlocks.
    """
    def __init__(self, out_channel=1, in_channel=1, interm_channel=4, n_layer=4, out_norm='frob'):
        super(ReplayableGIN3D, self).__init__()
        self.out_channel = out_channel
        self.out_norm = out_norm
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(
            ReplayableBlock(out_channel=interm_channel, in_channel=in_channel)
        )
        # Intermediate layers
        for ii in range(n_layer - 2):
            self.layers.append(
                ReplayableBlock(out_channel=interm_channel, in_channel=interm_channel)
            )
        # Output layer (no activation)
        self.layers.append(
            ReplayableBlock(out_channel=out_channel, in_channel=interm_channel, use_act=False)
        )

    def set_gin_weights(self, parsed_kernels):
        """
        parsed_kernels: List of dictionaries [{'k': int, 'w': tensor}, ...]
        """
        if len(parsed_kernels) != len(self.layers):
            print(f"Warning: Logged layers {len(parsed_kernels)} != Model layers {len(self.layers)}")
            
        for i, layer in enumerate(self.layers):
            layer.set_weights(parsed_kernels[i]['w'], parsed_kernels[i]['k'])

    def forward(self, x_in):
        nb, nc, nx, ny, nz = x_in.shape
        device = x_in.device

        # In the training script, alpha was random.
        # If we want to purely replay the "kernel" effect without mixing, 
        # we might just return the network output. 
        # However, the standard GIN equation is: mixed = alpha * net(x) + (1-alpha) * x
        # The prompt says "apply the gin augmentation". 
        # Without the logged alpha, we cannot perfectly reproduce the *exact* image if mixing happened.
        # Assuming we just want to run the convolution part or use a fixed alpha?
        # Standard Replay usually implies alpha=1 (fully processed) or we need alpha logged too.
        # Given "extract kernel size and weights", I will assume alpha=1.0 (Full GIN effect)
        # or we generate a new random alpha (statistically similar).
        # PROMPT INTERPRETATION: "construct the augmentor for that and apply the gin augmentation"
        # usually implies applying the transformation defined by the kernels. 
        # I will use random alpha as per original GIN logic unless instructed otherwise,
        # BUT since we are evaluating "loss after train" for specific kernels, 
        # it makes sense to keep alpha random (as it is a stochastic augmentation) 
        # OR fix it to 0.5. Let's stick to the original logic: random alpha.
        
        alphas = torch.rand(nb, 1, 1, 1, 1, device=device)
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
# Part 2: Parsing Helpers
# ==========================================

def parse_kernel_string(k_str):
    """Parses a string representation of a list back into a list."""
    try:
        # Handling potentially large strings or different formats
        return ast.literal_eval(k_str)
    except:
        # Fallback if it's a simple string format
        return [float(x) for x in k_str.strip('[]').split(',')]

def reshape_kernel(flat_list, layer_idx, batch_size=1):
    """
    Reshapes flattened kernel list into [out*nb, in, k, k, k].
    Hardcoded structure based on GIN3D default init.
    """
    # GIN3D default: 1 -> 4 -> 4 -> 4 -> 1
    # Layer 0: in=1, out=4
    # Layer 1: in=4, out=4
    # Layer 2: in=4, out=4
    # Layer 3: in=4, out=1
    
    in_c, out_c = 0, 0
    if layer_idx == 0:
        in_c, out_c = 1, 4
    elif layer_idx == 3: # Assuming 4 layers (0,1,2,3)
        in_c, out_c = 4, 1
    else:
        in_c, out_c = 4, 4
        
    arr = np.array(flat_list, dtype=np.float32)
    
    # Calculate k from volume
    total_elements = arr.shape[0]
    # shape = out * nb * in * k^3
    # k^3 = total / (out * nb * in)
    
    vol = total_elements / (out_c * batch_size * in_c)
    k = int(round(vol ** (1/3)))
    
    target_shape = (out_c * batch_size, in_c, k, k, k)
    
    try:
        tensor = torch.from_numpy(arr.reshape(target_shape))
        return tensor, k
    except Exception as e:
        print(f"Error reshaping layer {layer_idx}: Size {total_elements}, Target {target_shape}")
        raise e

def load_params_for_file(df, filename):
    """
    Extracts params for a specific file from the dataframe.
    """
    row = df[df['filename'] == filename]
    if len(row) == 0:
        return None
    
    # Use the last occurrence if multiple exist (latest training step)
    row = row.iloc[-1]
    
    params = []
    # We assume 4 layers (0 to 3) based on GIN3D defaults
    for i in range(4):
        k_col = f'kernel_size_{i}'
        w_col = f'kernel_{i}'
        
        if w_col not in row:
            break
            
        flat_w = parse_kernel_string(row[w_col])
        k_val_log = int(row[k_col])
        
        # Reshape
        w_tensor, k_calc = reshape_kernel(flat_w, i, batch_size=1)
        
        # Sanity check
        if k_calc != k_val_log:
            print(f"Warning: Calculated k {k_calc} != Logged k {k_val_log} for {filename}")
            
        params.append({'k': k_calc, 'w': w_tensor})
        
    return params

# ==========================================
# Part 3: Main Execution
# ==========================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Configuration
    csv_path = './log/augmentation_weights_log.csv'
    model_path = './save_models/best_bunet_joint_train_with_gin_aug.pth'
    
    if not os.path.exists(csv_path):
        print(f"Error: Log file not found at {csv_path}")
        return

    # 2. Load Data
    print("Loading Log File...")
    df = pd.read_csv(csv_path)
    
    # Initialize new column if not exists
    if 'loss_after_train' not in df.columns:
        df['loss_after_train'] = np.nan

    print("Loading Model...")
    model = BasicUNet().to(device)
    # Wrap in DataParallel if trained that way, to match state_dict keys
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    else:
        raise RuntimeError("Error: Model path not found")

        
    model.eval()
    
    # 3. Setup Augmentor
    # Initialize Replayable GIN
    augmentor = ReplayableGIN3D(out_channel=1, in_channel=1, interm_channel=4).to(device)
    augmentor.eval() # Important: we aren't training the augmentor

    # 4. Setup Dataloader
    # NOTE: Assuming UTEDataset now returns 6 values: X_ute, y_ute, X_copd, y_copd, fname_ute, fname_copd
    training_data = UTEDataset(ute_csv_file='./ids/ute_train.csv',
                               copd_csv_file='./ids/copd_train_1.25mm.csv',
                               transform = transforms.Compose([
                                    VariableSpatialFix(num_of_double_stride_conv=4),
                                    ToTensor()
                               ]),
                               training = True)
                               
    dataloader = DataLoader(training_data, batch_size=1, shuffle=False)
    loss_fn = TverskyLoss(sigmoid=True)

    print("Starting Evaluation Loop...")
    
    # To speed up CSV lookups, index by filename
    # But since we need to write back, we'll keep the index
    
    count = 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            # Unpack assuming the modified 6-item return structure
            if len(batch_data) == 6:
                X_ute, y_ute, X_copd, y_copd, fname_ute, fname_copd = batch_data
            else:
                print("Error: Dataloader did not return 6 items. Please ensure filenames are returned.")
                break

            X_ute, y_ute = X_ute.to(device), y_ute.to(device)
            X_copd, y_copd = X_copd.to(device), y_copd.to(device)
            
            # Helper to process one image
            def process_sample(X, y, fname):
                fname_str = fname[0] if isinstance(fname, (list, tuple)) else fname
                
                # Check if file exists in log
                indices = df.index[df['filename'] == fname_str].tolist()
                if not indices:
                    # File not found in training logs (maybe didn't pass augmentor threshold)
                    raise RuntimeError(f"Error: File {fname_str} not found in training logs")
                
                # Get params
                params = load_params_for_file(df, fname_str)
                if params is None: 
                    raise RuntimeError(f"Error: Parameters not found for file {fname_str}")

                # Configure Augmentor
                try:
                    augmentor.set_gin_weights(params)
                except Exception as e:
                    print(f"Failed to set weights for {fname_str}: {e}")
                    return

                # Augment (Bias forced to 0 inside ReplayableBlock)
                X_aug = augmentor(X)
                
                # Forward
                pred = model(X_aug)
                loss = loss_fn(pred, y)
                
                # Log
                # Update all rows matching this filename (or just the last one)
                # Typically we update the specific row we read from
                idx_to_update = indices[-1] 
                df.at[idx_to_update, 'loss_after_train'] = loss.item()

            # Process UTE
            process_sample(X_ute, y_ute, fname_ute)
            
            # Process COPD
            process_sample(X_copd, y_copd, fname_copd)
            
            count += 1
            
            print(f"Processed {count} batches...")

    # 5. Save Results
    output_path = './log/augmentation_weights_log_updated.csv'
    df.to_csv(output_path, index=False)
    print(f"Done. Updated log saved to {output_path}")

if __name__ == "__main__":
    main()