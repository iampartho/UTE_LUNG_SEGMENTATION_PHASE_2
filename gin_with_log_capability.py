#
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.fftpack import fftn
import math

# ==========================================
# Part 2: GIN Implementation (Modified)
# ==========================================

# class GradlessGCReplayNonlinBlock3D(nn.Module):
#     def __init__(self, out_channel=32, in_channel=3, scale_pool=[1, 3], layer_id=0, use_act=True, requires_grad=False):
#         super(GradlessGCReplayNonlinBlock3D, self).__init__()
#         self.in_channel = in_channel
#         self.out_channel = out_channel
#         self.scale_pool = scale_pool
#         self.use_act = use_act
#         self.requires_grad = requires_grad
        
#         # Placeholders for logging
#         self.last_k = None
#         self.last_ker = None

#     def forward(self, x_in):
#         idx_k = torch.randint(high=len(self.scale_pool), size=(1,))
#         k = self.scale_pool[idx_k[0]]
        
#         nb, nc, nx, ny, nz = x_in.shape
#         device = x_in.device

#         # print(f"The current layer has the following parameters:")
#         # print("Input channel: ", self.in_channel)
#         # print("Output channel: ", self.out_channel)
        

#         ker = torch.randn([self.out_channel * nb, self.in_channel, k, k, k], 
#                           requires_grad=self.requires_grad, device=device)
#         shift = torch.randn([self.out_channel * nb, 1, 1, 1], 
#                             requires_grad=self.requires_grad, device=device)

#         # print(f"Kernel size: {ker.shape}")
#         # print(f"Shift size: {shift.shape}")

#         # --- Store parameters for logging ---
#         # We detach and move to CPU to save memory/compute when logging
#         self.last_k = k
#         self.last_ker = ker.detach().cpu()
#         # ------------------------------------

#         x_in = x_in.view(1, nb * nc, nx, ny, nz)
#         x_conv = F.conv3d(x_in, ker, stride=1, padding=k // 2, dilation=1, groups=nb)
#         x_conv = x_conv + shift
        
#         if self.use_act:
#             x_conv = F.leaky_relu(x_conv)

#         x_conv = x_conv.view(nb, self.out_channel, nx, ny, nz)
#         return x_conv



class GradlessGCReplayNonlinBlock3D(nn.Module):
    def __init__(self, out_channel=32, in_channel=3, scale_pool=[1, 2, 3], layer_id=0, use_act=True, requires_grad=False):
        super(GradlessGCReplayNonlinBlock3D, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.scale_pool = scale_pool
        self.use_act = use_act
        self.requires_grad = requires_grad
        self.last_k = None
        self.last_ker = None
        

    # def _generate_tuned_kernel(self, shape, device, k):
    #     # 1. 50% chance to be High or Low Roughness
    #     is_high_roughness = torch.rand(1).item() > 0.5
        
    #     # 2. Pick a specific target roughness in the desired ranges
    #     if is_high_roughness:
    #         target_r = torch.empty(1).uniform_(0.8, 0.999).item() # High: [0.8, ~1.0]
    #     else:
    #         target_r = torch.empty(1).uniform_(0.001, 0.8).item() # Low: [~0.0, 0.8]

    #     # 3. Generate raw noise and zero-mean it (Roughness = 1.0)
    #     ker_raw = torch.randn(shape, requires_grad=self.requires_grad, device=device)
    #     ker_zero = ker_raw - ker_raw.mean()
        
    #     # 4. Calculate Spatial FFT
    #     # Spatial kernel is the mean over Batch*Out and In dimensions
    #     w_spatial = ker_zero.mean(dim=(0, 1)) # Shape: [k, k, k]
        
    #     # PyTorch N-Dimensional FFT
    #     fft_vals = torch.abs(torch.fft.fftn(w_spatial))
        
    #     # S_ac is the sum of all magnitudes (DC is 0 because we zero-meaned it)
    #     S_ac = torch.sum(fft_vals).item()
    #     N = w_spatial.numel() # Total elements in spatial kernel (k^3)
        
    #     # 5. Calculate exact DC shift needed to hit target_r
    #     # Avoid division by zero
    #     target_r = max(min(target_r, 0.9999), 0.0001) 
    #     c = (S_ac * (1.0 - target_r)) / (target_r * N)
        
    #     # Randomize whether the bias makes the image lighter or darker
    #     if torch.rand(1).item() > 0.5:
    #         c = -c
            
    #     # 6. Apply the shift
    #     ker_final = ker_zero + c
        
    #     return ker_final

    def compute_roughness(self,tensor_w, kernel_size):
        """
        Computes 'Roughness' (High Frequency Ratio) for a weight tensor.
        tensor_w shape: [Batch*Out, In, k, k, k]
        """
        # Move to CPU numpy
        if tensor_w.is_cuda:
            w_np = tensor_w.detach().cpu().numpy()
        else:
            w_np = tensor_w.detach().numpy()

        if kernel_size < 2:
            return 0.0
            
        # Average over batch and channels to get a single representative 3D kernel
        # This captures the general frequency characteristic of the layer
        w_spatial = np.mean(w_np, axis=(0, 1))
        
        # Compute 3D FFT
        fft_vals = np.abs(fftn(w_spatial))
        
        total_energy = np.sum(fft_vals)
        low_freq_energy = fft_vals[0, 0, 0] # DC Component
        
        # Calculate High Frequency Ratio
        roughness = 1.0 - (low_freq_energy / (total_energy + 1e-9))
        return roughness

        
    def forward(self, x_in):
        idx_k = torch.randint(high=len(self.scale_pool), size=(1,))
        k = self.scale_pool[idx_k[0]]
        
        nb, nc, nx, ny, nz = x_in.shape
        device = x_in.device

        is_high_roughness = torch.rand(1).item() > 0.22 # 
        

        # Generate the tuned kernel
        shape = [self.out_channel * nb, self.in_channel, k, k, k]
        ker = torch.randn(shape, requires_grad=self.requires_grad, device=device) # #self._generate_tuned_kernel(shape, device, k)

        shift = torch.randn([self.out_channel * nb, 1, 1, 1], 
                            requires_grad=self.requires_grad, device=device)
        
        # roughness = self.compute_roughness(ker)
        # print(f"Roughness: {roughness}")

        # print(f"is high roughness: {is_high_roughness}")
        while is_high_roughness:
            if k < 2:
                k = self.scale_pool[torch.randint(high=len(self.scale_pool), low=1, size=(1,))[0]]
                shape = [self.out_channel * nb, self.in_channel, k, k, k]
                ker = torch.randn(shape, requires_grad=self.requires_grad, device=device) # #self._generate_tuned_kernel(shape, device, k)
            roughness = self.compute_roughness(ker, k)
            # print(f"Roughness: {roughness}")
            if roughness > 0.7:
                break
            else:
                ker = torch.randn(shape, requires_grad=self.requires_grad, device=device) # #self._generate_tuned_kernel(shape, device, k)
                ker -= ker.mean()
                # print("\nKer after mean subtraction\n")
        
        while not is_high_roughness:
            roughness = self.compute_roughness(ker, k)
            # print(f"Roughness: {roughness}")
            if roughness < 0.7:
                break
            else:
                k = 1
                shape = [self.out_channel * nb, self.in_channel, k, k, k]
                ker = torch.randn(shape, requires_grad=self.requires_grad, device=device) # #self._generate_tuned_kernel(shape, device, k)
                


        # roughness = self.compute_roughness(ker)
        # print(f"\n High roughness flag: {is_high_roughness} and Final Roughness: {roughness}")


        # Store for logging
        self.last_k = k
        self.last_ker = ker.detach().cpu()

        x_in = x_in.reshape(1, nb * nc, nx, ny, nz)
        
        
        pad = math.ceil(k / 2) - 1  # works for k = 1, 2, 3
        # print(f"Kernel size: {k} and Padding: {pad}")
        
        if k == 2:
            # nn.ZeroPad3d is not available in older torch builds; functional pad is equivalent.
            x_in = F.pad(x_in, pad=(0, 1, 0, 1, 0, 1), mode='constant', value=0.0)
        
        

        
        x_conv = F.conv3d(x_in, ker, stride=1, padding=pad, dilation=1, groups=nb)
        # print("x_conv shape: ", x_conv.shape)

        x_conv = x_conv + shift
        
        if self.use_act:
            x_conv = F.leaky_relu(x_conv)

        x_conv = x_conv.reshape(nb, self.out_channel, nx, ny, nz)
        return x_conv

class GIN3D(nn.Module):
    def __init__(self, out_channel=1, in_channel=1, interm_channel=4, scale_pool=[1, 2, 3], n_layer=4, out_norm='frob'):
        super(GIN3D, self).__init__()
        self.scale_pool = scale_pool
        self.out_channel = out_channel
        self.out_norm = out_norm
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(
            GradlessGCReplayNonlinBlock3D(out_channel=interm_channel, in_channel=in_channel, scale_pool=scale_pool)
        )
        # Intermediate layers
        for ii in range(n_layer - 2):
            self.layers.append(
                GradlessGCReplayNonlinBlock3D(out_channel=interm_channel, in_channel=interm_channel, scale_pool=scale_pool)
            )
        # Output layer
        self.layers.append(
            GradlessGCReplayNonlinBlock3D(out_channel=out_channel, in_channel=interm_channel, scale_pool=scale_pool, use_act=False)
        )

    def forward(self, x_in):
        nb, nc, nx, ny, nz = x_in.shape
        device = x_in.device
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

    def get_layer_params(self):
        """Retrieves (kernel_size, weights) from all layers."""
        params = []
        for idx, layer in enumerate(self.layers):
            params.append({
                'idx': idx,
                'k': layer.last_k,
                'w': layer.last_ker
            })
        return params

# ==========================================
# Part 3: Causality (IPA) Wrapper (Modified)
# ==========================================

class CausalityAugmentation3D(nn.Module):
    def __init__(self, in_channels=1, control_point_spacing=[32, 32, 32], interpolation_order=3):
        super(CausalityAugmentation3D, self).__init__()
        self.gin = GIN3D(out_channel=in_channels, in_channel=in_channels, interm_channel=4)
        self.spacing = np.array(control_point_spacing)
        self.order = interpolation_order


    # Normalisation functions
    def normalise_zero_one(self, image):
        minimum = torch.min(image).item()
        maximum = torch.max(image).item()
        if maximum > minimum:
            return (image - minimum) / (maximum - minimum)
        return image * 0.

    def normalise_one_one(self, image):
        ret = self.normalise_zero_one(image)
        return ret * 2. - 1.
    # End of normalisation functions

    # Forward pass
    def forward(self, x):
        gin_1 = self.gin(x)
        # gin_1 = self.normalise_one_one(gin_1)
        return gin_1

    def get_gin_params(self):
        """Helper to expose GIN parameters."""
        return self.gin.get_layer_params()