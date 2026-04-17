import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==========================================
# Part 1: Helper Functions (From adv_bias.py)
# ==========================================

def bspline_kernel_3d(sigma=[1, 1, 1], order=2, asTensor=False, dtype=torch.float32, device='cuda'):
    """
    Generate bspline 3D kernel matrix for interpolation.
    Optimized version for faster execution.
    """
    # Pre-calculate normalization factor (constant across iterations)
    norm_factor = sigma[0] * sigma[1] * sigma[2]
    
    # Convert sigma to torch tensor and move to target device immediately
    sigma_tensor = torch.tensor(sigma, dtype=dtype, device=device)
    
    # Create kernel_ones on target device from the start
    kernel_ones = torch.ones(1, 1, *sigma, dtype=dtype, device=device)
    kernel = kernel_ones.clone()
    
    # Pre-calculate padding values as torch tensor (avoid numpy conversion in loop)
    padding_tensor = sigma_tensor

    # Optimized loop: operations on target device, pre-calculated values
    for i in range(1, order + 1):
        # Calculate padding once per iteration (avoid list conversion)
        padding_val = (i * padding_tensor).int().tolist()
        kernel = F.conv3d(kernel, kernel_ones, padding=padding_val) / norm_factor

    if asTensor:
        return kernel[0, 0, ...]
    else:
        return kernel[0, 0, ...].cpu().numpy()

# ==========================================
# Part 2: GIN Implementation (From imagefilter3d.py)
# ==========================================

class GradlessGCReplayNonlinBlock3D(nn.Module):
    """
    Conv-leaky relu layer using group convolutions for efficiency.
   
    """
    def __init__(self, out_channel=32, in_channel=3, scale_pool=[1, 3], layer_id=0, use_act=True, requires_grad=False):
        super(GradlessGCReplayNonlinBlock3D, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.scale_pool = scale_pool
        self.use_act = use_act
        self.requires_grad = requires_grad

    def forward(self, x_in):
        # Random size of kernel from the pool
        idx_k = torch.randint(high=len(self.scale_pool), size=(1,))
        k = self.scale_pool[idx_k[0]]
        
        nb, nc, nx, ny, nz = x_in.shape
        device = x_in.device

        # Random weights generated on the fly (not learned)
        ker = torch.randn([self.out_channel * nb, self.in_channel, k, k, k], 
                          requires_grad=self.requires_grad, device=device)
        shift = torch.randn([self.out_channel * nb, 1, 1, 1], 
                            requires_grad=self.requires_grad, device=device)

        x_in = x_in.view(1, nb * nc, nx, ny, nz)
        # Group convolution allows independent processing per batch item
        x_conv = F.conv3d(x_in, ker, stride=1, padding=k // 2, dilation=1, groups=nb)
        x_conv = x_conv + shift
        
        if self.use_act:
            x_conv = F.leaky_relu(x_conv)

        x_conv = x_conv.view(nb, self.out_channel, nx, ny, nz)
        return x_conv

class GIN3D(nn.Module):
    """
    Global Intensity Non-linear Augmentation for 3D Images.
   
    """
    def __init__(self, out_channel=1, in_channel=1, interm_channel=2, scale_pool=[1, 3], n_layer=4, out_norm='frob'):
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
        # Output layer (no activation)
        self.layers.append(
            GradlessGCReplayNonlinBlock3D(out_channel=out_channel, in_channel=interm_channel, scale_pool=scale_pool, use_act=False)
        )

    def forward(self, x_in):
        nb, nc, nx, ny, nz = x_in.shape
        device = x_in.device

        # Random mixing coefficient alpha ~ U(0,1)
        alphas = torch.rand(nb, 1, 1, 1, 1, device=device)
        alphas = alphas.repeat(1, nc, 1, 1, 1)

        x = self.layers[0](x_in)
        for blk in self.layers[1:]:
            x = blk(x)
        
        # Mix original image with random network output (Eq 4 in paper)
        mixed = alphas * x + (1.0 - alphas) * x_in

        # Re-normalize to match original Frobenius norm
        if self.out_norm == 'frob':
            _in_frob = torch.norm(x_in.view(nb, nc, -1), dim=(-1, -2), p='fro', keepdim=False)
            _in_frob = _in_frob[:, None, None, None, None].repeat(1, nc, 1, 1, 1)
            
            _self_frob = torch.norm(mixed.view(nb, self.out_channel, -1), dim=(-1, -2), p='fro', keepdim=False)
            _self_frob = _self_frob[:, None, None, None, None].repeat(1, self.out_channel, 1, 1, 1)
            
            mixed = mixed * (1.0 / (_self_frob + 1e-5)) * _in_frob

        return mixed

# ==========================================
# Part 3: Causality (IPA) Wrapper
# ==========================================

class CausalityAugmentation3D(nn.Module):
    """
    Implements Interventional Pseudo-Correlation Augmentation (IPA) + GIN.
    This module takes an input batch and returns two augmented views suitable 
    for the consistency loss described in the paper.
    
   
    """
    def __init__(self, in_channels=1, control_point_spacing=[32, 32, 32], interpolation_order=3):
        super(CausalityAugmentation3D, self).__init__()
        # Initialize GIN generator
        # Note: in_channel and out_channel should match your data (usually 1 for CT)
        self.gin = GIN3D(out_channel=in_channels, in_channel=in_channels, interm_channel=4)
        
        self.spacing = np.array(control_point_spacing)
        self.order = interpolation_order

    def generate_pseudo_correlation_map(self, x_shape, device):
        """
        Generates the map 'b' using B-spline interpolation from random control points.
        Adapted from AdvBias logic to produce a [0,1] mask.
        """
        N, C, D, H, W = x_shape
        image_size = np.array([D, H, W])
        
        # Calculate grid size
        cp_grid = np.ceil(image_size / self.spacing).astype(int) + 2
        
        # 1. Initialize random control points
        # Shape: [N, 1, grid_d, grid_h, grid_w]
        control_points = torch.rand(N, 1, *cp_grid, device=device) 
        
        # 2. Get B-Spline kernel
        kernel = bspline_kernel_3d(sigma=self.spacing.tolist(), order=self.order, asTensor=True, device=device)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        
        # 3. Interpolate (Deconvolution)
        # Create a padding calc based on kernel size
        padding = ((np.array(kernel.shape[2:]) - 1) / 2).astype(int).tolist()
        stride = self.spacing.tolist()
        
        map_3d = F.conv_transpose3d(control_points, kernel, padding=padding, stride=stride, groups=1)
        
        # 4. Crop to image size
        # We perform center cropping to match D, H, W
        curr_d, curr_h, curr_w = map_3d.shape[2:]
        d_crop = (curr_d - D) // 2
        h_crop = (curr_h - H) // 2
        w_crop = (curr_w - W) // 2
        
        map_3d = map_3d[:, :, d_crop:d_crop+D, h_crop:h_crop+H, w_crop:w_crop+W]
        
        # 5. Normalize to [0, 1] to use as a mixing mask
        # Sigmoid is a smooth way to ensure [0,1] range for the mask
        map_3d = torch.sigmoid(map_3d) 
        
        return map_3d

    def forward(self, x):
        """
        Args:
            x: Input tensor (N, C, D, H, W)
        Returns:
            aug_1: First augmented view
            aug_2: Second augmented view (counterfactual)
        """
        # 1. Generate two independent GIN augmentations
        # Per paper: "sample intensity/texture transformations g_theta1, g_theta2 using GIN"

        # print("\n\n\nThis is inside the augmentatior forward line 1\n\n\n")
        gin_1 = self.gin(x)
        # gin_2 = self.gin(x)

        # print("\n\n\nThis is after gin, still in forward 2\n\n\n")
        
        # 2. Generate Pseudo-correlation map 'b'
        # Per paper: "Compute a pseudo-correlation map b"
        # b = self.generate_pseudo_correlation_map(x.shape, x.device)

        # print("\n\n\nThis is after generating bias field map, still in forward 3\n\n\n")
        
        # 3. Spatially Variable Blending (IPA)
        # Eq 6: T1 = g1 * b + g2 * (1-b)
        # aug_1 = gin_1 * b + gin_2 * (1 - b)
        
        # The paper implies the second view swaps the blending to separate correlations
        # aug_2 = gin_1 * (1 - b) + gin_2 * b
        
        return gin_1 #, gin_2 #aug_1, aug_2

# ==========================================
# Part 4: Training Loop Integration Example
# ==========================================

def causality_loss(pred1, pred2, lambda_div=10.0):
    """
    Computes the KL divergence consistency loss between predictions of the two augmented views.
    Equation 3 in the paper.
    """
    # Softmax probabilities
    prob1 = F.softmax(pred1, dim=1)
    prob2 = F.softmax(pred2, dim=1)
    
    # KL Divergence: D(p1 || p2)
    # Note: Pytorch's kl_div expects log_prob as input and target prob as target
    log_prob1 = F.log_softmax(pred1, dim=1)
    
    # "batchmean" averages over the batch dimension
    kl_loss = F.kl_div(log_prob1, prob2, reduction='batchmean')
    
    return lambda_div * kl_loss