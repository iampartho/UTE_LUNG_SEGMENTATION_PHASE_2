# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# class GIN3D(nn.Module):
#     """
#     Global Intensity Non-linear Augmentation (GIN) - 3D Version
    
#     Reference: Section III-B "Global intensity non-linear augmentation"
#     Goal: Transforms image appearances (intensities/textures) using randomly-weighted 
#     shallow networks to force the model to learn shape (domain-invariant) instead of texture.
#     """
#     def __init__(self, in_channels=1, intermediate_channels=2, num_layers=4):
#         super().__init__()
#         self.num_layers = num_layers
        
#         # Reference: Section III-B (Design-of-choices)
#         # "Transformations... are instantiated as shallow multi-layer convolutional networks"
#         self.net = nn.Sequential()
#         for i in range(num_layers):
#             in_c = in_channels if i == 0 else intermediate_channels
#             out_c = intermediate_channels if i < num_layers - 1 else in_channels
            
#             # Reference: Section III-B
#             # "Random convolutional kernels... with small receptive fields (to avoid over-blurring)"
#             # We use kernel_size=1 for 3D to mix intensities without distorting spatial structures.
#             self.net.add_module(f"conv_{i}", nn.Conv3d(in_c, out_c, kernel_size=1, padding=0, bias=False))
            
#             # Reference: Section III-B
#             # "Leaky ReLU non-linearities between two neighboring convolutional layers"
#             if i < num_layers - 1:
#                 self.net.add_module(f"act_{i}", nn.LeakyReLU(0.1))
        
#         self.randomize_weights()

#     def randomize_weights(self):
#         """
#         Reference: Section III-B
#         "Sampled from Gaussian distributions N(0, I)... At each iteration, new g_theta are sampled"
#         """
#         for m in self.net.modules():
#             if isinstance(m, nn.Conv3d):
#                 nn.init.normal_(m.weight, mean=0.0, std=1.0)

#     def forward(self, x):
#         # 1. Randomize weights for this iteration (Section III-B)
#         # self.randomize_weights()
        
#         # 2. Pass through random shallow network
#         out_net = self.net(x)
        
#         # 3. Sample alpha (mixing coefficient)
#         # Reference: Section III-B, Paragraph "Design-of-choices"
#         # "random interpolation coefficient sampled from uniform distribution U(0,1) as alpha"
#         alpha = torch.rand(1, device=x.device)
        
#         # 4. Linear Interpolation (Equation 4)
#         # g_theta(x) = alpha * Network(x) + (1 - alpha) * x
#         mixed = alpha * out_net + (1 - alpha) * x
        
#         # 5. Re-normalization (Equation 4 denominator)
#         # "Output image is re-normalized to have the same Frobenius norm as the original input x"
#         norm_x = torch.norm(x.view(x.shape[0], -1), p='fro', dim=1, keepdim=True)
#         norm_mixed = torch.norm(mixed.view(mixed.shape[0], -1), p='fro', dim=1, keepdim=True)
        
#         # Avoid division by zero
#         norm_mixed = norm_mixed + 1e-6
        
#         # Reshape for broadcasting
#         norm_x = norm_x.view(-1, 1, 1, 1, 1)
#         norm_mixed = norm_mixed.view(-1, 1, 1, 1, 1)

#         final_output = mixed * (norm_x / norm_mixed)
#         difference = x + final_output
        
#         return final_output, difference

        
        
#         # return mixed * (norm_x / norm_mixed)


# class IPA3D(nn.Module):
#     """
#     Interventional Pseudo-Correlation Augmentation (IPA) - 3D Version
    
#     Reference: Section III-C "Interventional pseudo-correlation augmentation"
#     Goal: Mitigate shifted-correlation effect by blending two GIN-augmented images 
#     using a spatially variable map (removing confounders).
#     """
#     def __init__(self, input_size=(192, 192, 192), control_point_spacing_ratio=0.25):
#         super().__init__()
#         self.input_size = input_size
#         # Reference: Section III-C, Paragraph "Pseudo-correlation maps"
#         # "Spacing between two neighboring control points is empirically set to be 1/4 of image length"
#         self.spacing_ratio = control_point_spacing_ratio

#     def generate_map(self, batch_size, device):
#         """
#         Reference: Section III-C (Fig. 4-B)
#         "Interpolating along a lattice of randomly-valued control points... using cubic B-spline"
        
#         Note: For 3D, we approximate B-spline using Trilinear interpolation of random noise 
#         to generate smooth, low-frequency 3D clouds.
#         """
#         d, h, w = [int(s * self.spacing_ratio) for s in self.input_size]
        
#         # Random control points [0, 1]
#         control_points = torch.rand((batch_size, 1, d, h, w), device=device)
        
#         # Interpolate to full size to create smooth "cloud-like" correlations
#         pseudo_map = F.interpolate(control_points, size=self.input_size, mode='trilinear', align_corners=False)
        
#         return pseudo_map

#     def forward(self, x_gin1, x_gin2):
#         """
#         Reference: Section III-C (Equation 6)
#         T(x) = g_theta1(x) * b + g_theta2(x) * (1 - b)
#         """
#         b_map = self.generate_map(x_gin1.shape[0], x_gin1.device)
        
#         # Reference: Section III-C, Paragraph "Spatially-variable blending"
#         # View 1: Uses map 'b'
#         aug_img_1 = x_gin1 * b_map + x_gin2 * (1 - b_map)
        
#         # View 2: Uses map '1-b' (swapping positions as mentioned in text below Eq. 6)
#         # "We simultaneously obtain one additional augmented image T2(x) by swapping the positions of b and 1-b"
#         aug_img_2 = x_gin1 * (1 - b_map) + x_gin2 * b_map
        
#         return aug_img_1, aug_img_2


# 3D version of GIN in case you are using a 3D network. 3D ver. of IPA will be released soon
# import torch
# from torch import nn
# from torch.nn import functional as F
# import numpy as np


# class GradlessGCReplayNonlinBlock3D(nn.Module):
#     def __init__(self, out_channel = 32, in_channel = 1, scale_pool = [1, 3], layer_id = 0, use_act = True, requires_grad = False, init_scale = 'default', **kwargs):
#         """
#         Conv-leaky relu layer. Efficient implementation by using group convolutions
#         """
#         super(GradlessGCReplayNonlinBlock3D, self).__init__()
#         self.in_channel     = in_channel
#         self.out_channel    = out_channel
#         self.scale_pool     = scale_pool
#         self.layer_id       = layer_id
#         self.use_act        = use_act
#         self.requires_grad  = requires_grad
#         self.init_scale     = init_scale
#         assert requires_grad == False

#     def forward(self, x_in, requires_grad = False):
#         # random size of kernel
#         idx_k = torch.randint(high = len(self.scale_pool), size = (1,))
#         k = self.scale_pool[idx_k[0]]

#         nb, nc, nx, ny, nz = x_in.shape

#         ker = torch.randn([self.out_channel * nb, self.in_channel , k, k, k  ], requires_grad = self.requires_grad  ).cuda()
#         shift = torch.randn( [self.out_channel * nb, 1, 1, 1 ], requires_grad = self.requires_grad  ).cuda() * 1.0

#         x_in = x_in.view(1, nb * nc, nx, ny, nz)
#         x_conv = F.conv3d(x_in, ker, stride =1, padding = k // 2, dilation = 1, groups = nb )
#         x_conv = x_conv + shift
#         if self.use_act:
#             x_conv = F.leaky_relu(x_conv)

#         x_conv = x_conv.view(nb, self.out_channel, nx, ny, nz)
#         return x_conv

# class GINGroupConv3D(nn.Module):
#     def __init__(self, out_channel = 1, in_channel = 1, interm_channel = 2, scale_pool = [1, 3 ], n_layer = 4, out_norm = 'frob', init_scale = 'default', **kwargs):
#         '''
#         GIN
#         '''
#         super(GINGroupConv3D, self).__init__()
#         self.scale_pool = scale_pool # don't make it tool large as we have multiple layers
#         self.n_layer = n_layer
#         self.layers = []
#         self.out_norm = out_norm
#         self.out_channel = out_channel

#         self.layers.append(
#             GradlessGCReplayNonlinBlock3D(out_channel = interm_channel, in_channel = in_channel, scale_pool = scale_pool, init_scale = init_scale, layer_id = 0).cuda()
#                 )
#         for ii in range(n_layer - 2):
#             self.layers.append(
#             GradlessGCReplayNonlinBlock3D(out_channel = interm_channel, in_channel = interm_channel, scale_pool = scale_pool, init_scale = init_scale,layer_id = ii + 1).cuda()
#                 )
#         self.layers.append(
#             GradlessGCReplayNonlinBlock3D(out_channel = out_channel, in_channel = interm_channel, scale_pool = scale_pool, init_scale = init_scale, layer_id = n_layer - 1, use_act = False).cuda()
#                 )

#         self.layers = nn.ModuleList(self.layers)


#     def forward(self, x_in):
#         if isinstance(x_in, list):
#             x_in = torch.cat(x_in, dim = 0)

#         nb, nc, nx, ny, nz = x_in.shape

#         alphas = torch.rand(nb)[:, None, None, None, None] # nb, 1, 1, 1, 1
#         alphas = alphas.repeat(1, nc, 1, 1, 1).cuda() # nb, nc, 1, 1

#         x = self.layers[0](x_in)
#         for blk in self.layers[1:]:
#             x = blk(x)
#         mixed = alphas * x + (1.0 - alphas) * x_in

#         if self.out_norm == 'frob':
#             _in_frob = torch.norm(x_in.view(nb, nc, -1), dim = (-1, -2), p = 'fro', keepdim = False)
#             _in_frob = _in_frob[:, None, None, None, None].repeat(1, nc, 1, 1, 1)
#             _self_frob = torch.norm(mixed.view(nb, self.out_channel, -1), dim = (-1,-2), p = 'fro', keepdim = False)
#             _self_frob = _self_frob[:, None, None, None, None].repeat(1, self.out_channel, 1, 1, 1)
#             mixed = mixed * (1.0 / (_self_frob + 1e-5 ) ) * _in_frob

#         return mixed


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# # ==============================================================================
# # PART 1: Global Intensity Non-linear Augmentation (GIN) - 3D Version
# # Adapted from: models/imagefilter3d.py
# # ==============================================================================

# class GradlessGCReplayNonlinBlock3D(nn.Module):
#     def __init__(self, out_channel=1, in_channel=1, scale_pool=[1, 3], 
#                  layer_id=0, use_act=True, requires_grad=False):
#         """
#         A single layer of the GIN randomly weighted network.
#         It uses Group Convolutions to apply different random kernels to each 
#         image in the batch efficiently.
#         """
#         super(GradlessGCReplayNonlinBlock3D, self).__init__()
#         self.in_channel = in_channel
#         self.out_channel = out_channel
#         self.scale_pool = scale_pool
#         self.layer_id = layer_id
#         self.use_act = use_act
#         self.requires_grad = requires_grad
        
#         # GIN networks are random and do not require gradient updates
#         assert requires_grad == False

#     def forward(self, x_in):
#         # Input shape: [Batch, Channel, D, H, W]
#         nb, nc, nx, ny, nz = x_in.shape
        
#         # 1. Randomly select a kernel size from the pool (e.g., 1 or 3)
#         idx_k = torch.randint(high=len(self.scale_pool), size=(1,))
#         k = self.scale_pool[idx_k[0]]
        
#         # 2. Generate Random Weights on the fly
#         # Shape: [Batch * Out_Chan, In_Chan, k, k, k]
#         # Groups = Batch Size (nb). This means each item in the batch gets 
#         # its own set of random filters.
#         device = x_in.device
#         ker = torch.randn([self.out_channel * nb, self.in_channel, k, k, k], 
#                           requires_grad=self.requires_grad, device=device)
#         shift = torch.randn([self.out_channel * nb, 1, 1, 1], 
#                             requires_grad=self.requires_grad, device=device)
        
#         # 3. Reshape input for Group Conv
#         # Combine Batch and Channel dimensions temporarily
#         x_in_reshaped = x_in.view(1, nb * nc, nx, ny, nz)
        
#         # 4. Perform Convolution
#         # Groups=nb ensures the first 'nc' channels (image 1) are convolved with 
#         # the first chunk of kernels, and so on.
#         x_conv = F.conv3d(x_in_reshaped, ker, stride=1, padding=k // 2, 
#                           dilation=1, groups=nb)
        
#         # 5. Add Bias (Shift)
#         x_conv = x_conv + shift
        
#         # 6. Apply Activation (Leaky ReLU)
#         if self.use_act:
#             x_conv = F.leaky_relu(x_conv)
            
#         # 7. Reshape back to [Batch, Out_Channel, D, H, W]
#         x_conv = x_conv.view(nb, self.out_channel, nx, ny, nz)
#         return x_conv

# class GINGroupConv3D(nn.Module):
#     def __init__(self, out_channel=1, in_channel=1, interm_channel=2, 
#                  scale_pool=[1, 3], n_layer=4, out_norm='frob'):
#         """
#         The main GIN module. Stacks shallow random networks and blends 
#         the output with the original input.
#         """
#         super(GINGroupConv3D, self).__init__()
#         self.scale_pool = scale_pool
#         self.n_layer = n_layer
#         self.out_norm = out_norm
#         self.out_channel = out_channel
#         self.layers = nn.ModuleList()

#         # Layer 1: Input -> Interm
#         self.layers.append(GradlessGCReplayNonlinBlock3D(
#             out_channel=interm_channel, in_channel=in_channel, 
#             scale_pool=scale_pool, layer_id=0))
        
#         # Middle Layers: Interm -> Interm
#         for ii in range(n_layer - 2):
#             self.layers.append(GradlessGCReplayNonlinBlock3D(
#                 out_channel=interm_channel, in_channel=interm_channel, 
#                 scale_pool=scale_pool, layer_id=ii + 1))
        
#         # Final Layer: Interm -> Output (No activation)
#         self.layers.append(GradlessGCReplayNonlinBlock3D(
#             out_channel=out_channel, in_channel=interm_channel, 
#             scale_pool=scale_pool, layer_id=n_layer - 1, use_act=False))

#     def forward(self, x_in):
#         nb, nc, nx, ny, nz = x_in.shape
        
#         # 1. Generate mixing coefficient alpha for Eq. 4 in the paper
#         # alpha ~ U(0, 1)
#         alphas = torch.rand(nb, device=x_in.device)[:, None, None, None, None]
#         alphas = alphas.repeat(1, nc, 1, 1, 1)

#         # 2. Pass through the random shallow network
#         x = self.layers[0](x_in)
#         for blk in self.layers[1:]:
#             x = blk(x)
        
#         # 3. Mix original image and transformed image
#         # Eq 4: alpha * G(x) + (1 - alpha) * x
#         mixed = alphas * x + (1.0 - alphas) * x_in

#         # 4. Re-normalization (Frobenius Norm)
#         # Ensures the augmented image has the same energy as the input
#         if self.out_norm == 'frob':
#             _in_frob = torch.norm(x_in.view(nb, nc, -1), dim=(-1, -2), p='fro', keepdim=False)
#             _in_frob = _in_frob[:, None, None, None, None].repeat(1, nc, 1, 1, 1)
            
#             _self_frob = torch.norm(mixed.view(nb, self.out_channel, -1), dim=(-1, -2), p='fro', keepdim=False)
#             _self_frob = _self_frob[:, None, None, None, None].repeat(1, self.out_channel, 1, 1, 1)
            
#             mixed = mixed * (1.0 / (_self_frob + 1e-5)) * _in_frob
            
#         return mixed

# # ==============================================================================
# # PART 2: Interventional Pseudo-Correlation Augmentation (IPA) - 3D Extension
# # Logic derived from biasfield_interpolate_cchen/adv_bias.py but extended to 3D
# # ==============================================================================

# def get_bspline_kernel_3d(order=2, device='cpu'):
#     """
#     Generates a 3D B-spline kernel for interpolation.
#     This is the 3D equivalent of 'bspline_kernel_2d' from adv_bias.py.
#     """
#     # Start with a simple box filter (order 0 equivalent context)
#     kernel_ones = torch.ones(1, 1, 1, 1, 1, device=device)
#     kernel = kernel_ones
#     padding = 1
    
#     # Convolve with itself 'order' times to approximate B-spline
#     for i in range(1, order + 1):
#         # We pad manually to maintain size growth logic
#         kernel = F.conv3d(kernel, kernel_ones, padding=i) / 2.0 # Approximate scaling
        
#     return kernel

# class IPA_GIN_3D(nn.Module):
#     def __init__(self, in_channels=1, control_point_spacing=32, downscale=2):
#         """
#         The combined IPA + GIN augmentation module.
#         Args:
#             control_point_spacing: Distance between random control points 
#                                    (controls the frequency of the bias field).
#             downscale: Optimization parameter from original code (usually 2 or 4).
#         """
#         super(IPA_GIN_3D, self).__init__()
#         # We use the GIN module defined above
#         self.gin = GINGroupConv3D(out_channel=in_channels, in_channel=in_channels)
        
#         self.spacing = control_point_spacing
#         self.downscale = downscale
#         self.order = 2 # B-spline order

#     def generate_bias_field(self, x_shape, device):
#         """
#         Generates the pseudo-correlation map 'b' (Eq. 6).
#         Logic: Random Control Points -> Upsample via B-Spline Conv -> Normalize
#         """
#         batch_size, _, D, H, W = x_shape
        
#         # 1. Define grid size for control points
#         stride = self.spacing
#         grid_d = int(np.ceil(D / stride)) + 2
#         grid_h = int(np.ceil(H / stride)) + 2
#         grid_w = int(np.ceil(W / stride)) + 2
        
#         # 2. Generate Random Control Points
#         ctrl_pts = torch.rand(batch_size, 1, grid_d, grid_h, grid_w, device=device)
        
#         # 3. Create Interpolation Kernel
#         kernel = get_bspline_kernel_3d(order=self.order, device=device)
#         kernel = kernel.repeat(batch_size, 1, 1, 1, 1) 
        
#         # 4. Upsample (Transposed Convolution)
#         bias = F.conv_transpose3d(ctrl_pts, kernel, stride=stride, padding=self.order, groups=batch_size)
        
#         # 5. Crop to match input image size
#         # This slicing operation makes the tensor non-contiguous in memory
#         curr_d, curr_h, curr_w = bias.shape[2:]
#         d_start = (curr_d - D) // 2
#         h_start = (curr_h - H) // 2
#         w_start = (curr_w - W) // 2
#         bias = bias[:, :, d_start:d_start+D, h_start:h_start+H, w_start:w_start+W]
        
#         # 6. Normalize to [0, 1]
#         # FIX: Use .reshape() instead of .view() to handle non-contiguous memory from step 5
#         bias_flat = bias.reshape(batch_size, -1)
        
#         bias_min = bias_flat.min(dim=1, keepdim=True)[0].reshape(batch_size, 1, 1, 1, 1)
#         bias_max = bias_flat.max(dim=1, keepdim=True)[0].reshape(batch_size, 1, 1, 1, 1)
        
#         bias = (bias - bias_min) / (bias_max - bias_min + 1e-6)
        
#         return bias

#     def forward(self, x):
#         """
#         Implements IPA Equation 6: 
#         T(x) = g1(x) * b + g2(x) * (1-b)
#         """
#         # 1. Generate two different GIN-transformed images.
#         # Since GIN layers generate random weights internally every forward pass,
#         # calling it twice produces two distinct transformations g1 and g2.
#         g1_x = self.gin(x)
#         g2_x = self.gin(x)
        
#         # 2. Generate Pseudo-Correlation Map 'b'
#         b = self.generate_bias_field(x.shape, x.device)

#         # FIX: Check if GIN returned a list (common cause of the error) and extract the tensor
#         # if isinstance(g1_x, (list, tuple)):
#         #     g1_x = g1_x[0]
#         # if isinstance(g2_x, (list, tuple)):
#         #     g2_x = g2_x[0]


        
#         # 3. Spatially Variable Blending
#         out = g1_x * b + g2_x * (1.0 - b)
        
#         return out
        
#         # # 3. Spatially Variable Blending
#         # # Eq 6 from the paper
#         # out = g1_x * b + g2_x * (1.0 - b)
        
#         # return out


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==========================================
# 3D GIN (Global Intensity Non-linear Augmentation)
# ==========================================

class GradlessGCReplayNonlinBlock3D(nn.Module):
    def __init__(self, features, domain_shuffling=False):
        super(GradlessGCReplayNonlinBlock3D, self).__init__()
        self.conv = nn.Conv3d(features, features, kernel_size=1, stride=1, padding=0)
        self.norm = nn.InstanceNorm3d(features)
        self.activation = nn.PReLU()
        self.domain_shuffling = domain_shuffling

    def forward(self, x):
        # x: (B, C, D, H, W)
        out = self.norm(x)
        out = self.activation(out)
        out = self.conv(out)
        
        if self.domain_shuffling:
            # Shuffle batch dimension to mix styles/domains
            perm = torch.randperm(out.size(0)).to(out.device)
            out = out[perm]
            
        return x + out

class GINGroupConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=2, num_blocks=2):
        super(GINGroupConv3D, self).__init__()
        self.num_groups = num_groups
        self.conv_in = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        
        self.blocks = nn.ModuleList([
            GradlessGCReplayNonlinBlock3D(out_channels, domain_shuffling=True)
            for _ in range(num_blocks)
        ])
        
        self.conv_out = nn.Conv3d(out_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, C, D, H, W)
        residual = x
        out = self.conv_in(x)
        
        for block in self.blocks:
            out = block(out)
            
        out = self.conv_out(out)
        
        # Alpha blending for soft application
        alpha = torch.rand(x.size(0), 1, 1, 1, 1, device=x.device)
        return (1 - alpha) * residual + alpha * out

# ==========================================
# 3D IPA (Interventional Pseudo-correlation Augmentation)
# ==========================================

class AdvBias3D(nn.Module):
    def __init__(self, spatial_dims=(32, 128, 128), config=None):
        super(AdvBias3D, self).__init__()
        # Assuming spatial_dims is (D, H, W)
        self.spatial_dims = spatial_dims
        # Control points for the bias field (low resolution grid)
        self.control_points_shape = (4, 4, 4) 
        
    def generate_bias_field(self, batch_size, device):
        # Generate random control points
        # Shape: (B, 1, D_ctrl, H_ctrl, W_ctrl)
        control_points = torch.randn(batch_size, 1, *self.control_points_shape, device=device)
        
        # Upsample to image size using trilinear interpolation to create smooth field
        bias_field = F.interpolate(
            control_points, 
            size=self.spatial_dims, 
            mode='trilinear', 
            align_corners=False
        )
        
        # Normalize or scale bias field if needed
        # Usually bias field is multiplicative and close to 1
        # Let's assume we want variations around 1, e.g., [0.8, 1.2]
        bias_field = bias_field * 0.2 # Scale variance
        bias_field = torch.exp(bias_field) # Ensure positivity
        
        return bias_field

    def forward(self, x):
        # x: (B, C, D, H, W)
        bs, c, d, h, w = x.shape
        # Update spatial dims if dynamic
        if (d, h, w) != self.spatial_dims:
            self.spatial_dims = (d, h, w)
            
        bias = self.generate_bias_field(bs, x.device)
        
        # Apply bias field
        return bias

# ==========================================
# Combined Module
# ==========================================

class IPA_GIN_3D(nn.Module):
    def __init__(self, in_channels=1, gin_channels=16):
        super(IPA_GIN_3D, self).__init__()
        self.gin = GINGroupConv3D(in_channels, gin_channels)
        self.ipa = AdvBias3D()

    def forward(self, x):
        # Apply GIN
        x_gin_1 = self.gin(x)
        x_gin_2 = self.gin(x)

        # Apply IPA
        b = self.ipa(x_gin_1)
        out = x_gin_1 * b + x_gin_2 * (1 - b)
        return out


# ==============================================================================
# Main Execution / Testing
# ==============================================================================
if __name__ == "__main__":
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # Create dummy 3D Input: [Batch=2, Channel=1, Depth=32, Height=64, Width=64]
    input_image = torch.randn(1, 1, 64, 64, 64).to(device)
    
    # Initialize IPA (which includes GIN)
    augmenter = IPA_GIN_3D(in_channels=1, control_point_spacing=16).to(device)
    
    # Forward Pass
    output_image = augmenter(input_image)
    
    print("Input Shape: ", input_image.shape)
    print("Output Shape:", output_image.shape)
    print(f"Output Range: [{output_image.min():.3f}, {output_image.max():.3f}]")
    print("Success: GIN and IPA applied to 3D single-channel input.")