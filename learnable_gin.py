import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableBlock3D(nn.Module):
    """
    Standard Conv-LeakyReLU block to replace GradlessGCReplayNonlinBlock3D.
    Uses learnable weights instead of random on-the-fly weights.
    """
    def __init__(self, out_channel=32, in_channel=3, kernel_size=3, use_act=True):
        super(LearnableBlock3D, self).__init__()
        self.use_act = use_act
        
        # Padding to keep dimensions same: k//2
        padding = kernel_size // 2
        
        # Standard 3D Convolution
        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size, 
                              stride=1, padding=padding, bias=True)
        
        # Initialize weights if needed, though PyTorch default is usually fine.
        # GIN uses Kaiming/He initialization implicitly via randn in the original script? 
        # Actually original used randn without scaling, but then normalized? 
        # Standard initialization is safer for optimization.

    def forward(self, x):
        x = self.conv(x)
        if self.use_act:
            x = F.leaky_relu(x)
        return x

class LearnableGIN3D(nn.Module):
    """
    Learnable version of GIN3D.
    Matches the architecture and normalization of the original GIN3D.
    """
    def __init__(self, out_channel=1, in_channel=1, interm_channel=2, kernel_size=3, n_layer=4, out_norm='frob'):
        super(LearnableGIN3D, self).__init__()
        self.out_channel = out_channel
        self.out_norm = out_norm
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(
            LearnableBlock3D(out_channel=interm_channel, in_channel=in_channel, 
                             kernel_size=kernel_size)
        )
        # Intermediate layers
        for ii in range(n_layer - 2):
            self.layers.append(
                LearnableBlock3D(out_channel=interm_channel, in_channel=interm_channel, 
                                 kernel_size=kernel_size)
            )
        # Output layer (no activation)
        self.layers.append(
            LearnableBlock3D(out_channel=out_channel, in_channel=interm_channel, 
                             kernel_size=kernel_size, use_act=False)
        )
        
        # Learnable mixing coefficient (alpha)
        # Initialize to 0.0 (sigmoid(0) = 0.5)
        self.raw_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x_in):
        nb, nc, nx, ny, nz = x_in.shape
        
        # Pass through layers
        x = self.layers[0](x_in)
        for blk in self.layers[1:]:
            x = blk(x)
        
        # Get alpha in [0, 1]
        alpha = torch.sigmoid(self.raw_alpha)
        
        # Mix original image with network output
        # Broadcasting alpha: (1) -> (1,1,1,1,1) matches (nb, nc, nx, ny, nz) via broadcast
        mixed = alpha * x + (1.0 - alpha) * x_in

        # Re-normalize to match original Frobenius norm (Exact match from GIN3D)
        if self.out_norm == 'frob':
            # Compute input norm
            _in_frob = torch.norm(x_in.view(nb, nc, -1), dim=(-1, -2), p='fro', keepdim=False)
            _in_frob = _in_frob[:, None, None, None, None].repeat(1, nc, 1, 1, 1)
            
            # Compute mixed norm
            _self_frob = torch.norm(mixed.view(nb, self.out_channel, -1), dim=(-1, -2), p='fro', keepdim=False)
            _self_frob = _self_frob[:, None, None, None, None].repeat(1, self.out_channel, 1, 1, 1)
            
            # Normalize
            mixed = mixed * (1.0 / (_self_frob + 1e-5)) * _in_frob

        return mixed

if __name__ == "__main__":
    # Test script
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(2, 1, 32, 32, 32).to(device)
    model = LearnableGIN3D(out_channel=1, in_channel=1, interm_channel=2, kernel_size=3).to(device)
    
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    loss = out.mean()
    loss.backward()
    print("Backward pass successful. Gradients computed.")
    print(f"Alpha grad: {model.raw_alpha.grad}")

