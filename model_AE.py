"""
Stage-1 shape autoencoder for ACNN (Oktay et al., 2018, IEEE TMI).

Paper-faithful binary lung-mask AE:

    input (B, 1, H_in, W_in, D_in)   -- H_in/W_in/D_in are FIXED to ``input_size``
        -> 4 conv blocks (Conv3d + BatchNorm3d + LeakyReLU) with 3 stride-2 downsamples
        -> flatten + Linear -> 64-D code              (paper Sec. II-B)
        -> Linear + reshape back to (B, c4, H/8, W/8, D/8)
        -> 3 ConvTranspose3d (stride 2) + conv refinement blocks
        -> 1-channel output logits (same spatial shape as input)

Because the bottleneck is a fully-connected layer, the AE expects a FIXED
input size. ``ShapeAEDataset`` (dataset_AE.py) handles the resize. In
Stage-2 (causality_train_with_ACNN.py) the segmenter prediction and the
ground-truth mask are resized via ``F.interpolate`` to ``input_size``
before passing through the frozen encoder for the L_he shape-regularisation
loss.

Used in two places:
    * train_AE.py        : stage-1 training of the whole AE.
    * causality_train_with_ACNN.py : encoder is frozen and used to compute
      the latent-space shape regularisation loss L_he (paper Eq. 1).
"""
import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    """Two 3x3x3 conv + BatchNorm + LeakyReLU layers.

    BatchNorm is used here (matching the paper's stacked conv AE style) because
    the AE is trained with a large batch size (64), which makes BatchNorm stable
    and well-suited. InstanceNorm is intentionally avoided here -- it is used in
    the segmenter (BasicUNet) for domain-generalisation reasons, but the AE only
    sees binary masks where no domain shift exists.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm3d(out_channels)
        self.act1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm3d(out_channels)
        self.act2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x


class ShapeAE(nn.Module):
    """Convolutional autoencoder for binary lung-mask shape priors.

    Paper-faithful version: fixed input size, flatten + Linear bottleneck,
    ConvTranspose3d-based decoder.

    Args:
        code_dim: dimension of the global latent code (paper used 64).
        base_channels: number of feature maps in the first encoder block.
            Doubles at every level: base, 2*base, 4*base, 8*base.
        input_size: fixed (H, W, D) the AE accepts. Each dim must be
            divisible by 8 (3 stride-2 downsamples).
    """

    def __init__(self, code_dim=64, base_channels=16, input_size=(96, 96, 96)):
        super().__init__()
        self.code_dim = int(code_dim)
        self.input_size = tuple(int(s) for s in input_size)
        for s in self.input_size:
            assert s % 8 == 0, (
                f"input_size must be divisible by 8 (3 stride-2 downsamples), "
                f"got {self.input_size}"
            )

        c1 = int(base_channels)
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 8

        self.enc1 = _ConvBlock(1, c1)
        self.down1 = nn.Conv3d(c1, c1, kernel_size=2, stride=2)

        self.enc2 = _ConvBlock(c1, c2)
        self.down2 = nn.Conv3d(c2, c2, kernel_size=2, stride=2)

        self.enc3 = _ConvBlock(c2, c3)
        self.down3 = nn.Conv3d(c3, c3, kernel_size=2, stride=2)

        self.enc4 = _ConvBlock(c3, c4)

        self.deepest_size = tuple(s // 8 for s in self.input_size)
        self.deepest_channels = c4
        self.flatten_dim = (
            c4
            * self.deepest_size[0]
            * self.deepest_size[1]
            * self.deepest_size[2]
        )

        self.fc_enc = nn.Linear(self.flatten_dim, self.code_dim)
        self.fc_dec = nn.Linear(self.code_dim, self.flatten_dim)

        self.up3 = nn.ConvTranspose3d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = _ConvBlock(c3, c3)

        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = _ConvBlock(c2, c2)

        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = _ConvBlock(c1, c1)

        self.final_conv = nn.Conv3d(c1, 1, kernel_size=1)

    def encode(self, x):
        """Encode a (B, 1, *input_size) tensor to a (B, code_dim) latent code."""
        assert tuple(x.shape[2:]) == self.input_size, (
            f"ShapeAE expects spatial size {self.input_size}, got {tuple(x.shape[2:])}. "
            f"Resize the input before passing it to ShapeAE."
        )

        h = self.enc1(x)
        h = self.down1(h)

        h = self.enc2(h)
        h = self.down2(h)

        h = self.enc3(h)
        h = self.down3(h)

        h = self.enc4(h)

        h_flat = h.view(h.size(0), -1)
        z = self.fc_enc(h_flat)
        return z

    def decode(self, z):
        """Decode a (B, code_dim) latent code back to (B, 1, *input_size) logits."""
        h = self.fc_dec(z)
        h = h.view(z.size(0), self.deepest_channels, *self.deepest_size)

        h = self.up3(h)
        h = self.dec3(h)

        h = self.up2(h)
        h = self.dec2(h)

        h = self.up1(h)
        h = self.dec1(h)

        out = self.final_conv(h)
        return out

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    def encode_only(self, x):
        """Convenience wrapper used at Stage-2; returns just the code tensor."""
        return self.encode(x)

    def freeze_encoder(self):
        """Freeze every parameter that participates in ``encode``.

        This is what Stage-2 (causality_train_with_ACNN.py) calls before
        starting the segmenter optimisation.
        """
        modules_to_freeze = [
            self.enc1, self.down1,
            self.enc2, self.down2,
            self.enc3, self.down3,
            self.enc4,
            self.fc_enc,
        ]
        for m in modules_to_freeze:
            for p in m.parameters():
                p.requires_grad = False
            m.eval()
