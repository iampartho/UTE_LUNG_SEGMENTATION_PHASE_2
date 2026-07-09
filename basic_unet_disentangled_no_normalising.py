import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# =====================================================================
#  Normalisation options
# =====================================================================

# Default norm type used when ``BasicUNet()`` is called without arguments.
# Switch to "layernorm" to fall back to LayerNorm3d.
NORM_TYPE = "groupnorm"          # "layernorm" or "groupnorm"

# Per-layer GroupNorm schedule.  This implements a hypothesis we want to test
# empirically against a uniform-G baseline:
#
#   - Encoder shallow stages use G == C (Instance-Norm-like) on the
#     assumption that early channels encode heterogeneous low-level features
#     (different intensity/edge responses) and benefit from per-channel
#     standardisation.
#   - As the encoder deepens, G shrinks so more channels share statistics —
#     deep semantic features are more correlated and tolerate (or benefit
#     from) shared normalisation.
#   - The decoder continues shrinking G; the final decoder stage uses G == 1
#     (LayerNorm-like over (C, D, H, W)).  Note: G == 1 does NOT semantically
#     "merge" channels — the channels remain distinct and the next conv
#     still combines them — but it preserves inter-channel magnitude ratios,
#     which the trailing 1x1 ``final_conv`` then collapses to a single logit.
#
# Caveat: this schedule is an experimental design choice, not an established
# best practice.  The literature (Wu & He 2018; nnU-Net) supports a uniform
# G in {8, 32} or InstanceNorm.  Treat this schedule as a hypothesis to
# ablate, not a default to trust.  Pass ``num_groups=8`` (or any int) to the
# constructor to fall back to a uniform schedule for comparison.
DEFAULT_GN_SCHEDULE = {
    "conv_0":     32,   # C=32   -> G=C  (InstanceNorm)
    "conv_1":     32,   # C=64   -> 2 channels per group
    "conv_2":     16,   # C=128  -> 8 channels per group
    "conv_3":     8,   # C=256  -> 32 channels per group  (bottleneck)
    "conv_up_4":   8,   # C=128  -> 16 channels per group
    "conv_up_3":   4,   # C=64   -> 16 channels per group
    "conv_up_2":   1,   # C=32   -> G=1  (LayerNorm-like)
}

# Kept as a convenience for callers that want a single-value default.
NUM_GROUPS = 8


class LayerNorm3d(nn.Module):
    """Channel-axis LayerNorm for 3D feature maps of shape (B, C, D, H, W).

    Normalises across the channel dimension at every (d, h, w) position with a
    per-channel learnable gain / bias — i.e. the same LayerNorm formulation
    used in ConvNeXt for 2D conv features, extended to 3D.  Unlike PyTorch's
    `nn.LayerNorm`, it does NOT need the spatial extent to be known at module
    construction time, so it is compatible with the variable input sizes this
    U-Net is fed via ``VariableSpatialFix``.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, D, H, W) -> (B, D, H, W, C) so the channel dim is last
        x = x.permute(0, 2, 3, 4, 1)
        x = F.layer_norm(x, (x.size(-1),), self.weight, self.bias, self.eps) # explain this line of code : this is the layer norm operation, it normalises the input x across the channel dimension, the first parameter is the shape of the input, the second parameter is the shape of the weight and bias, the third parameter is the weight, the fourth parameter is the bias, the fifth parameter is the epsilon

        return x.permute(0, 4, 1, 2, 3).contiguous()


def _gn(num_channels: int, num_groups: int = NUM_GROUPS) -> nn.GroupNorm:
    """GroupNorm helper that auto-handles channel counts that don't divide
    ``num_groups`` cleanly by walking down to the largest valid divisor."""
    g = min(num_groups, num_channels)
    while num_channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=num_channels)


def _make_norm(num_channels: int, norm_type: str, num_groups: int) -> nn.Module:
    """Dispatch to the requested norm layer.  Single source of truth for the
    layer-construction logic so ``BasicUNet.__init__`` stays terse."""
    nt = norm_type.lower()
    if nt == "layernorm":
        return LayerNorm3d(num_channels)
    if nt == "groupnorm":
        return _gn(num_channels, num_groups)
    raise ValueError(
        f"Unknown norm_type {norm_type!r}; expected 'layernorm' or 'groupnorm'."
    )


# =====================================================================
#  U-Net
# =====================================================================

def _resolve_schedule(num_groups):
    """Turn a flexible ``num_groups`` argument into a full per-stage dict.

    Accepted inputs:
      * ``None``           -> use :data:`DEFAULT_GN_SCHEDULE` verbatim.
      * ``int``            -> uniform G across every stage (the GN-paper /
                              nnU-Net-style baseline).
      * ``dict[str, int]`` -> per-stage override.  Missing keys fall back to
                              :data:`DEFAULT_GN_SCHEDULE`.
    """
    if num_groups is None:
        return dict(DEFAULT_GN_SCHEDULE)
    if isinstance(num_groups, int):
        return {k: num_groups for k in DEFAULT_GN_SCHEDULE}
    if isinstance(num_groups, dict):
        return {**DEFAULT_GN_SCHEDULE, **num_groups}
    raise TypeError(
        f"num_groups must be None, int, or dict[str, int]; got {type(num_groups).__name__}."
    )


class BasicUNet(nn.Module):
    """Basic 3D U-Net with gradient checkpointing.

    Each conv block runs ``act(norm(conv(x)))`` where ``norm`` is selected
    via the ``norm_type`` constructor argument:

    - ``"groupnorm"`` (default) -> :class:`torch.nn.GroupNorm`.  The number
      of groups can vary per stage via the ``num_groups`` argument; see
      :data:`DEFAULT_GN_SCHEDULE` for the rationale and current defaults.
    - ``"layernorm"``           -> :class:`LayerNorm3d` (channel-axis,
      ConvNeXt-style; ignores ``num_groups``).

    The seven encoder/decoder stages are keyed by:
        ``conv_0``, ``conv_1``, ``conv_2``, ``conv_3``  (encoder, deepening)
        ``conv_up_4``, ``conv_up_3``, ``conv_up_2``     (decoder, going up)

    Gradient checkpointing trades compute for memory: intermediate
    activations are recomputed during the backward pass instead of being
    stored.  This is essential at ``batch=1`` with ``256^3`` inputs.
    """

    def __init__(self,
                 norm_type: str = NORM_TYPE,
                 num_groups=None):
        super().__init__()
        self.norm_type = norm_type
        self.gn_schedule = _resolve_schedule(num_groups)

        def n(c, stage):
            return _make_norm(c, norm_type, self.gn_schedule[stage])

        self.conv_0_0 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.norm_0_0 = n(32, "conv_0")
        self.act_0_0 = nn.LeakyReLU(0.1)

        self.conv_0_1 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.norm_0_1 = n(32, "conv_0")
        self.act_0_1 = nn.LeakyReLU(0.1)

        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv_1_0 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.norm_1_0 = n(64, "conv_1")
        self.act_1_0 = nn.LeakyReLU(0.1)

        self.conv_1_1 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.norm_1_1 = n(64, "conv_1")
        self.act_1_1 = nn.LeakyReLU(0.1)

        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv_2_0 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.norm_2_0 = n(128, "conv_2")
        self.act_2_0 = nn.LeakyReLU(0.1)

        self.conv_2_1 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.norm_2_1 = n(128, "conv_2")
        self.act_2_1 = nn.LeakyReLU(0.1)

        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv_3_0 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.norm_3_0 = n(256, "conv_3")
        self.act_3_0 = nn.LeakyReLU(0.1)

        self.conv_3_1 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.norm_3_1 = n(256, "conv_3")
        self.act_3_1 = nn.LeakyReLU(0.1)

        self.up_4 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)

        self.conv_up_4_0 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.norm_up_4_0 = n(128, "conv_up_4")
        self.act_up_4_0 = nn.LeakyReLU(0.1)

        self.conv_up_4_1 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.norm_up_4_1 = n(128, "conv_up_4")
        self.act_up_4_1 = nn.LeakyReLU(0.1)

        self.up_3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)

        self.conv_up_3_0 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.norm_up_3_0 = n(64, "conv_up_3")
        self.act_up_3_0 = nn.LeakyReLU(0.1)

        self.conv_up_3_1 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.norm_up_3_1 = n(64, "conv_up_3")
        self.act_up_3_1 = nn.LeakyReLU(0.1)

        self.up_2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)

        self.conv_up_2_0 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.norm_up_2_0 = n(32, "conv_up_2")
        self.act_up_2_0 = nn.LeakyReLU(0.1)

        self.conv_up_2_1 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.norm_up_2_1 = n(32, "conv_up_2")
        self.act_up_2_1 = nn.LeakyReLU(0.1)

        self.final_conv = nn.Conv3d(32, 1, kernel_size=1)

    def _maybe_checkpoint(self, fn, x):
        """Use checkpointing only when gradients can flow from inputs."""
        if self.training and x.requires_grad:
            return checkpoint(fn, x)
        return fn(x)

    def forward(self, x):
        x1 = self._maybe_checkpoint(lambda x: self.act_0_0(self.norm_0_0(self.conv_0_0(x))), x)
        x1 = self._maybe_checkpoint(lambda x: self.act_0_1(self.norm_0_1(self.conv_0_1(x))), x1)

        x2 = self.pool_1(x1)
        x2 = self._maybe_checkpoint(lambda x: self.act_1_0(self.norm_1_0(self.conv_1_0(x))), x2)
        x2 = self._maybe_checkpoint(lambda x: self.act_1_1(self.norm_1_1(self.conv_1_1(x))), x2)

        x3 = self.pool_2(x2)
        x3 = self._maybe_checkpoint(lambda x: self.act_2_0(self.norm_2_0(self.conv_2_0(x))), x3)
        x3 = self._maybe_checkpoint(lambda x: self.act_2_1(self.norm_2_1(self.conv_2_1(x))), x3)

        x4 = self.pool_3(x3)
        x4 = self._maybe_checkpoint(lambda x: self.act_3_0(self.norm_3_0(self.conv_3_0(x))), x4)
        x4 = self._maybe_checkpoint(lambda x: self.act_3_1(self.norm_3_1(self.conv_3_1(x))), x4)

        x4 = self.up_4(x4)
        x4 = torch.cat([x4, x3], dim=1)
        x4 = self._maybe_checkpoint(lambda x: self.act_up_4_0(self.norm_up_4_0(self.conv_up_4_0(x))), x4)
        x4 = self._maybe_checkpoint(lambda x: self.act_up_4_1(self.norm_up_4_1(self.conv_up_4_1(x))), x4)

        x3 = self.up_3(x4)
        x3 = torch.cat([x3, x2], dim=1)
        x3 = self._maybe_checkpoint(lambda x: self.act_up_3_0(self.norm_up_3_0(self.conv_up_3_0(x))), x3)
        x3 = self._maybe_checkpoint(lambda x: self.act_up_3_1(self.norm_up_3_1(self.conv_up_3_1(x))), x3)

        x2 = self.up_2(x3)
        x2 = torch.cat([x2, x1], dim=1)
        x2 = self._maybe_checkpoint(lambda x: self.act_up_2_0(self.norm_up_2_0(self.conv_up_2_0(x))), x2)
        x2 = self._maybe_checkpoint(lambda x: self.act_up_2_1(self.norm_up_2_1(self.conv_up_2_1(x))), x2)

        x_out = self.final_conv(x2)
        return x_out
