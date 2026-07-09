"""
V-JEPA 2 style self-supervised pretraining model, adapted to 3D lung anatomy.

This is a scaled-down, 3D adaptation of the V-JEPA 2 pretraining stage
(Assran et al., 2025 -- "V-JEPA 2", Figure 2, *left*). The original method is
designed for video, but the joint-embedding-predictive architecture (JEPA) is
modality agnostic: a transformer makes predictions in a *learned latent space*
rather than in pixel space, which is exactly what we want for learning a prior
over lung anatomy.

Faithful-to-the-paper ingredients (Figure 2 left / Eq. 1):
    * Patchify the volume into a sequence of 3D tubelets via a strided Conv3d.
    * Apply a multi-block mask by *dropping* a subset of the tokens.
    * The (context) encoder ``E_theta`` processes only the *visible* tokens.
    * The encoder outputs are concatenated with learnable mask tokens (whose
      positions are supplied via RoPE) and processed by the predictor ``P_phi``.
    * The predictor outputs are regressed onto prediction targets with an L1
      loss applied *only to the masked patches* (Eq. 1).
    * The prediction targets come from an ``ema`` (exponential-moving-average)
      copy of the encoder, with a ``stop-grad`` on that branch.
    * Relative position is encoded with 3D-RoPE (a 3D extension of rotary
      position embedding), partitioning the head dimension into three axis
      segments -- exactly as described in Sec. 2.1 of the paper.

Scaling-down vs. the paper (the only intentional deviations):
    * Encoder is ~19M params (a "ViT-S/16" depth-10 width-384) instead of the
      300M-1B ViT-L/ViT-g. The paper notes transformers scale easily, so we
      shrink width/depth and keep everything else.
    * Predictor is ~2M params (width-192, depth-4) instead of the 22M ViT-s.
    * 3 spatial axes (D,H,W) instead of (T,H,W); the temporal axis of the paper
      becomes the 3rd spatial axis. Multi-block masking therefore samples 3D
      blocks rather than spatial blocks that span all time.

The trained *context encoder* (``VJEPAEncoder``) is the artefact consumed at
Stage-2 (causality_train_with_ACNN.py) where it is frozen and used to compute a
latent-space anatomy-consistency loss, mirroring how the ACNN ``ShapeAE``
encoder was used for ``L_he``.
"""
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Differentiable Gaussian smoothing.
#
# The segmenter prediction at Stage-2 is a *soft* sigmoid map, not a hard 0/1
# mask. To keep the encoder's input distribution consistent between pretraining
# (binary masks) and Stage-2 (soft predictions), we blur the binary mask with a
# Gaussian filter during pretraining (done in dataset_vjepa.py, on numpy) and
# apply the *same* blur to both branches at Stage-2 (done here, on tensors so it
# stays differentiable for the prediction branch).
# ---------------------------------------------------------------------------
def gaussian_blur3d(x, sigma=1.0, kernel_size=5):
    """Separable 3D Gaussian blur on a (B, C, H, W, D) tensor (differentiable).

    Implemented as three depthwise 1D convolutions (one per spatial axis), so
    gradients flow back through it (needed for the prediction branch of the
    Stage-2 anatomy loss).
    """
    if sigma is None or sigma <= 0:
        return x

    device, dtype = x.device, x.dtype
    coords = torch.arange(kernel_size, dtype=dtype, device=device) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()

    channels = x.shape[1]
    pad = kernel_size // 2

    kx = g.view(1, 1, kernel_size, 1, 1).repeat(channels, 1, 1, 1, 1)
    ky = g.view(1, 1, 1, kernel_size, 1).repeat(channels, 1, 1, 1, 1)
    kz = g.view(1, 1, 1, 1, kernel_size).repeat(channels, 1, 1, 1, 1)

    x = F.conv3d(x, kx, padding=(pad, 0, 0), groups=channels)
    x = F.conv3d(x, ky, padding=(0, pad, 0), groups=channels)
    x = F.conv3d(x, kz, padding=(0, 0, pad), groups=channels)
    return x


# ---------------------------------------------------------------------------
# 3D Rotary Position Embedding (3D-RoPE), Sec. 2.1 of the paper.
# ---------------------------------------------------------------------------
def _split_even(total, parts=3):
    """Split ``total`` (even) into ``parts`` even ints, as equal as possible.

    The paper partitions the head dimension into three "approximately equal"
    segments for the (T,H,W) axes; here the axes are the three spatial dims.
    Each segment must be even because RoPE rotates feature *pairs*.
    """
    assert total % 2 == 0, "head_dim must be even for RoPE"
    base = total // parts
    if base % 2 == 1:
        base -= 1
    dims = [base] * parts
    rem = total - base * parts
    i = 0
    while rem > 0:
        dims[i % parts] += 2
        rem -= 2
        i += 1
    assert sum(dims) == total and all(d % 2 == 0 for d in dims)
    return dims


class RoPE3D(nn.Module):
    """Builds per-token cos/sin tables for 3D rotary position embedding.

    Each token sits on a (gh, gw, gd) grid. The head dimension is split into
    three even segments; segment ``a`` is rotated using the token's coordinate
    along axis ``a``. The tables have shape (N, head_dim // 2) and are gathered
    by token index so that masked / context subsets keep their absolute
    positions.

    The tables are computed *on demand* for whatever grid the current input
    produces (so the encoder/predictor are size-agnostic and can run on native,
    un-resized volumes) and cached per (grid, device). RoPE is parameter-free,
    so nothing here is learned or stored in the checkpoint -- this is fully
    backward compatible with checkpoints trained under a fixed grid.
    """

    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        self.head_dim = int(head_dim)
        self.base = float(base)
        self._cache = {}  # (grid, device_str) -> (cos, sin), each (N, head_dim//2)

    def _build(self, grid, device):
        gh, gw, gd = grid
        dims = _split_even(self.head_dim, 3)
        ii, jj, kk = torch.meshgrid(
            torch.arange(gh), torch.arange(gw), torch.arange(gd), indexing='ij'
        )
        coords = torch.stack([ii.reshape(-1), jj.reshape(-1), kk.reshape(-1)], dim=1).float()
        cos_list, sin_list = [], []
        for axis in range(3):
            pairs = dims[axis] // 2
            inv_freq = self.base ** (-torch.arange(pairs, dtype=torch.float32) / max(pairs, 1))
            ang = coords[:, axis:axis + 1] * inv_freq[None, :]  # (N, pairs)
            cos_list.append(torch.cos(ang))
            sin_list.append(torch.sin(ang))
        cos = torch.cat(cos_list, dim=1).to(device)  # (N, head_dim//2)
        sin = torch.cat(sin_list, dim=1).to(device)
        return cos, sin

    def get(self, grid, device):
        key = (tuple(int(g) for g in grid), str(device))
        if key not in self._cache:
            self._cache[key] = self._build(key[0], device)
        return self._cache[key]

    def gather(self, grid, idx):
        """idx: (B, K) long -> (cos, sin) each (B, K, head_dim//2)."""
        cos, sin = self.get(grid, idx.device)
        return cos[idx], sin[idx]

    def full(self, grid, batch_size, device):
        """All tokens -> (cos, sin) each (B, N, head_dim//2)."""
        cos, sin = self.get(grid, device)
        return cos.unsqueeze(0).expand(batch_size, -1, -1), sin.unsqueeze(0).expand(batch_size, -1, -1)


def _apply_rotary(t, cos, sin):
    """Rotate (B, heads, N, head_dim) tensor with (B, N, head_dim//2) cos/sin."""
    b, h, n, hd = t.shape
    t = t.view(b, h, n, hd // 2, 2)
    t1, t2 = t[..., 0], t[..., 1]
    cos = cos.unsqueeze(1)  # (B, 1, N, hd//2)
    sin = sin.unsqueeze(1)
    r1 = t1 * cos - t2 * sin
    r2 = t2 * cos + t1 * sin
    return torch.stack([r1, r2], dim=-1).view(b, h, n, hd)


# ---------------------------------------------------------------------------
# Transformer building blocks (standard pre-norm ViT with RoPE attention).
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5 # this is the root of the head dimension, used in the attention formula
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, cos, sin):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, heads, N, head_dim)

        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


def _gather_tokens(tokens, idx):
    """tokens: (B, N, C); idx: (B, K) -> (B, K, C)."""
    c = tokens.shape[-1]
    return torch.gather(tokens, 1, idx.unsqueeze(-1).expand(-1, -1, c)) # this is used to gather the tokens at the masked positions


# ---------------------------------------------------------------------------
# Encoder E_theta(.)  -- this is the downstream artefact used at Stage-2.
# ---------------------------------------------------------------------------
class VJEPAEncoder(nn.Module):
    """3D Vision Transformer encoder for the V-JEPA objective.

    With the defaults (input 96^3, patch 16^3 -> 6^3 = 216 tokens, width 384,
    depth 10, 6 heads) this is ~19M parameters, within the 10-20M target.

    Args:
        in_chans: input channels (1 for the binary/soft lung mask).
        input_size: fixed (H, W, D); each dim must be divisible by patch_size.
        patch_size: 3D tubelet size (the paper uses 2x16x16 for video).
        embed_dim / depth / num_heads / mlp_ratio: standard ViT knobs.
    """

    def __init__(
        self,
        in_chans=1,
        input_size=(96, 96, 96),
        patch_size=(16, 16, 16),
        embed_dim=384,
        depth=10,
        num_heads=6,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.input_size = tuple(int(s) for s in input_size)
        self.patch_size = tuple(int(p) for p in patch_size)
        for s, p in zip(self.input_size, self.patch_size):
            assert s % p == 0, f"input_size {self.input_size} not divisible by patch_size {self.patch_size}"

        # Default grid implied by ``input_size`` (used for backward-compat helpers
        # such as ``self.grid`` / ``self.num_patches``). The encoder is, however,
        # size-agnostic: the actual grid is recomputed per forward from the input
        # spatial size, so native un-resized volumes (any size divisible by
        # ``patch_size``) are supported without changing the architecture.
        self.grid = tuple(s // p for s, p in zip(self.input_size, self.patch_size))
        self.num_patches = self.grid[0] * self.grid[1] * self.grid[2]
        self.embed_dim = int(embed_dim)

        self.patch_embed = nn.Conv3d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.rope = RoPE3D(embed_dim // num_heads)

        self.apply(_init_weights)

    def grid_of(self, x):
        """Token grid (gh, gw, gd) implied by the input spatial size."""
        spatial = tuple(int(s) for s in x.shape[2:])
        for s, p in zip(spatial, self.patch_size):
            assert s % p == 0, (
                f"VJEPAEncoder input spatial size {spatial} must be divisible by "
                f"patch_size {self.patch_size}; pad the volume to a multiple of the patch first."
            )
        return tuple(s // p for s, p in zip(spatial, self.patch_size))

    def patchify(self, x):
        x = self.patch_embed(x)               # (B, dim, gh, gw, gd)
        x = x.flatten(2).transpose(1, 2)      # (B, N, dim), C-order over (gh,gw,gd)
        return x

    def forward(self, x, keep_idx=None):
        """Encode a volume.

        If ``keep_idx`` (B, Nkeep) is given, only those (visible/context) tokens
        are processed -- this is the masked context branch of V-JEPA. If it is
        ``None``, all tokens are processed (used for the EMA-target branch and
        for downstream embedding at Stage-2).
        """
        grid = self.grid_of(x)
        t = self.patchify(x)
        b = t.shape[0]
        if keep_idx is None:
            cos, sin = self.rope.full(grid, b, x.device)
        else:
            t = _gather_tokens(t, keep_idx)
            cos, sin = self.rope.gather(grid, keep_idx)

        for blk in self.blocks:
            t = blk(t, cos, sin)
        return self.norm(t)

    def embed(self, x):
        """Mean-pooled global representation (B, embed_dim) over all tokens.

        Kept for backward compatibility / monitoring.  Stage-2 anatomy loss
        now uses ``embed_tokens`` to preserve per-position spatial information.
        """
        tokens = self.forward(x, keep_idx=None)
        return tokens.mean(dim=1)

    def embed_tokens(self, x):
        """Full per-token representation (B, N, embed_dim) over all tokens.

        Used at Stage-2 for the token-level anatomy-consistency loss.  Comparing
        every token individually (then reducing) is strictly stronger than
        comparing mean-pooled vectors, because errors at different spatial
        positions cannot cancel each other out before the loss is applied.

        N = grid[0] * grid[1] * grid[2]  (e.g. 6*6*6 = 216 for a 96^3 volume
        with 16^3 patches).  Each of the N tokens corresponds to one 16^3 patch
        and carries the encoder's description of that local region of the mask.
        """
        return self.forward(x, keep_idx=None)   # (B, N, embed_dim)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()


# ---------------------------------------------------------------------------
# Predictor P_phi(.)
# ---------------------------------------------------------------------------
class VJEPAPredictor(nn.Module):
    """Lightweight transformer predicting masked-token representations.

    With the defaults (width 192, depth 4, 4 heads) this is ~2M parameters.
    It receives the context-encoder tokens (projected to the predictor width)
    together with learnable mask tokens placed at the masked positions, and
    regresses the representations of the masked tokens.
    """

    def __init__(
        self,
        encoder_dim,
        grid=None,
        pred_dim=192,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
    ):
        super().__init__()
        # ``grid`` is accepted for backward-compatible construction but no longer
        # constrains the predictor: RoPE is built per actual grid at forward time.
        self.input_proj = nn.Linear(encoder_dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        self.blocks = nn.ModuleList([Block(pred_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(pred_dim)
        self.output_proj = nn.Linear(pred_dim, encoder_dim)
        self.rope = RoPE3D(pred_dim // num_heads)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.apply(_init_weights)

    def forward(self, ctx_tokens, keep_idx, mask_idx, grid):
        """Predict representations at ``mask_idx`` positions.

        Args:
            ctx_tokens: (B, Nkeep, encoder_dim) from the context encoder.
            keep_idx:   (B, Nkeep) positions of the visible/context tokens.
            mask_idx:   (B, Nmask) positions of the masked (predicted) tokens.
            grid:       (gh, gw, gd) token grid of the current input (for RoPE).

        Returns:
            (B, Nmask, encoder_dim) predicted representations.
        """
        b = ctx_tokens.shape[0]
        n_mask = mask_idx.shape[1] # number of masked tokens

        ctx = self.input_proj(ctx_tokens)               # (B, Nkeep, pred_dim)
        masks = self.mask_token.expand(b, n_mask, -1)   # (B, Nmask, pred_dim)
        seq = torch.cat([ctx, masks], dim=1)            # order: [context, masked]

        cos_k, sin_k = self.rope.gather(grid, keep_idx)
        cos_m, sin_m = self.rope.gather(grid, mask_idx)
        cos = torch.cat([cos_k, cos_m], dim=1)
        sin = torch.cat([sin_k, sin_m], dim=1)

        for blk in self.blocks:
            seq = blk(seq, cos, sin)
        seq = self.norm(seq)

        pred = seq[:, -n_mask:, :]                       # outputs at masked positions
        return self.output_proj(pred)


# ---------------------------------------------------------------------------
# Full V-JEPA module: encoder + predictor + EMA target encoder.
# ---------------------------------------------------------------------------
class VJEPA(nn.Module):
    """Container that wires the context encoder, predictor and EMA target.

    Only used during pretraining (train_vjepa.py). At Stage-2 we load just the
    ``encoder`` (or the ``target_encoder``) as a frozen feature extractor.
    """

    def __init__(
        self,
        in_chans=1,
        input_size=(96, 96, 96),
        patch_size=(16, 16, 16),
        embed_dim=384,
        depth=10,
        num_heads=6,
        mlp_ratio=4.0,
        pred_dim=192,
        pred_depth=4,
        pred_num_heads=4,
    ):
        super().__init__()
        self.encoder = VJEPAEncoder(
            in_chans=in_chans,
            input_size=input_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        self.predictor = VJEPAPredictor(
            encoder_dim=embed_dim,
            grid=self.encoder.grid,
            pred_dim=pred_dim,
            depth=pred_depth,
            num_heads=pred_num_heads,
            mlp_ratio=mlp_ratio,
        )

        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.target_encoder.eval()

    @torch.no_grad()
    def update_target(self, momentum):
        """EMA update of the target encoder: theta_bar <- m*theta_bar + (1-m)*theta."""
        for pe, pt in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            pt.data.mul_(momentum).add_(pe.data, alpha=1.0 - momentum)
        for be, bt in zip(self.encoder.buffers(), self.target_encoder.buffers()):
            bt.data.copy_(be.data) # this is used to copy the data from the encoder to the target encoder, it is necessary to copy the buffers because the parameters are not saved to the checkpoint

    def forward(self, x, keep_idx, mask_idx, target_norm=True):
        """One V-JEPA forward pass.

        Returns (pred, target) each of shape (B, Nmask, embed_dim). The L1 loss
        between them (Eq. 1) is computed by the caller.
        """
        grid = self.encoder.grid_of(x)
        ctx = self.encoder(x, keep_idx=keep_idx)
        pred = self.predictor(ctx, keep_idx, mask_idx, grid) # keep_idx is the indices of the context tokens, mask_idx is the indices of the masked tokens

        with torch.no_grad():
            target_full = self.target_encoder(x, keep_idx=None)   # (B, N, dim)
            target = _gather_tokens(target_full, mask_idx)         # (B, Nmask, dim)
            if target_norm:
                # LayerNorm (no affine) over the feature dim stabilises JEPA
                # targets and prevents trivial-scale solutions.
                target = F.layer_norm(target, (target.shape[-1],))
        return pred, target


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    elif isinstance(m, nn.Conv3d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Multi-block 3D masking (the V-JEPA masking strategy, adapted to 3 spatial
# axes). Produces a *fixed* number of masked tokens per sample so the batched
# tensors have consistent shapes (mirrors the V-JEPA mask collator).
# ---------------------------------------------------------------------------
def _sample_block(grid, scale_range, aspect_range, rng):
    gh, gw, gd = grid
    n = gh * gw * gd
    frac = rng.uniform(*scale_range)
    vol = frac * n

    ar1 = rng.uniform(*aspect_range)
    ar2 = rng.uniform(*aspect_range)
    bh = (vol / (ar1 * ar2)) ** (1.0 / 3.0)
    bw = bh * ar1
    bd = bh * ar2

    bh = int(max(1, min(gh, round(bh))))
    bw = int(max(1, min(gw, round(bw))))
    bd = int(max(1, min(gd, round(bd))))

    top = rng.randint(0, gh - bh + 1)
    left = rng.randint(0, gw - bw + 1)
    front = rng.randint(0, gd - bd + 1)
    return top, bh, left, bw, front, bd


def generate_masks(
    batch_size,
    grid,
    mask_ratio=0.6,
    num_blocks=4,
    block_scale=(0.05, 0.25),
    block_aspect=(0.75, 1.5),
    device='cpu',
    rng=None,
):
    """Generate (keep_idx, mask_idx) for a batch.

    Returns:
        keep_idx: (B, Nkeep) long tensor of visible/context token indices.
        mask_idx: (B, Nmask) long tensor of masked/predicted token indices.
    where Nmask = round(mask_ratio * N) and Nkeep = N - Nmask are identical for
    every sample in the batch.
    """
    rng = rng or np.random
    gh, gw, gd = grid
    n = gh * gw * gd
    num_mask = max(1, min(n - 1, int(round(mask_ratio * n))))

    keep_list, mask_list = [], []
    for _ in range(batch_size):
        m = np.zeros((gh, gw, gd), dtype=bool)
        attempts = 0
        while m.sum() < num_mask and attempts < num_blocks * 4:
            top, bh, left, bw, front, bd = _sample_block(grid, block_scale, block_aspect, rng)
            m[top:top + bh, left:left + bw, front:front + bd] = True
            attempts += 1

        flat = m.reshape(-1)
        masked = np.where(flat)[0]
        unmasked = np.where(~flat)[0]

        if len(masked) > num_mask:
            masked = rng.choice(masked, num_mask, replace=False)
        elif len(masked) < num_mask:
            extra = rng.choice(unmasked, num_mask - len(masked), replace=False)
            masked = np.concatenate([masked, extra])

        masked = np.sort(masked)
        keep = np.setdiff1d(np.arange(n), masked, assume_unique=True)
        mask_list.append(masked)
        keep_list.append(keep)

    keep_idx = torch.as_tensor(np.stack(keep_list), dtype=torch.long, device=device)
    mask_idx = torch.as_tensor(np.stack(mask_list), dtype=torch.long, device=device)
    return keep_idx, mask_idx


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())
