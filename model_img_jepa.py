"""
Phase 1, step 2 model: image encoder ``E_img`` + cross-modal predictor ``P``,
regressing the FROZEN Step-1 mask encoder ``E_mask``.

This wires the cross-modal "two-witnesses" energy that Phase 2 steers:

    E_x(x, m) = || P(E_img(x)) - E_mask(m) ||_1

where
    * ``E_img`` (trainable) is a VJEPAEncoder over the CT image -- same patch grid
      as E_mask (256^3 / patch 32 -> 8^3 = 512 tokens) so its tokens line up
      one-to-one, positionally, with the mask tokens.
    * ``P`` (trainable, ImgPredictor) is a DETERMINISTIC full-token predictor: it
      sees every image token and regresses the FULL set of mask-latent tokens at
      the same grid positions. No token hiding, no mask token, no K-pattern
      averaging -- unlike the V-JEPA pretraining predictor, this is a plain
      image-latent -> mask-latent map (the energy must be deterministic).
    * ``E_mask`` (frozen) is the Step-1 EMA target encoder, loaded from the
      mask-JEPA checkpoint. We deliberately take the ``target_encoder`` (EMA) and
      LayerNorm its output, identical to how phase0_energy_vs_dice.py computes the
      mask-only target -- so the training objective here is the very same energy
      the diagnostics and Phase 2 will use.

Only ``E_img`` and ``P`` are optimised; ``E_mask`` stays frozen (no EMA here --
the target network was already EMA-trained in Step 1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the size-agnostic V-JEPA building blocks (per-forward 3D-RoPE, pre-norm
# transformer Block, the encoder itself, and the shared init / param counter).
from model_vjepa import (
    VJEPAEncoder,
    Block,
    RoPE3D,
    _init_weights,
    count_parameters,
)


def load_frozen_mask_encoder(ckpt_path, device, which="target_encoder"):
    """Build the Step-1 mask encoder from its checkpoint and FREEZE it.

    Args:
        ckpt_path: path to a train_mask_jepa.py checkpoint (carries 'config' and
            both 'encoder' / 'target_encoder' state dicts).
        which: 'target_encoder' (the EMA target -- default, matches the Phase-0
            energy) or 'encoder' (the context encoder).

    Returns:
        (encoder, cfg) where ``encoder`` is a frozen VJEPAEncoder on ``device``
        and ``cfg`` is the saved model config dict.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    enc = VJEPAEncoder(
        in_chans=cfg["in_chans"],
        input_size=cfg["input_size"],
        patch_size=cfg["patch_size"],
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        mlp_ratio=cfg["mlp_ratio"],
    )
    enc.load_state_dict(ckpt[which])
    enc.freeze()                      # requires_grad=False + eval()
    return enc.to(device), cfg


class ImgPredictor(nn.Module):
    """Deterministic full-token predictor P: image-latent -> mask-latent.

    Receives ALL image tokens (B, N, in_dim), projects to the predictor width,
    runs a small RoPE transformer over the full token set, and projects to the
    mask-encoder width (B, N, out_dim). Every position is predicted (no masking),
    because the cross-modal energy must be a deterministic function of (x, m).

    Defaults give the predictor encoder-width (384) and a touch more depth than
    the Step-1 pretraining predictor, since it now carries the full cross-modal
    mapping rather than just filling masked tokens.
    """

    def __init__(self, in_dim, out_dim, pred_dim=384, depth=6, num_heads=6, mlp_ratio=4.0):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, pred_dim)
        self.blocks = nn.ModuleList([Block(pred_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(pred_dim)
        self.output_proj = nn.Linear(pred_dim, out_dim)
        self.rope = RoPE3D(pred_dim // num_heads)
        self.apply(_init_weights)

    def forward(self, img_tokens, grid):
        """img_tokens: (B, N, in_dim); grid: (gh, gw, gd) -> (B, N, out_dim)."""
        b = img_tokens.shape[0]
        seq = self.input_proj(img_tokens)
        cos, sin = self.rope.full(grid, b, seq.device)   # all tokens keep absolute pos
        for blk in self.blocks:
            seq = blk(seq, cos, sin)
        seq = self.norm(seq)
        return self.output_proj(seq)


class ImgJEPA(nn.Module):
    """Container: trainable (E_img, P) + frozen E_mask, plus the energy itself.

    Args:
        mask_ckpt_path: Step-1 mask-JEPA checkpoint for the frozen E_mask.
        device: where to place the frozen E_mask.
        img_embed_dim / img_depth / img_heads: E_img ViT knobs (embed_dim
            defaults to the mask encoder's width so the two latents are
            comparable; depth/heads default to the Step-1 encoder shape).
        pred_dim / pred_depth / pred_heads: ImgPredictor knobs.
        mlp_ratio: shared transformer MLP ratio.
        mask_which: which Step-1 weights to freeze ('target_encoder' = EMA,
            default, matching the Phase-0 energy).

    E_img and E_mask share input_size / patch_size (so the 8^3 token grids line
    up), which is asserted at construction.
    """

    def __init__(
        self,
        mask_ckpt_path=None,
        device=None,
        img_embed_dim=None,
        img_depth=10,
        img_heads=6,
        pred_dim=384,
        pred_depth=6,
        pred_heads=6,
        mlp_ratio=4.0,
        mask_which="target_encoder",
        frozen_mask_encoder=None,
        mask_cfg=None,
    ):
        super().__init__()
        # E_mask: either build+freeze from a Step-1 mask checkpoint (training), or
        # accept an already-built frozen encoder (used by from_checkpoint, so the
        # whole energy reloads from one Step-2 file with no dependency on the
        # original mask checkpoint still being present).
        if frozen_mask_encoder is not None:
            assert mask_cfg is not None, "pass mask_cfg alongside frozen_mask_encoder"
            self.mask_encoder = frozen_mask_encoder
            self.mask_cfg = mask_cfg
        else:
            self.mask_encoder, mask_cfg = load_frozen_mask_encoder(
                mask_ckpt_path, device, which=mask_which
            )
            self.mask_cfg = mask_cfg
        mask_dim = int(self.mask_cfg["embed_dim"])
        img_embed_dim = int(img_embed_dim) if img_embed_dim else mask_dim

        self.encoder = VJEPAEncoder(
            in_chans=self.mask_cfg["in_chans"],
            input_size=self.mask_cfg["input_size"],
            patch_size=self.mask_cfg["patch_size"],
            embed_dim=img_embed_dim,
            depth=img_depth,
            num_heads=img_heads,
            mlp_ratio=mlp_ratio,
        )
        # Token grids must match for a position-wise regression target.
        assert self.encoder.grid == self.mask_encoder.grid, (
            f"E_img grid {self.encoder.grid} != E_mask grid {self.mask_encoder.grid}"
        )

        self.predictor = ImgPredictor(
            in_dim=img_embed_dim,
            out_dim=mask_dim,
            pred_dim=pred_dim,
            depth=pred_depth,
            num_heads=pred_heads,
            mlp_ratio=mlp_ratio,
        )

    def forward(self, image, mask, target_norm=True):
        """One cross-modal forward pass.

        Returns (pred, target), each (B, N, mask_dim). The L1 between them (the
        energy) is computed by the caller as the training loss.
        """
        grid = self.encoder.grid_of(image)
        img_tokens = self.encoder(image, keep_idx=None)        # (B, N, img_dim), all tokens
        pred = self.predictor(img_tokens, grid)                # (B, N, mask_dim)

        with torch.no_grad():
            target = self.mask_encoder(mask, keep_idx=None)     # (B, N, mask_dim)
            if target_norm:
                # LayerNorm (no affine) over the feature dim -- identical to the
                # Step-1 target and to phase0_energy_vs_dice.py, so the trained
                # objective equals the deployed energy.
                target = F.layer_norm(target, (target.shape[-1],))
        return pred, target

    @torch.no_grad()
    def energy(self, image, mask, target_norm=True, reduction="mean"):
        """Cross-modal energy E_x = ||P(E_img(image)) - E_mask(mask)||_1.

        Kept on the model so Phase-0 eval and Phase-2 guidance use the EXACT same
        computation as the training loss. ``reduction`` in {'mean','sum','none'}.
        (Phase 2 will re-enable grad on the prediction branch; this helper is for
        monitoring / diagnostics where no grad is needed.)
        """
        pred, target = self.forward(image, mask, target_norm=target_norm)
        l1 = (pred - target).abs()
        if reduction == "mean":
            return l1.mean()
        if reduction == "sum":
            return l1.sum()
        return l1

    def trainable_parameters(self):
        """E_img + predictor params only (E_mask is frozen)."""
        return list(self.encoder.parameters()) + list(self.predictor.parameters())

    @classmethod
    def from_checkpoint(cls, ckpt_path, device):
        """Rebuild a (eval-ready) ImgJEPA from a train_img_jepa.py checkpoint.

        Fully self-contained: the frozen E_mask is rebuilt from the 'mask_config'
        + 'mask_encoder' state stored inside this same file, so reloading does NOT
        depend on the original Step-1 mask checkpoint still being present.

        Returns (model, config) with the model in eval() mode on ``device``.
        """
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = ckpt["config"]
        mcfg = ckpt["mask_config"]

        e_mask = VJEPAEncoder(
            in_chans=mcfg["in_chans"],
            input_size=mcfg["input_size"],
            patch_size=mcfg["patch_size"],
            embed_dim=mcfg["embed_dim"],
            depth=mcfg["depth"],
            num_heads=mcfg["num_heads"],
            mlp_ratio=mcfg["mlp_ratio"],
        )
        e_mask.load_state_dict(ckpt["mask_encoder"])
        e_mask.freeze()
        e_mask.to(device)

        model = cls(
            device=device,
            img_embed_dim=cfg["img_embed_dim"],
            img_depth=cfg["img_depth"],
            img_heads=cfg["img_heads"],
            pred_dim=cfg["pred_dim"],
            pred_depth=cfg["pred_depth"],
            pred_heads=cfg["pred_heads"],
            mlp_ratio=cfg["mlp_ratio"],
            frozen_mask_encoder=e_mask,
            mask_cfg=mcfg,
        )
        model.encoder.load_state_dict(ckpt["encoder"])
        model.predictor.load_state_dict(ckpt["predictor"])
        model.to(device)
        model.eval()
        return model, cfg
