"""
compute_local_complexity_unet.py

Computes local complexity (LC) of a trained BasicUNet on specified 3D scans.

Based on: "Deep Networks Always Grok and Here is Why"
          Humayun, Balestriero, Baraniuk (ICML 2024)

LC measures the density of activation-pattern changes (neuron hyperplane
crossings) inside a cross-polytopal neighborhood around each input sample,
approximating the local non-linearity of the learned input-output mapping.

Compatibility with BasicUNet
-----------------------------
- LeakyReLU is continuous piecewise-linear → paper validates this (Appendix A).
- 3D convolutions are circulant weight matrices → same theory as 2D (Sec 2.1).
- Skip / residual connections → paper validates on ResNet18 (Fig 1, 23).
- InstanceNorm at eval time is a fixed per-sample affine transform and does not
  cause the partition-adaptation issues that BatchNorm introduces (Sec 4 / App B).
- MaxPool is piecewise-linear; its nonlinearities are implicitly captured through
  the neighborhood deformation but not explicitly counted (same as reference code).

Hookable layers (pre-LeakyReLU)
--------------------------------
The pre-LeakyReLU tensor is whichever module's output feeds directly into the
activation in the forward pass.  This depends on the BasicUNet variant:
    - With InstanceNorm  (act(norm(conv(x))))  → hook the norm_* modules.
    - Without InstanceNorm (act(conv(x)))       → hook the conv_* modules.
Use the HOOK_TARGET configuration variable below to switch between the two.

Encoder:  enc_0_0, enc_0_1  (scale 1)
          enc_1_0, enc_1_1  (scale 1/2)
          enc_2_0, enc_2_1  (scale 1/4)
          enc_3_0, enc_3_1  (scale 1/8, bottleneck)
Decoder:  dec_4_0, dec_4_1  (scale 1/4)
          dec_3_0, dec_3_1  (scale 1/2)
          dec_2_0, dec_2_1  (scale 1)

Voxel-specific LC (no extra forward passes)
--------------------------------------------
After the standard N_HULL forward passes, activation signs at every layer are
available.  For each requested output-voxel (d, h, w), we map to the
corresponding spatial position in each layer (d // scale, h // scale, w // scale)
and count sign changes across the channel dimension at that position.
The final_conv (1x1 Conv3d, no activation) is purely linear, so it adds zero
hyperplane crossings — voxel LC is fully determined by preceding layers.
Thousands of voxel indices are handled via vectorised advanced indexing.
"""

import random
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import csv
import time
from torchvision import transforms
from basic_unet_disentangled import BasicUNet

from eval_step_1 import normalise_one_one, normalise_zero_one, normalise_hu, VariableSpatialFix, ToTensor
from save_lc_nifti import save_lc_nifti

# =====================================================================
#  CONFIGURATION — edit these variables before running
# =====================================================================

# df = pd.read_csv("./ids/UTE_MRI_previous_numpy_without_clipping.csv")
# SCAN_PATHS = df["filepaths"].tolist()
# random.shuffle(SCAN_PATHS)
# SCAN_PATHS = SCAN_PATHS[:110]
SCAN_PATHS = ["/Shared/lss_segerard/parthghosh/data/COPDgene_CT_1.25mm_numpy/15278Y_FRC.npy"]
SCAN_TYPE = ["CT" for _ in range(len(SCAN_PATHS))]

# SCAN_PATHS = [
#     "/Shared/lss_segerard/parthghosh/data/UTE_new_data_numpy/103-015/20171212/AnatCorrLungs.npy"# "/path/to/scan1.npy",
#     # "/path/to/scan2.npy",
# ]

# SCAN_TYPE = [
#     "UTE"
# #     "CT",
# ]
MODEL_WEIGHT_PATH = "./save_models/best_bunet_causality_paper_ct_train_UTE_test_w_tversky_wo_kl_only_gin_roughness_enforced_5_normalised_gin.pth"

OUTPUT_DIR = "log_local_complexity"

RADIUS = 0.005          # Neighbourhood radius — keep small for deep nets (paper uses 0.005)
N_HULL = 10             # Must be even.  P = N_HULL/2 orthogonal directions.
HULL_SEED = 42          # Reproducibility seed for hull sampling
HULL_BATCH_SIZE = 2     # Hull vertices forwarded per sub-batch (tune for GPU RAM)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Voxel-specific LC ---
COMPUTE_VOXEL_LC = True
SAVE_LC_NIFTI    = True   # Save a NIfTI image encoding per-voxel LC category (0/1/2/3)
VOXEL_INDICES = [
    # (d, h, w) tuples in output-space coordinates, e.g.:
    # (0, 120, 70),
    # (64, 64, 64),
]

# Which layers to include in the report.
# "all" → all 14 hookable layers.
# Or provide a list of names, e.g. ["enc_0_0", "enc_3_1", "dec_2_1"]
TARGET_LAYERS = "all"

# Where to attach the forward hooks (i.e. what counts as "pre-activation").
#   "conv" → hook the Conv3d outputs   (use when BasicUNet runs WITHOUT norm).
#   "norm" → hook the InstanceNorm3d outputs (use when BasicUNet runs WITH norm).
HOOK_TARGET = "conv"

# --- Common input shape (keeps the hull-direction cache to a single entry) ---
# Every scan is forced to TARGET_SHAPE before LC computation.  Validation scans
# are already 256^3 isotropic (no-op); larger training scans are centre-cropped
# and smaller ones are symmetrically padded.  Because the cached hull directions
# are keyed by input size, a single common shape means the cache holds exactly
# one tensor (~335 MB at 256^3, N_HULL=10) reused for all scans across all epochs.
TARGET_SHAPE = None #(256, 256, 256)

# Pad value used when a scan is smaller than TARGET_SHAPE on an axis.
# Images are in [-1, 1] after normalisation, so 0.0 == "zero pad" as requested.
# (Switch to -1.0 if you would rather the padding match the CT/air background.)
IMAGE_PAD_VALUE = 0.0
# Ground-truth lung-mask background is 0.
MASK_PAD_VALUE = 0


# =====================================================================
#  Layer definitions:  (display_name, conv_attribute, norm_attribute, spatial_scale_divisor)
# =====================================================================

ALL_LAYER_INFO = [
    ("enc_0_0", "conv_0_0",    "norm_0_0",    1),
    ("enc_0_1", "conv_0_1",    "norm_0_1",    1),
    ("enc_1_0", "conv_1_0",    "norm_1_0",    2),
    ("enc_1_1", "conv_1_1",    "norm_1_1",    2),
    ("enc_2_0", "conv_2_0",    "norm_2_0",    4),
    ("enc_2_1", "conv_2_1",    "norm_2_1",    4),
    ("enc_3_0", "conv_3_0",    "norm_3_0",    8),
    ("enc_3_1", "conv_3_1",    "norm_3_1",    8),
    ("dec_4_0", "conv_up_4_0", "norm_up_4_0", 4),
    ("dec_4_1", "conv_up_4_1", "norm_up_4_1", 4),
    ("dec_3_0", "conv_up_3_0", "norm_up_3_0", 2),
    ("dec_3_1", "conv_up_3_1", "norm_up_3_1", 2),
    ("dec_2_0", "conv_up_2_0", "norm_up_2_0", 1),
    ("dec_2_1", "conv_up_2_1", "norm_up_2_1", 1),
]


# =====================================================================
#  Shape normalisation
# =====================================================================

def center_crop_or_pad(volume, target_shape, pad_value=0.0):
    """Force a 3D array to ``target_shape`` via per-axis centre crop / pad.

    Each axis is handled independently:
      * axis longer  than target → centre-crop (drop equal amounts each side);
      * axis shorter than target → symmetric pad with ``pad_value``;
      * axis equal             → left unchanged.

    No interpolation/resampling of voxel values occurs — values are only
    cropped or padded — so real intensities are never blended.

    Parameters
    ----------
    volume       : (D, H, W) numpy array.
    target_shape : (D, H, W) desired size (e.g. (256, 256, 256)).
    pad_value    : constant used for padding (0.0 for [-1,1] images, 0 for masks).

    Returns
    -------
    numpy array with shape exactly ``target_shape`` and the input dtype.
    """
    result = volume
    for axis in range(result.ndim):
        current_len = result.shape[axis]
        target_len = target_shape[axis]

        if current_len == target_len:
            continue

        if current_len > target_len:
            # Too long: keep a centred chunk.
            extra = current_len - target_len
            start = extra // 2
            slicer = [slice(None)] * result.ndim
            slicer[axis] = slice(start, start + target_len)
            result = result[tuple(slicer)]
        else:
            # Too short: pad symmetrically with pad_value.
            needed = target_len - current_len
            pad_left = needed // 2
            pad_right = needed - pad_left
            padding = [(0, 0)] * result.ndim
            padding[axis] = (pad_left, pad_right)
            result = np.pad(result, padding, mode="constant",
                            constant_values=pad_value)

    assert result.shape == tuple(target_shape), (
        f"center_crop_or_pad produced {result.shape}, expected {tuple(target_shape)}"
    )
    return result.astype(volume.dtype)


def center_crop_or_pad_tensor(x, target_shape, pad_value=0.0):
    """Torch equivalent of ``center_crop_or_pad`` for a ``(1, C, D, H, W)`` tensor.

    Only the trailing three spatial dims are touched (channel/batch are kept).
    Each axis longer than ``target_shape`` is centre-cropped; each axis shorter
    is symmetrically padded with ``pad_value``.  No interpolation occurs.

    Parameters
    ----------
    x            : (1, C, D, H, W) tensor.
    target_shape : (D, H, W) desired spatial size.
    pad_value    : constant used for padding (0.0 for [-1, 1] images).

    Returns
    -------
    tensor on the same device/dtype as ``x`` with spatial size ``target_shape``.
    """
    assert x.dim() == 5, f"expected (1, C, D, H, W), got {tuple(x.shape)}"

    # 1) Centre-crop any spatial axis that is too long.
    for axis, tgt in zip((2, 3, 4), target_shape):
        cur = x.shape[axis]
        if cur > tgt:
            start = (cur - tgt) // 2
            x = x.narrow(axis, start, tgt)

    # 2) Symmetric-pad any spatial axis that is too short.  F.pad consumes the
    #    pad spec from the LAST dim backwards: (W_l, W_r, H_l, H_r, D_l, D_r).
    def _lr(cur, tgt):
        if cur >= tgt:
            return 0, 0
        need = tgt - cur
        left = need // 2
        return left, need - left

    d_l, d_r = _lr(x.shape[2], target_shape[0])
    h_l, h_r = _lr(x.shape[3], target_shape[1])
    w_l, w_r = _lr(x.shape[4], target_shape[2])

    if any((d_l, d_r, h_l, h_r, w_l, w_r)):
        x = F.pad(x, (w_l, w_r, h_l, h_r, d_l, d_r),
                  mode="constant", value=pad_value)

    assert tuple(x.shape[2:]) == tuple(target_shape), (
        f"center_crop_or_pad_tensor produced {tuple(x.shape[2:])}, "
        f"expected {tuple(target_shape)}"
    )
    return x


# =====================================================================
#  Hull sampling for 3D volumes
# =====================================================================

@torch.no_grad()
def get_ortho_hull_3d(x, r=0.005, n=10, seed=42):
    """
    Sample cross-polytopal hull vertices around a single 3D volume.

    NOTE: ``UNetLocalComplexity`` now builds the hull via its cached
    ``_build_hull`` / ``_get_unit_dirs`` methods (which reuse directions across
    scans/epochs).  This standalone function is kept for reference and produces
    numerically identical hull vertices, but is no longer on the hot path.

    x     : (1, C, D, H, W)
    r     : radius of the cross-polytope
    n     : number of vertices (must be even; P = n/2 orthogonal directions)
    seed  : random seed

    Returns (n, C, D, H, W) — the hull vertices (centre not included).
    """
    assert n % 2 == 0, "n must be even"
    if seed is not None:
        torch.manual_seed(seed)

    flat_dim = int(np.prod(x.shape[1:]))
    n_dirs = n // 2

    orth = torch.nn.utils.parametrizations.orthogonal(
        torch.nn.Linear(flat_dim, n_dirs).to(x.device),
        use_trivialization=False,
    )
    dirs = orth.weight * r                        # (n_dirs, flat_dim) — unit rows scaled by r

    x_flat = x.reshape(1, -1)                     # (1, flat_dim)
    hull = torch.cat([x_flat + dirs,
                      x_flat - dirs], dim=0)       # (n, flat_dim)
    return hull.reshape(n, *x.shape[1:])


# =====================================================================
#  Core LC computation class
# =====================================================================

class UNetLocalComplexity:

    def __init__(self, model, device="cuda", target_layers="all", hook_target="conv"):
        self.model = model
        self.device = device
        self.hooks = []
        self.activation_buffer = {}
        # Cache of orthonormal hull directions, keyed by (flat_size, n_dirs, seed).
        # Directions are a deterministic function of those three values only
        # (NOT of scan content), so they can be computed once and reused for
        # every scan of the same padded shape across the whole training run.
        # The master copy lives on CPU and is moved to the device per use; the
        # cache therefore grows with the number of DISTINCT input shapes, not
        # with the number of scans or epochs.
        self._dirs_cache = {}

        if hook_target not in ("conv", "norm"):
            raise ValueError(
                f"hook_target must be 'conv' or 'norm', got {hook_target!r}"
            )
        self.hook_target = hook_target

        if target_layers == "all":
            self.layer_info = ALL_LAYER_INFO[:]
        else:
            self.layer_info = [
                (n, ca, na, s)
                for n, ca, na, s in ALL_LAYER_INFO
                if n in target_layers
            ]
        self.layer_names = [name for name, _, _, _ in self.layer_info]

    # -----------------------------------------------------------------
    #  Hook management
    # -----------------------------------------------------------------

    def _attr_for(self, conv_attr, norm_attr):
        return conv_attr if self.hook_target == "conv" else norm_attr

    def _register_hooks(self):
        """Attach forward hooks on the pre-LeakyReLU module outputs.

        Which module that is depends on `self.hook_target`:
            "conv" → Conv3d output (BasicUNet variant without InstanceNorm).
            "norm" → InstanceNorm3d output (BasicUNet variant with InstanceNorm).
        """
        for name, conv_attr, norm_attr, _ in self.layer_info:
            attr = self._attr_for(conv_attr, norm_attr)
            module = getattr(self.model, attr)

            def _make_hook(hook_name):
                def hook_fn(_mod, _inp, out):
                    self.activation_buffer[hook_name] = out.detach()
                return hook_fn

            self.hooks.append(module.register_forward_hook(_make_hook(name)))

    def _remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.activation_buffer.clear()

    # -----------------------------------------------------------------
    #  Cached hull directions
    # -----------------------------------------------------------------

    @torch.no_grad()
    def _get_unit_dirs(self, x, n, seed):
        """Return cached orthonormal hull directions (unit rows) on ``x.device``.

        The directions are the rows of an orthonormal matrix produced by the
        Stiefel-manifold (orthogonal) parametrization, exactly as in the
        original ``get_ortho_hull_3d`` — but here they are computed once per
        ``(flat_size, n_dirs, seed)`` and reused on every subsequent call.

        The radius ``r`` is intentionally NOT applied here, so the cache is
        radius-independent (scaling by ``r`` happens in ``_build_hull``).  The
        master copy is kept on CPU and moved to the device per use; the H2D
        copy is negligible next to a single forward pass.
        """
        assert n % 2 == 0, "n must be even"
        n_dirs = n // 2
        flat_dim = int(np.prod(x.shape[1:]))
        key = (flat_dim, n_dirs, seed)

        cached = self._dirs_cache.get(key)
        if cached is None:
            if seed is not None:
                torch.manual_seed(seed)
            orth = torch.nn.utils.parametrizations.orthogonal(
                torch.nn.Linear(flat_dim, n_dirs).to(x.device),
                use_trivialization=False,
            )
            # (n_dirs, flat_dim) unit rows; detached CPU master copy.
            cached = orth.weight.detach().clone().cpu()
            self._dirs_cache[key] = cached
            print(f"[LC] cached hull directions for flat={flat_dim}, "
                  f"n_dirs={n_dirs} ({len(self._dirs_cache)} shape(s) cached)")

        return cached.to(x.device)

    @torch.no_grad()
    def _build_hull(self, x, r, n, seed):
        """Assemble the cross-polytope hull from cached unit directions.

        Numerically identical to the original ``get_ortho_hull_3d``:
        the reference vertex is ``hull[0] = x + r * dir_0`` and the centre is
        not included; only the direction generation is now cached.

        x : (1, C, D, H, W)  ->  returns (n, C, D, H, W)
        """
        dirs = self._get_unit_dirs(x, n, seed) * r        # (n_dirs, flat_dim)
        x_flat = x.reshape(1, -1)                          # (1, flat_dim)
        hull = torch.cat([x_flat + dirs, x_flat - dirs], dim=0)  # (n, flat_dim)
        return hull.reshape(n, *x.shape[1:])

    # -----------------------------------------------------------------
    #  Main computation
    # -----------------------------------------------------------------

    @torch.no_grad()
    def compute(self, scan, r=0.005, n_hull=10, seed=42,
                hull_batch_size=2, compute_voxel_lc=False, voxel_indices=None, ground_truth=None,
                target_shape=None):
        """
        Compute LC for a single scan.

        Parameters
        ----------
        scan            : (1, C, D, H, W) tensor — the input volume
        r               : cross-polytope radius
        n_hull          : number of hull vertices
        seed            : reproducibility seed
        hull_batch_size : vertices forwarded per sub-batch
        voxel_indices   : list of (d, h, w) or None
        target_shape    : optional (D, H, W).  When given, the scan (and
                          ground_truth, if provided) is centre-cropped /
                          zero-padded to this shape BEFORE any computation.
                          Pass the same shape on every call (e.g. (256,256,256))
                          to guarantee the hull-direction cache holds exactly
                          one entry regardless of the caller's preprocessing.
                          Images are padded with IMAGE_PAD_VALUE, masks with
                          MASK_PAD_VALUE.

        Returns
        -------
        dict with keys:
            total_lc        — int, sum of per-layer global LC
            per_layer_lc    — {name: int}
            compute_time    — float (seconds)
            compute_voxel_lc — boolean, whether to compute voxel-specific LC
            voxel_lc        — list of dicts (only if voxel_indices given)
        """
        was_training = self.model.training
        self.model.eval()

        # Enforce a common input shape up front so the prediction, the
        # ground truth and the hull all live in target_shape space.
        if target_shape is not None:
            scan = center_crop_or_pad_tensor(scan, target_shape,
                                             pad_value=IMAGE_PAD_VALUE)
            if ground_truth is not None:
                ground_truth = center_crop_or_pad(ground_truth, target_shape,
                                                  pad_value=MASK_PAD_VALUE)

        # Populate voxel_indices with the indices of the non-zero voxels in the predicted segmentation
        if compute_voxel_lc and len(voxel_indices) == 0:
            pred_seg = self.model(scan)
            pred_seg = torch.sigmoid(pred_seg)
            pred_seg[pred_seg > 0.5] = 1
            pred_seg[pred_seg <= 0.5] = 0
            pred_seg_np = pred_seg.detach().cpu().numpy()
            pred_seg_np = pred_seg_np.squeeze()
            pred_seg_np = pred_seg_np.astype(np.uint8)

            gt_h, gt_w, gt_d = ground_truth.shape
            pred_seg_np = pred_seg_np[:gt_h, :gt_w, :gt_d]

            # Populate voxel_indices with the indices of all voxels in the predicted segmentation
            voxel_indices = np.where(pred_seg_np == 1)
            voxel_indices = list(zip(voxel_indices[0], voxel_indices[1], voxel_indices[2]))

            # Populate voxel_indices with the indices of true positive(both predicted and ground truth are 1) voxels in the ground truth
            # voxel_indices = np.where((pred_seg_np == 1) & (ground_truth == 1))
            # voxel_indices = list(zip(voxel_indices[0], voxel_indices[1], voxel_indices[2]))

            # Populate voxel_indices with the indices of False Positive (predicted 1 and ground truth 0) voxels in the ground truth
            # voxel_indices = np.where((pred_seg_np == 1) & (ground_truth == 0))
            # voxel_indices = list(zip(voxel_indices[0], voxel_indices[1], voxel_indices[2]))

            # # Populate voxel_indices with the indices of False Negative (predicted 0 and ground truth 1) voxels in the ground truth
            # voxel_indices = np.where((pred_seg_np == 0) & (ground_truth == 1))
            # voxel_indices = list(zip(voxel_indices[0], voxel_indices[1], voxel_indices[2]))

            # # Populate voxel_indices with the indices of True Negative (predicted 0 and ground truth 0) voxels in the ground truth
            # voxel_indices = np.where(pred_seg_np == 0 & ground_truth == 0)
            # voxel_indices_true_negative = list(zip(voxel_indices[0], voxel_indices[1], voxel_indices[2]))


        self._register_hooks()

        # prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        # prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        # torch.backends.cudnn.allow_tf32 = False
        # torch.backends.cuda.matmul.allow_tf32 = False

        

        t0 = time.time()

        # Uses the instance-level direction cache; directions are generated
        # only the first time a given input shape is seen.
        hull = self._build_hull(scan.to(self.device), r=r, n=n_hull, seed=seed)

        # ---- reference vertex (index 0) ----
        self.model(hull[0:1])

        ref_signs = {}
        any_change = {}
        for name in self.layer_names:
            act = self.activation_buffer[name]          # (1, C, D', H', W')
            ref_signs[name] = torch.sign(act).to(torch.int8)
            any_change[name] = torch.zeros(
                act.shape, dtype=torch.bool, device=act.device
            )

        # ---- voxel-specific initialisation ----
        do_voxels = voxel_indices is not None and len(voxel_indices) > 0
        voxel_layer_idx = {}
        voxel_ref = {}
        voxel_change = {}

        if do_voxels:
            for name, _, _, scale in self.layer_info:
                act = self.activation_buffer[name]
                _, C_l, D_l, H_l, W_l = act.shape

                d_idx = torch.tensor(
                    [min(vd // scale, D_l - 1) for vd, _, _ in voxel_indices],
                    dtype=torch.long,
                )
                h_idx = torch.tensor(
                    [min(vh // scale, H_l - 1) for _, vh, _ in voxel_indices],
                    dtype=torch.long,
                )
                w_idx = torch.tensor(
                    [min(vw // scale, W_l - 1) for _, _, vw in voxel_indices],
                    dtype=torch.long,
                )
                voxel_layer_idx[name] = (d_idx, h_idx, w_idx)

                voxel_ref[name] = (
                    torch.sign(act[0, :, d_idx, h_idx, w_idx])
                    .to(torch.int8)
                    .cpu()
                )                                                # (C, n_voxels)
                voxel_change[name] = torch.zeros(
                    C_l, len(voxel_indices), dtype=torch.bool
                )

        # ---- remaining hull vertices ----
        for j in range(1, n_hull, hull_batch_size):
            end_j = min(j + hull_batch_size, n_hull)
            self.model(hull[j:end_j])

            for name, _, _, _ in self.layer_info:
                act = self.activation_buffer[name]
                signs = torch.sign(act).to(torch.int8)

                changed = (signs != ref_signs[name]).any(dim=0, keepdim=True)
                any_change[name] |= changed

                if do_voxels:
                    d_idx, h_idx, w_idx = voxel_layer_idx[name]
                    v_signs = (
                        torch.sign(act[:, :, d_idx, h_idx, w_idx])
                        .to(torch.int8)
                        .cpu()
                    )                                            # (bs, C, n_vox)
                    v_changed = (v_signs != voxel_ref[name].unsqueeze(0)).any(dim=0)
                    voxel_change[name] |= v_changed

        elapsed = time.time() - t0
        self._remove_hooks()

        # torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
        # torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32

        if was_training:
            self.model.train()

        # ---- aggregate global ----
        per_layer = {}
        total = 0
        for name in self.layer_names:
            n_inter = int(any_change[name].sum().item())
            per_layer[name] = n_inter
            total += n_inter

        result = {
            "total_lc": total,
            "per_layer_lc": per_layer,
            "compute_time": elapsed,
        }

        # ---- aggregate voxel-specific ----
        if do_voxels:
            per_layer_counts = {}
            for name in self.layer_names:
                per_layer_counts[name] = voxel_change[name].sum(dim=0).numpy() #dim=0 means sum over the channel dimension

            voxel_results = []
            for vi in range(len(voxel_indices)):
                v_per_layer = {}
                v_total = 0
                for name in self.layer_names:
                    n_inter = int(per_layer_counts[name][vi])
                    v_per_layer[name] = n_inter
                    v_total += n_inter
                voxel_results.append(
                    {
                        "voxel": voxel_indices[vi],
                        "total_lc": v_total, #v_total is the total number of hyperplane crossings for the voxel
                        "per_layer_lc": v_per_layer, #v_per_layer is the number of hyperplane crossings for each layer for the voxel
                    }
                )
            result["voxel_lc"] = voxel_results #voxel_results is a list of dictionaries, each containing the total number of hyperplane crossings for a voxel and the number of hyperplane crossings for each layer for that voxel

        return result


# =====================================================================
#  CSV logging
# =====================================================================

def log_results_to_csv(result, scan_path, layer_names, output_dir,
                       radius, n_hull, model_path):
    """Write one CSV per scan, named after the scan file."""
    os.makedirs(output_dir, exist_ok=True)
    scan_name = '-'.join(scan_path.split('/')[1:])[:-4]#os.path.splitext(os.path.basename(scan_path))[0]
    csv_path = os.path.join(output_dir, f"{scan_name}.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["# scan", scan_path])
        writer.writerow(["# model", model_path])
        writer.writerow(["# radius", radius])
        writer.writerow(["# n_hull", n_hull])
        writer.writerow([])

        header = ["type", "identifier", "total_lc"] + layer_names + ["compute_time_sec"]
        writer.writerow(header)

        row = ["global", "all_neurons", result["total_lc"]]
        for name in layer_names:
            row.append(result["per_layer_lc"][name])
        row.append(f"{result['compute_time']:.2f}")
        writer.writerow(row)

        if "voxel_lc" in result:
            for vr in result["voxel_lc"]:
                d, h, w = vr["voxel"]
                row = ["voxel", f"({d},{h},{w})", vr["total_lc"]]
                for name in layer_names:
                    row.append(vr["per_layer_lc"][name])
                row.append("")
                writer.writerow(row)


# =====================================================================
#  Main
# =====================================================================

def main():
    model = BasicUNet().to(DEVICE)
    state_dict = torch.load(MODEL_WEIGHT_PATH, map_location=DEVICE)#, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    lc = UNetLocalComplexity(
        model,
        device=DEVICE,
        target_layers=TARGET_LAYERS,
        hook_target=HOOK_TARGET,
    )
    print(f"Hook target: {HOOK_TARGET} | "
          f"Hookable layers ({len(lc.layer_names)}): {lc.layer_names}")

    transform = transforms.Compose(
                                    [
                                        VariableSpatialFix(num_of_double_stride_conv=4, padval=0.5),
                                        ToTensor()
                                    ])

    voxel_indices = VOXEL_INDICES if COMPUTE_VOXEL_LC else None

    for i, path in enumerate(SCAN_PATHS):
        print(f"[{i + 1}/{len(SCAN_PATHS)}] {os.path.basename(path)} ...",
              end=" ", flush=True)

        scan = np.load(path).astype(np.float32)
        
        img = scan[:,:,:,0]
        ground_truth = scan[:,:,:,1]

        if SCAN_TYPE[i] == "UTE":
            img = normalise_one_one(img)
        else:
            img = normalise_one_one(normalise_hu(img))

        # Force a common shape so the hull-direction cache holds a single entry.
        # Done AFTER normalisation so the padded voxels keep their exact pad
        # value and do not skew the min/max that normalise_one_one used.
        # The mask is cropped/padded identically so prediction, GT and the saved
        # NIfTI all stay aligned in TARGET_SHAPE space.
        img = center_crop_or_pad(img, TARGET_SHAPE, pad_value=IMAGE_PAD_VALUE)
        ground_truth = center_crop_or_pad(ground_truth, TARGET_SHAPE,
                                          pad_value=MASK_PAD_VALUE)

        scan_tensor = transform(img)
        scan_tensor = scan_tensor.unsqueeze(0).to(DEVICE)

        result = lc.compute(
            scan_tensor,
            r=RADIUS,
            n_hull=N_HULL,
            seed=HULL_SEED,
            hull_batch_size=HULL_BATCH_SIZE,
            compute_voxel_lc=COMPUTE_VOXEL_LC,
            voxel_indices=voxel_indices,
            ground_truth=ground_truth,
        )

        log_results_to_csv(
            result, path, lc.layer_names, OUTPUT_DIR,
            RADIUS, N_HULL, MODEL_WEIGHT_PATH,
        )

        if SAVE_LC_NIFTI:
            save_lc_nifti(result, path, ground_truth.shape, OUTPUT_DIR)

        print(f"done ({result['compute_time']:.1f}s)")

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
