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

Hookable layers (pre-LeakyReLU = InstanceNorm output)
------------------------------------------------------
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
import numpy as np
import pandas as pd
import os
import csv
import time
from torchvision import transforms
from basic_unet_disentagled import BasicUNet

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


# =====================================================================
#  Layer definitions:  (display_name, model_attribute, spatial_scale_divisor)
# =====================================================================

ALL_LAYER_INFO = [
    ("enc_0_0", "norm_0_0", 1),
    ("enc_0_1", "norm_0_1", 1),
    ("enc_1_0", "norm_1_0", 2),
    ("enc_1_1", "norm_1_1", 2),
    ("enc_2_0", "norm_2_0", 4),
    ("enc_2_1", "norm_2_1", 4),
    ("enc_3_0", "norm_3_0", 8),
    ("enc_3_1", "norm_3_1", 8),
    ("dec_4_0", "norm_up_4_0", 4),
    ("dec_4_1", "norm_up_4_1", 4),
    ("dec_3_0", "norm_up_3_0", 2),
    ("dec_3_1", "norm_up_3_1", 2),
    ("dec_2_0", "norm_up_2_0", 1),
    ("dec_2_1", "norm_up_2_1", 1),
]


# =====================================================================
#  Hull sampling for 3D volumes
# =====================================================================

@torch.no_grad()
def get_ortho_hull_3d(x, r=0.005, n=10, seed=42):
    """
    Sample cross-polytopal hull vertices around a single 3D volume.

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

    def __init__(self, model, device="cuda", target_layers="all"):
        self.model = model
        self.device = device
        self.hooks = []
        self.activation_buffer = {}

        if target_layers == "all":
            self.layer_info = ALL_LAYER_INFO[:]
        else:
            self.layer_info = [
                (n, a, s) for n, a, s in ALL_LAYER_INFO if n in target_layers
            ]
        self.layer_names = [name for name, _, _ in self.layer_info]

    # -----------------------------------------------------------------
    #  Hook management
    # -----------------------------------------------------------------

    def _register_hooks(self):
        """Attach forward hooks on InstanceNorm outputs (= pre-LeakyReLU)."""
        for name, attr, _ in self.layer_info:
            module = getattr(self.model, attr) # for example: module = self.model.norm_0_0

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
    #  Main computation
    # -----------------------------------------------------------------

    @torch.no_grad()
    def compute(self, scan, r=0.005, n_hull=10, seed=42,
                hull_batch_size=2, compute_voxel_lc=False, voxel_indices=None, ground_truth=None):
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

        t0 = time.time()

        hull = get_ortho_hull_3d(scan.to(self.device), r=r, n=n_hull, seed=seed)

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
            for name, _, scale in self.layer_info:
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

            for name, _, _ in self.layer_info:
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

    lc = UNetLocalComplexity(model, device=DEVICE, target_layers=TARGET_LAYERS)
    print(f"Hookable layers ({len(lc.layer_names)}): {lc.layer_names}")

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
