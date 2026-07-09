"""
visualize_gin_vs_fda.py
=======================

A *visualisation-only* script (no training, no model inference) that lets you
eyeball two different appearance-randomisation strategies on CT scans:

  1. GIN  -- your existing blind, single-source domain-generalisation
            augmentation (``CausalityAugmentation3D`` / ``GIN3D``), run with the
            same 70%-roughness-enforced behaviour you use in training.  GIN
            randomises the local intensity mapping with random conv kernels
            while preserving spatial layout.

  2. Target-aware FDA (Fourier Domain Adaptation, Yang & Soatto 2020) --
            keep the CT's Fourier *phase* (which carries the anatomy / edges)
            but overwrite the CT's *low-frequency amplitude* with the
            low-frequency amplitude of a real UTE MRI.  The result is "CT
            anatomy wearing the MRI's low-frequency look".  Unlike GIN this is
            NOT single-source: it injects knowledge of the target (MRI)
            spectrum.

The script takes 5 CT scans and 5 MRI scans, produces:
    * 5 GIN-augmented CT volumes,
    * 5 FDA CT volumes (CT_i paired with MRI_i),
and writes everything (plus the originals) to ``.nii.gz`` via SimpleITK so you
can compare them side-by-side in Slicer and decide whether the FDA "look" is
worth pursuing.

Nothing here trains or evaluates a network - it only generates images.
"""

import os

import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk

# Reuse the exact normalisation conventions used across the codebase.
from eval_step_1 import normalise_hu, normalise_one_one
# Your GIN augmentor (70%-roughness behaviour is the default in its forward()).
from gin_with_log_capability import CausalityAugmentation3D


# =====================================================================
#  CONFIG  -- edit then run.  (No argparse by design.)
# =====================================================================

CT_CSV = "./ids/only_copd_1.25mm.csv"     # source domain (CT)
MRI_CSV = "./ids/only_ute_1.25mm.csv"     # target domain (UTE MRI)

NUM_PAIRS = 5                              # how many CT/MRI scans to use

# FDA "beta": half-width of the centred low-frequency amplitude cube that gets
# copied from the MRI onto the CT, expressed as a FRACTION of each spatial
# dimension.  Small beta -> only the very lowest frequencies (overall
# brightness / smooth shading) are transferred; larger beta -> more of the
# MRI's mid-frequency texture comes across (and the CT anatomy starts to
# distort).  The FDA paper uses small values; 0.01-0.09 is a sensible sweep.
# This same beta is reused for the standalone amplitude-randomisation variant.
FDA_BETA = 0.001

# Standalone (blind) Fourier amplitude randomisation strength, in [0, 1].
# Instead of borrowing the MRI amplitude (target-aware FDA), we keep the CT
# phase and multiply the CT's OWN low-frequency amplitude by random factors
# drawn uniformly from [1 - strength, 1 + strength].  This is the single-source
# analogue of FDA: it perturbs the "look" without any knowledge of the target.
#   0.0 -> no change; 1.0 -> amplitudes scaled anywhere in [0, 2x].
FDA_RANDOM_STRENGTH = 0.8

# Re-stretch each output back to [-1, 1] before saving (keeps the .nii.gz
# windows comparable in Slicer). Set False to inspect raw amplitudes.
RENORMALISE_OUTPUT = True

OUTPUT_DIR = "./visualization/gin_vs_fda"
SPACING = (1.25, 1.25, 1.25)

# Seed so the random CT/MRI selection and GIN kernels are reproducible.
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
#  Helpers
# =====================================================================

def save_nifty(arr_np, spacing, output_path):
    """Write a 3D numpy array to a .nii.gz with the given voxel spacing."""
    img = sitk.GetImageFromArray(arr_np.astype(np.float32))
    img.SetSpacing(spacing)
    sitk.WriteImage(img, output_path)
    print(f"  saved {output_path}")


def load_ct(path):
    """Load a CT scan, take the image channel, apply HU clamp + [-1,1] norm."""
    arr = np.load(path)
    img = arr[:, :, :, 0]
    gt = arr[:, :, :, 1]
    gt[gt > 0] = 1
    return normalise_one_one(normalise_hu(img)).astype(np.float32), gt.astype(np.uint8)


def load_mri(path):
    """Load a UTE MRI scan, take the image channel, apply [-1,1] norm (no HU)."""
    arr = np.load(path)
    img = arr[:, :, :, 0]
    return normalise_one_one(img).astype(np.float32)


def resize_to(volume, target_shape):
    """Make ``volume`` the same 3D size as ``target_shape``, without resampling.

    Why this exists
    ---------------
    ``fda_ct_to_mri`` needs CT and MRI as two numpy arrays of *identical* shape
    so their FFTs line up voxel-for-voxel.  In practice the MRI is often a
    different size than the CT, so we adjust the MRI to ``ct.shape`` before FDA.

    What we do (crop or zero-pad only — no trilinear resize)
    ---------------------------------------------------------
    We fix each axis (depth, height, width) independently:

      * MRI **longer** than CT on that axis → **crop** the extra voxels from the
        centre (keep the middle, discard equal amounts from both ends).
      * MRI **shorter** than CT on that axis → **pad** with zeros on both sides
        so the MRI sits in the middle and the outer voxels are 0.
      * Same length → leave that axis alone.

    Real MRI voxel values are never blended or smoothed; only padding or trimming.

    1D intuition (target length = 5)
    --------------------------------
      Too long:  [A B C D E F G]  --crop centre-->  [B C D E F]
      Too short: [A B C]          --pad zeros-->    [0 A B C 0]
      Exact:     [A B C D E]      --unchanged-->

    Parameters
    ----------
    volume : (D, H, W) MRI (typically).
    target_shape : (D, H, W) desired size, usually ``ct.shape``.

    Returns
    -------
    float32 array with shape ``target_shape``.
    """
    result = volume

    # Process depth (axis 0), then height (axis 1), then width (axis 2).
    for axis_index in range(result.ndim):
        current_len = result.shape[axis_index]
        target_len = target_shape[axis_index]

        if current_len == target_len:
            continue

        if current_len > target_len:
            # MRI is too long on this axis: keep a contiguous middle chunk.
            extra_voxels = current_len - target_len
            first_keep_index = extra_voxels // 2
            last_keep_index = first_keep_index + target_len

            slicer = [slice(None)] * result.ndim
            slicer[axis_index] = slice(first_keep_index, last_keep_index)
            result = result[tuple(slicer)]
        else:
            # MRI is too short: add zero voxels on left and right of this axis.
            voxels_to_add = target_len - current_len
            pad_left = voxels_to_add // 2
            pad_right = voxels_to_add - pad_left

            # np.pad format: one (before, after) pair per dimension.
            padding = [(0, 0)] * result.ndim
            padding[axis_index] = (pad_left, pad_right)
            result = np.pad(result, padding, mode="constant", constant_values=0)

    assert result.shape == tuple(target_shape)
    return result.astype(np.float32)


def gin_augment(ct_volume, augmentor):
    """Run one GIN augmentation pass on a CT volume.

    The augmentor's forward() already implements the 70%-roughness-enforced
    kernel sampling (``is_high_roughness = rand > 0.3``), so we just call it.
    """
    x = torch.from_numpy(ct_volume)[None, None].float().to(DEVICE)  # (1,1,D,H,W)
    augmentor.eval()
    with torch.no_grad():
        out = augmentor(x)
    return out.squeeze().detach().cpu().numpy().astype(np.float32)


def fda_ct_to_mri(ct_volume, mri_volume, beta):
    """Target-aware Fourier Domain Adaptation: CT phase + MRI low-freq amplitude.

    Algorithm (Yang & Soatto, 2020, extended to 3D):
      1. FFT both volumes.
      2. Keep the CT's phase and amplitude; keep only the MRI's amplitude.
      3. Replace the CT's *low-frequency* amplitude (a centred cube of
         half-width beta * dim) with the MRI's low-frequency amplitude.
      4. Recombine the modified amplitude with the **CT phase** and inverse-FFT.

    Because phase carries structure/edges (Oppenheim & Lim, 1981), the anatomy
    stays CT while the smooth low-frequency "look" becomes MRI-like.

    Both volumes must already share the same shape.
    """
    assert ct_volume.shape == mri_volume.shape, "FDA needs matching shapes"

    fft_ct = np.fft.fftn(ct_volume)
    fft_mri = np.fft.fftn(mri_volume)

    amp_ct, pha_ct = np.abs(fft_ct), np.angle(fft_ct)
    amp_mri = np.abs(fft_mri)

    # Shift DC to the centre so the "low-frequency" region is a centred cube.
    amp_ct_sh = np.fft.fftshift(amp_ct)
    amp_mri_sh = np.fft.fftshift(amp_mri)

    d, h, w = ct_volume.shape
    cz, cy, cx = d // 2, h // 2, w // 2
    # Half-widths of the low-frequency cube (>= 1 voxel so something is swapped).
    bz = max(1, int(beta * d))
    by = max(1, int(beta * h))
    bx = max(1, int(beta * w))

    # Overwrite the CT's low-frequency amplitude with the MRI's.
    amp_ct_sh[cz - bz:cz + bz + 1,
              cy - by:cy + by + 1,
              cx - bx:cx + bx + 1] = amp_mri_sh[cz - bz:cz + bz + 1,
                                                cy - by:cy + by + 1,
                                                cx - bx:cx + bx + 1]

    # Un-shift, recombine with the CT phase, invert.
    amp_ct_new = np.fft.ifftshift(amp_ct_sh)
    fft_new = amp_ct_new * np.exp(1j * pha_ct)
    out = np.real(np.fft.ifftn(fft_new)).astype(np.float32)
    return out


def fda_mri_to_ct(mri_volume, ct_volume, beta):
    """Reverse FDA: MRI phase + CT low-frequency amplitude.

    This is the exact mirror of ``fda_ct_to_mri``:
      - ``fda_ct_to_mri``  keeps CT phase,  swaps in MRI amplitude  -> "CT wearing MRI look"
      - ``fda_mri_to_ct``  keeps MRI phase, swaps in CT amplitude   -> "MRI wearing CT look"

    Why this is useful
    ------------------
    Saving this lets you check whether the CT amplitude genuinely
    carries the "CT look" (bright bones, dark air).  If the output
    looks CT-like in brightness/contrast but has MRI spatial structure,
    the FDA amplitude-swap is working as intended.

    Algorithm (same steps as ``fda_ct_to_mri``, roles swapped)
    -----------------------------------------------------------
      1. FFT both volumes.
      2. Shift DC to centre so the low-frequency region is a centred cube.
      3. Overwrite the **MRI's** low-frequency amplitude cube with the **CT's**.
      4. Un-shift, recombine the modified amplitude with the **MRI phase**,
         and inverse-FFT back to image space.

    Both volumes must already share the same shape (call ``resize_to`` first).
    """
    assert mri_volume.shape == ct_volume.shape, "FDA needs matching shapes"

    fft_mri = np.fft.fftn(mri_volume)
    fft_ct = np.fft.fftn(ct_volume)

    # Split each FFT into amplitude (how strong each frequency is) and
    # phase (where the edges / structures are).
    amp_mri, pha_mri = np.abs(fft_mri), np.angle(fft_mri)
    amp_ct = np.abs(fft_ct)

    # Shift so the low-frequency (DC) component sits at the array centre.
    amp_mri_sh = np.fft.fftshift(amp_mri)
    amp_ct_sh  = np.fft.fftshift(amp_ct)

    d, h, w = mri_volume.shape
    cz, cy, cx = d // 2, h // 2, w // 2
    # Half-widths of the low-frequency cube (at least 1 voxel).
    bz = max(1, int(beta * d))
    by = max(1, int(beta * h))
    bx = max(1, int(beta * w))

    # Overwrite the MRI's low-frequency amplitude cube with the CT's.
    amp_mri_sh[cz - bz:cz + bz + 1,
               cy - by:cy + by + 1,
               cx - bx:cx + bx + 1] = amp_ct_sh[cz - bz:cz + bz + 1,
                                                 cy - by:cy + by + 1,
                                                 cx - bx:cx + bx + 1]

    # Un-shift, recombine with the MRI phase, invert back to image space.
    amp_mri_new = np.fft.ifftshift(amp_mri_sh)
    fft_new = amp_mri_new * np.exp(1j * pha_mri)
    out = np.real(np.fft.ifftn(fft_new)).astype(np.float32)
    return out


def fda_amplitude_randomize(ct_volume, beta, strength):
    """Standalone (blind) Fourier amplitude randomisation.

    The single-source counterpart of ``fda_ct_to_mri``: keep the CT's phase
    (anatomy / edges) but randomly perturb the CT's OWN low-frequency amplitude
    rather than copying it from a target MRI.  No target domain is used, so
    this stays within the single-source DG setting.

    Algorithm:
      1. FFT the CT; split into amplitude and phase.
      2. In the centred low-frequency cube (half-width beta * dim), multiply the
         amplitude by random factors ~ Uniform[1 - strength, 1 + strength]
         (clamped at 0 so amplitudes stay non-negative).
      3. Recombine the perturbed amplitude with the **CT phase** and inverse-FFT.

    Parameters
    ----------
    ct_volume : np.ndarray (D, H, W).
    beta      : fraction of each dim defining the low-frequency cube to perturb.
    strength  : in [0, 1]; magnitude of the multiplicative amplitude jitter.
    """
    fft_ct = np.fft.fftn(ct_volume)
    amp_ct, pha_ct = np.abs(fft_ct), np.angle(fft_ct)

    amp_ct_sh = np.fft.fftshift(amp_ct)

    d, h, w = ct_volume.shape
    cz, cy, cx = d // 2, h // 2, w // 2
    bz = max(1, int(beta * d))
    by = max(1, int(beta * h))
    bx = max(1, int(beta * w))

    cube = (slice(cz - bz, cz + bz + 1),
            slice(cy - by, cy + by + 1),
            slice(cx - bx, cx + bx + 1))

    # Random multiplicative jitter on the low-frequency amplitudes.
    rand_scale = np.random.uniform(
        low=max(0.0, 1.0 - strength),
        high=1.0 + strength,
        size=amp_ct_sh[cube].shape,
    ).astype(np.float32)
    amp_ct_sh[cube] = np.clip(amp_ct_sh[cube] * rand_scale, 0.0, None)

    amp_ct_new = np.fft.ifftshift(amp_ct_sh)
    fft_new = amp_ct_new * np.exp(1j * pha_ct)
    out = np.real(np.fft.ifftn(fft_new)).astype(np.float32)
    return out


def maybe_renormalise(volume):
    return normalise_one_one(volume) if RENORMALISE_OUTPUT else volume


# =====================================================================
#  Main
# =====================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"[config] device={DEVICE}  num_pairs={NUM_PAIRS}  fda_beta={FDA_BETA}")
    print(f"[config] writing outputs to {OUTPUT_DIR}")

    ct_files = pd.read_csv(CT_CSV)["filepaths"].values
    mri_files = pd.read_csv(MRI_CSV)["filepaths"].values

    # Randomly pick NUM_PAIRS of each (reproducible via SEED).
    ct_sel = np.random.choice(ct_files, size=NUM_PAIRS, replace=False)
    mri_sel = np.random.choice(mri_files, size=NUM_PAIRS, replace=False)

    # One GIN augmentor instance, reused for every CT (single channel in/out).
    augmentor = CausalityAugmentation3D(in_channels=1).to(DEVICE)

    for i in range(NUM_PAIRS):
        print(f"\n[pair {i}] CT={ct_sel[i]}")
        print(f"[pair {i}] MRI={mri_sel[i]}")

        ct, gt = load_ct(ct_sel[i])
        mri = load_mri(mri_sel[i])

        # ---- 1. GIN-augmented CT (single-source, blind) ----
        gin = maybe_renormalise(gin_augment(ct, augmentor))

        # ---- 2. Target-aware FDA in both directions (CT <-> MRI amplitude swap) ----
        # FDA requires both arrays to be the same shape; bring MRI -> CT shape first.
        mri_resized = resize_to(mri, ct.shape)

        # CT phase + MRI amplitude  -> "CT anatomy wearing the MRI's low-freq look"
        fda = maybe_renormalise(fda_ct_to_mri(ct, mri_resized, FDA_BETA))

        # MRI phase + CT amplitude  -> "MRI anatomy wearing the CT's low-freq look"
        mri_fda = maybe_renormalise(fda_mri_to_ct(mri_resized, ct, FDA_BETA))

        # ---- 3. Standalone FDA / blind amplitude randomisation (no MRI) ----
        fda_rand = maybe_renormalise(
            fda_amplitude_randomize(ct, FDA_BETA, FDA_RANDOM_STRENGTH)
        )

        # ---- save everything for side-by-side viewing in Slicer ----
        tag = f"pair{i}"
        save_nifty(ct, SPACING, f"{OUTPUT_DIR}/{tag}_ct_original.nii.gz")
        save_nifty(gt, SPACING, f"{OUTPUT_DIR}/{tag}_gt_original.nii.gz")
        save_nifty(gin, SPACING, f"{OUTPUT_DIR}/{tag}_ct_gin.nii.gz")
        save_nifty(fda, SPACING, f"{OUTPUT_DIR}/{tag}_ct_fda.nii.gz")
        save_nifty(fda_rand, SPACING,
                   f"{OUTPUT_DIR}/{tag}_ct_fda_random.nii.gz")
        save_nifty(mri, SPACING, f"{OUTPUT_DIR}/{tag}_mri_original.nii.gz")
        save_nifty(mri_resized, SPACING,
                   f"{OUTPUT_DIR}/{tag}_mri_resized_to_ct.nii.gz")
        save_nifty(mri_fda, SPACING, f"{OUTPUT_DIR}/{tag}_mri_fda.nii.gz")

    print("\nDone. Compare in Slicer:")
    print("  *_ct_original  vs  *_ct_gin         -> what GIN does (blind randomisation)")
    print("  *_ct_original  vs  *_ct_fda         -> target-aware FDA (CT phase + MRI amplitude)")
    print("  *_ct_original  vs  *_ct_fda_random  -> standalone FDA (blind amplitude jitter)")
    print("  *_mri_original vs  *_mri_fda        -> reverse FDA (MRI phase + CT amplitude)")
    print("  *_ct_fda       vs  *_mri_fda        -> are the two FDA directions symmetric?")


if __name__ == "__main__":
    main()
