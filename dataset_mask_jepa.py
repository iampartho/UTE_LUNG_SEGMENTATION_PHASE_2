"""
Dataset for the FRESH fixed-256^3 mask-encoder pretraining (Phase 1, step 1).

Design (decided with the target domain in mind):
    * CT masks ONLY (no UTE anywhere in this pipeline) -- the CSV passed in is
      expected to list only COPDgene CT scans.
    * Every mask is fit to a fixed ``input_size`` (default 256^3) -- the exact
      size the encoder will see at Phase 2, since UTE is always 256^3. UTE will
      therefore need NO resize/crop at Phase 2.
    * CT is fit to 256^3 by one of two modes (``fit_mode``):
        - "pad_crop" (default, recommended): keep native 1.25mm voxels; centre-
          crop axes larger than the target and centre-pad axes smaller than it.
          This preserves *physical scale*, so a lung of a given real-world size
          occupies the same number of tokens in CT and in UTE (both 1.25mm).
          Tradeoff: the largest TLC scans get their periphery cropped slightly.
        - "resize": trilinear/nearest resize the whole volume to the target.
          Guarantees the whole lung stays in view but rescales CT voxels (a big
          CT lung becomes smaller per-voxel than an equally large UTE lung).

The numpy files have layout (H, W, D, 2); channel 1 is the lung mask.

Hygiene + domain-generalisation augmentation (added after the Phase-0 shape
diagnostics):
    * clean_mask: keep the major connected component(s) + fill holes. UTE GT masks
      carry a heavy speck/hole artifact (up to ~18 components); this strips it.
      The anatomy check confirmed the CT<->UTE shape gap survives cleaning, so
      this removes noise without touching the real signal.
    * augment (train only): a SYMMETRIC shape augmentation that broadens the
      CT-only shape distribution to cover the UTE region. The Phase-0 analysis
      showed UTE lungs differ from CT mainly on a compactness/inflation axis
      (more elongated, lower bbox extent), and that this axis is the strongest
      driver of the mask-only energy. We therefore augment exactly that axis:
        - anisotropic per-axis scaling  -> elongation
        - elastic deformation           -> extent
      Applied to the single returned mask (which feeds both encoders), so the
      encoder learns elongated/low-extent lungs are normal -> a correct UTE mask
      stops being scored as high-energy purely for being UTE-shaped.

The V-JEPA token masking (dropping tokens) is generated on the fly during
training (``model_vjepa.generate_masks``); this dataset only returns the volume.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage.transform import resize as sk_resize
from scipy.ndimage import (
    gaussian_filter,
    label as ndi_label,
    binary_fill_holes,
    zoom as ndi_zoom,
    map_coordinates,
)


def _normalise_zero_one(image):
    image = image.astype(np.float32)
    minimum = float(np.min(image))
    maximum = float(np.max(image))
    if maximum > minimum:
        return (image - minimum) / (maximum - minimum)
    return image * 0.0


# ---------------------------------------------------------------------------
# Mask cleaning (hygiene) -- remove the speck/hole artifact that is heavy in UTE
# masks (connected components up to ~18) but should never be part of a lung shape.
# Verified anatomy-vs-artifact: the CT<->UTE shape gap (elongation/extent)
# survives this cleaning, so cleaning only strips noise, not the real signal.
# ---------------------------------------------------------------------------
def _clean_binary(mask, keep_frac=0.10, fill_2d_axial=True):
    """Keep connected component(s) >= keep_frac * largest, then fill holes."""
    m = mask > 0.5
    if not m.any():
        return mask.astype(np.float32)
    lbl, n = ndi_label(m, structure=np.ones((3, 3, 3))) # ndi_label returns the labels and the number of connected components        
    counts = np.bincount(lbl.ravel()) # bincount returns the number of occurrences of each value in the array for instance counts[0] is the number of occurrences of label 0 and counts[1] is the number of occurrences of label 1 and lbl.ravel() returns the flattened array
    counts[0] = 0  # drop background
    sizes = counts[1:]  # label id i -> sizes[i-1]
    largest = sizes.max()
    keep_ids = np.where(sizes >= keep_frac * largest)[0] + 1
    if keep_ids.size == 0:
        keep_ids = np.array([int(np.argmax(sizes)) + 1])
    out = np.isin(lbl, keep_ids)
    out = binary_fill_holes(out)
    if fill_2d_axial:  # also fill in-plane holes (e.g. vessel cross-sections)
        for k in range(out.shape[2]):
            if out[:, :, k].any():
                out[:, :, k] = binary_fill_holes(out[:, :, k])
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# SYMMETRIC shape augmentation. Applied to the single returned mask, which feeds
# BOTH the context encoder and the EMA target encoder -> it broadens what the
# encoder regards as a "plausible lung" to cover the UTE region of shape space.
# Targets the two descriptors that separate the modalities:
#   * anisotropic scaling  -> moves ELONGATION (per-axis stretch/squash)
#   * elastic deformation  -> moves EXTENT (bends the lung so its axis-aligned
#                             bounding box grows relative to its volume)
# Isotropic scaling is deliberately NOT used: it changes neither descriptor.
# ---------------------------------------------------------------------------
def _aniso_zoom(mask, rng, scale_range):
    """Independent per-axis rescale (order=1, re-binarised). Changes elongation."""
    f = rng.uniform(scale_range[0], scale_range[1], size=3)
    z = ndi_zoom(mask, (float(f[0]), float(f[1]), float(f[2])), order=1)
    return (z > 0.5).astype(np.float32)


def _elastic(mask, rng, alpha, ctrl):
    """Smooth random displacement field (coarse ctrl^3 grid, upsampled).

    ``alpha`` is the displacement magnitude in voxels; small ``ctrl`` => smoother,
    more global warps (a few large bends rather than high-frequency jitter).
    """
    shape = mask.shape
    # --- Previous per-axis field construction (uncomment to revert) ----------
    # indices = []
    # for ax in range(3):
    #     g = rng.standard_normal((ctrl, ctrl, ctrl)).astype(np.float32)
    #     zoom_facs = tuple(shape[i] / ctrl for i in range(3))
    #     field = ndi_zoom(g, zoom_facs, order=1)[:shape[0], :shape[1], :shape[2]]
    #     field = field * alpha
    #     base_shape = [1, 1, 1]
    #     base_shape[ax] = shape[ax]
    #     base = np.arange(shape[ax], dtype=np.float32).reshape(base_shape)
    #     coord = np.clip(base + field, 0, shape[ax] - 1)
    #     indices.append(coord)
    # --- Optimised: one batched zoom for all three fields, then build the
    # sampling coordinates in-place. Factor 1.0 on the stacked axis keeps the
    # three displacement fields independent, so each is identical to upsampling
    # it on its own -- bit-for-bit the same warp with far fewer scipy calls and
    # full-volume temporaries.
    g = rng.standard_normal((3, ctrl, ctrl, ctrl)).astype(np.float32)
    zoom_facs = (1.0, shape[0] / ctrl, shape[1] / ctrl, shape[2] / ctrl)
    fields = ndi_zoom(g, zoom_facs, order=1)[:, :shape[0], :shape[1], :shape[2]]
    fields *= alpha
    indices = []
    for ax in range(3):
        base_shape = [1, 1, 1]
        base_shape[ax] = shape[ax]
        base = np.arange(shape[ax], dtype=np.float32).reshape(base_shape)
        coord = fields[ax]
        coord += base
        np.clip(coord, 0, shape[ax] - 1, out=coord)
        indices.append(coord)
    out = map_coordinates(mask, indices, order=1, mode="constant", cval=0.0)
    return (out > 0.5).astype(np.float32)


def _center_pad_crop(vol, target, pad_value=0.0):
    """Fit ``vol`` to ``target`` shape via centre-crop (larger axes) then
    centre-pad (smaller axes). Preserves native voxel resolution."""
    # Centre-crop any axis that is larger than the target.
    slices = []
    for s, t in zip(vol.shape, target):
        if s > t:
            start = (s - t) // 2
            slices.append(slice(start, start + t))
        else:
            slices.append(slice(0, s))
    vol = vol[tuple(slices)]
    # Centre-pad any axis that is smaller than the target.
    pads = []
    for s, t in zip(vol.shape, target):
        if s < t:
            before = (t - s) // 2
            pads.append((before, t - s - before))
        else:
            pads.append((0, 0))
    return np.pad(vol, pads, mode="constant", constant_values=pad_value)


class MaskJEPADataset(Dataset):
    """Returns (mask_volume, filename) per sample, fit to a fixed ``input_size``.

    Args:
        csv_file: CSV with a single 'filepaths' column (CT scans only).
        input_size: fixed (H, W, D) every mask is fit to. Each dim must be
            divisible by the ViT patch size (default 256 / 32 = 8 tokens/axis).
        fit_mode: "pad_crop" (preserve 1.25mm scale, centre pad/crop) or "resize"
            (rescale whole volume to input_size).
        gaussian_sigma: std-dev (voxels) of the Gaussian softening applied to the
            binary mask, so the encoder's input distribution resembles the soft
            sigmoid predictions seen at Phase 2. Set 0.0 to disable.
        pad_value: value used for the (background) padding; 0.0 for masks.
        clean_mask: if True, keep major connected component(s) + fill holes before
            anything else (hygiene; recommended for both train and val).
        clean_keep_frac: keep components >= this fraction of the largest one.
        augment: if True, apply the SYMMETRIC shape augmentation (anisotropic
            scale + elastic). Set True for training, False for validation.
        scale_prob / scale_range: probability and per-axis factor range for the
            anisotropic rescale (moves elongation). Use a non-isotropic range.
        elastic_prob / elastic_alpha / elastic_ctrl: probability, displacement
            magnitude (voxels) and control-grid resolution for the elastic warp
            (moves extent). Smaller ctrl => smoother, more global bends.
        path_replace: optional (old, new) substring pair to remap CSV paths to the
            current machine (e.g. ("/Shared/", "/Volumes/")). None = use verbatim.
        transform: optional callable applied to the final (H, W, D) numpy volume;
            if None, a leading channel dim is added and it is returned as a tensor.
    """

    def __init__(
        self,
        csv_file,
        input_size=(256, 256, 256),
        fit_mode="pad_crop",
        gaussian_sigma=1.0,
        pad_value=0.0,
        clean_mask=True,
        clean_keep_frac=0.10,
        augment=False,
        scale_prob=0.8,
        scale_range=(0.80, 1.25),
        elastic_prob=0.5,
        elastic_alpha=12.0,
        elastic_ctrl=8,
        path_replace=None,
        transform=None,
    ):
        assert fit_mode in ("pad_crop", "resize"), fit_mode
        df = pd.read_csv(csv_file)
        self.arr_files = df["filepaths"].values
        self.input_size = tuple(int(s) for s in input_size)
        self.fit_mode = fit_mode
        self.gaussian_sigma = float(gaussian_sigma)
        self.pad_value = float(pad_value)
        self.clean_mask = bool(clean_mask)
        self.clean_keep_frac = float(clean_keep_frac)
        self.augment = bool(augment)
        self.scale_prob = float(scale_prob)
        self.scale_range = tuple(float(s) for s in scale_range)
        self.elastic_prob = float(elastic_prob)
        self.elastic_alpha = float(elastic_alpha)
        self.elastic_ctrl = int(elastic_ctrl)
        self.path_replace = path_replace
        self.transform = transform

    def __len__(self):
        return len(self.arr_files)

    def _resolve(self, p):
        if self.path_replace is not None:
            p = p.replace(self.path_replace[0], self.path_replace[1])
        return p

    def _fit(self, mask):
        if tuple(mask.shape) == self.input_size:
            return mask
        if self.fit_mode == "pad_crop":
            return _center_pad_crop(mask, self.input_size, pad_value=self.pad_value)
        # order=0 (nearest) keeps the mask binary; no anti-alias / value rescaling.
        return sk_resize(mask, self.input_size, order=0,
                         preserve_range=True, anti_aliasing=False)

    def __getitem__(self, index):
        filename = self.arr_files[index]
        path = self._resolve(filename)

        arr = np.load(path)
        mask = arr[:, :, :, 1]
        mask[mask > 0] = 1
        mask = _normalise_zero_one(mask).astype(np.float32)

        # Hygiene first: strip specks / fill holes (verified to be artifact, not
        # the real CT<->UTE shape signal).
        if self.clean_mask:
            mask = _clean_binary(mask, keep_frac=self.clean_keep_frac)

        # Symmetric shape augmentation. The same returned mask feeds BOTH the
        # context and EMA-target encoders, so this is symmetric by construction.
        #   * anisotropic scale: on the NATIVE mask, before the fit, so it
        #     interacts with physical extent (and re-fit re-centres the result).
        #   * elastic warp: AFTER the fit, on the fixed input_size grid, so its
        #     cost/voxel-scale are bounded regardless of the native volume size.
        rng = np.random.default_rng() if self.augment else None
        if self.augment and rng.random() < self.scale_prob:
            mask = _aniso_zoom(mask, rng, self.scale_range)

        mask = self._fit(mask).astype(np.float32)

        if self.augment and rng.random() < self.elastic_prob:
            mask = _elastic(mask, rng, self.elastic_alpha, self.elastic_ctrl)

        # Soften so the encoder sees a distribution like the soft Phase-2 predictions.
        if self.gaussian_sigma > 0.0:
            mask = gaussian_filter(mask, sigma=self.gaussian_sigma).astype(np.float32)
            mask = np.clip(mask, 0.0, 1.0)

        if self.transform:
            mask = self.transform(mask)
        else:
            mask = torch.from_numpy(np.expand_dims(mask, axis=0))

        return mask, filename


class MaskToTensor(object):
    """Adds a leading channel dim to a single (H, W, D) mask volume."""

    def __call__(self, mask):
        return torch.from_numpy(np.expand_dims(mask, axis=0))
