import torch
import pandas as pd
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch import nn
from torch.utils.data import DataLoader
import math
from scipy.ndimage import (
    label as ndi_label,
    binary_fill_holes,
    zoom as ndi_zoom,
    map_coordinates,
)




def normalise_hu(image, hu_range=[-1024.0,200.0]):
    assert (hu_range[0] < hu_range[1])
    return np.clip(image, hu_range[0], hu_range[1]).astype(np.float32)

def normalise_zero_one(image):
    image = image.astype(np.float32)
    minimum = np.min(image)
    maximum = np.max(image)
    if maximum > minimum:
        ret = (image - minimum) / (maximum - minimum)
    else:
        ret = image * 0.
    return ret

def normalise_one_one(image):
    ret = normalise_zero_one(image)
    ret *= 2.
    ret -= 1.
    return ret

# ---------------------------------------------------------------------------
# Mask cleaning (hygiene) -- ported verbatim from dataset_mask_jepa.py.
# Keeps the major connected component(s) (>= keep_frac * largest) and fills
# holes. Removes the speck/hole artifact (heavy in UTE GT masks) without
# touching the real lung shape. Applied to BOTH train and test masks.
# ---------------------------------------------------------------------------
def _clean_binary(mask, keep_frac=0.10, fill_2d_axial=True):
    """Keep connected component(s) >= keep_frac * largest, then fill holes."""
    m = mask > 0.5
    if not m.any():
        return mask.astype(np.float32)
    lbl, n = ndi_label(m, structure=np.ones((3, 3, 3)))
    counts = np.bincount(lbl.ravel())
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
# PAIRED geometric augmentation (image + GT mask together). Adapted from the
# mask-only versions in dataset_mask_jepa.py: here the random parameters are
# drawn ONCE and applied identically to the image and the mask so they stay
# pixel-aligned. The image uses linear interpolation and an air-valued fill
# (-1, the background after normalise_one_one); the mask uses linear interp +
# re-binarise and a 0 fill.
# ---------------------------------------------------------------------------
class RandomAnisoScale(object):
    """Independent per-axis rescale (changes lung elongation).

    Applied BEFORE VariableSpatialFix so the change in physical extent is
    re-padded to a valid network input size afterwards. The same scale factors
    are used for the image (order=1) and the mask (order=1 + re-binarise),
    preserving alignment.
    """

    def __init__(self, prob=0.5, scale_range=(0.80, 1.25)):
        self.prob = float(prob)
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))

    def __call__(self, sample):
        assert len(sample) == 2, "Expected an [image, mask] pair"
        image, mask = sample[0], sample[1]
        if np.random.rand() < self.prob:
            f = np.random.uniform(self.scale_range[0], self.scale_range[1], size=3)
            factors = (float(f[0]), float(f[1]), float(f[2]))
            # `image` is already float32 (normalise_one_one), and ndi_zoom keeps
            # the input dtype, so the previous `.astype(np.float32)` only forced
            # an extra full-volume copy -- drop it.
            image = ndi_zoom(image, factors, order=1)
            mask = ndi_zoom(mask, factors, order=1)
            mask = (mask > 0.5).astype(np.float32)
        return [image, mask]


class RandomElastic(object):
    """Smooth random displacement-field warp (changes lung extent).

    Applied AFTER VariableSpatialFix on the fixed, padded grid so its
    voxel-scale cost is bounded. A single displacement field is built and used
    to resample BOTH the image (linear, air fill) and the mask (linear +
    re-binarise, 0 fill), keeping them aligned.

    ``alpha`` is the displacement magnitude in voxels; smaller ``ctrl`` gives
    smoother, more global bends.
    """

    def __init__(self, prob=0.5, alpha=12.0, ctrl=8, image_cval=-1.0, mask_cval=0.0):
        self.prob = float(prob)
        self.alpha = float(alpha)
        self.ctrl = int(ctrl)
        self.image_cval = float(image_cval)
        self.mask_cval = float(mask_cval)

    def __call__(self, sample):
        assert len(sample) == 2, "Expected an [image, mask] pair"
        image, mask = sample[0], sample[1]
        if np.random.rand() < self.prob:
            shape = image.shape
            # Fresh generator per call: with num_workers > 0 each forked worker
            # would otherwise share a persistent generator's state and emit
            # identical fields. default_rng() draws fresh OS entropy each call,
            # so the fields stay diverse across workers (the allocation is
            # negligible next to the zoom / map_coordinates work below).
            rng = np.random.default_rng()
            # Draw all three control-point grids at once and upsample them in a
            # SINGLE zoom call. Factor 1.0 on the stacked axis keeps the three
            # displacement fields independent, so each field is identical to
            # upsampling it on its own -- we just trade three scipy calls (and
            # their full-resolution temporaries) for one.
            g = rng.standard_normal(
                (3, self.ctrl, self.ctrl, self.ctrl)
            ).astype(np.float32)
            zoom_facs = (1.0,
                         shape[0] / self.ctrl,
                         shape[1] / self.ctrl,
                         shape[2] / self.ctrl)
            fields = ndi_zoom(g, zoom_facs, order=1)[:, :shape[0], :shape[1], :shape[2]]
            fields *= self.alpha
            indices = []
            for ax in range(3):
                base_shape = [1, 1, 1]
                base_shape[ax] = shape[ax]
                base = np.arange(shape[ax], dtype=np.float32).reshape(base_shape)
                # Build the sampling coordinates in-place inside `fields` instead
                # of allocating a fresh `base + field` array and a second array
                # for the clip -- saves two full-volume temporaries per axis.
                coord = fields[ax]
                coord += base
                np.clip(coord, 0, shape[ax] - 1, out=coord)
                indices.append(coord)
            image = map_coordinates(
                image, indices, order=1, mode="constant", cval=self.image_cval
            ).astype(np.float32)
            mask = map_coordinates(
                mask, indices, order=1, mode="constant", cval=self.mask_cval
            )
            mask = (mask > 0.5).astype(np.float32)
        return [image, mask]


class CausalityDataset(Dataset):
    """Dataset that returns a single image-mask pair per sample."""

    def __init__(self, csv_file, transform=None, training=True):
        df = pd.read_csv(csv_file)
        self.arr_files = df['filepaths'].values
        self.transform = transform
        self.training = training

    def __len__(self):
        return len(self.arr_files)

    def __getitem__(self, index):
        filename = self.arr_files[index]
        
        arr = np.load(filename)
        
        img = arr[:,:,:,0] # taking the image array
        mask = arr[:,:,:,1]
        
        mask[mask > 0] = 1
        
        if self.training:
            img = normalise_hu(img) # clipping the pixel value
            img = normalise_one_one(img) # normalizing the image in -1 to +1 values
        else:
            img = normalise_one_one(img)

        mask = normalise_zero_one(mask) # normalising to 0 to 1 values

        # Hygiene (both train and test): keep major connected component(s) and
        # fill holes -- strips the speck/hole artifact without altering the lung.
        mask = _clean_binary(mask)

        sample = [img, mask]

        if self.transform:
            sample = self.transform(sample)

        return sample[0], sample[1], filename

class ToTensor(object):
    def __call__(self, sample):
        assert len(sample) == 2, "There should be an image and mask pair. So two arrays in total"
        
        image, mask = sample[0], sample[1]

        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)
        
        return [torch.from_numpy(image), torch.from_numpy(mask)]

class VariableSpatialFix(object):
    
    def __init__(self, num_of_double_stride_conv, padval=-1):
        self.mul_factor = 2 ** num_of_double_stride_conv
        self.padval = padval

    def __call__(self, sample):
        assert len(sample) == 2, "There should be an image and mask pair. So two arrays in total"
        image, mask = sample[0], sample[1]

        h, w, d = image.shape

        new_h = self.mul_factor * math.ceil(h/self.mul_factor)
        new_w = self.mul_factor * math.ceil(w/self.mul_factor)
        new_d = self.mul_factor * math.ceil(d/self.mul_factor)

        pad_h = new_h - h
        pad_w = new_w - w
        pad_d = new_d - d

        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, pad_d)), 'constant', constant_values=self.padval)
        mask = np.pad(mask, ((0, pad_h), (0, pad_w), (0, pad_d)), 'constant', constant_values=0)
        
        return [image, mask]

class PadOrCrop(object):
    """Force a fixed spatial size by center-cropping (when larger) or
    center-padding (when smaller) on each axis independently.

    All test scans are isotropic 256^3, so forcing the training scans to the
    same 256^3 size keeps train/test resolution consistent and bounds the
    FP32 activation memory (avoids CUDA OOM). The image is padded with -1
    (background after normalise_one_one) and the mask with 0.
    """

    def __init__(self, output_size=(256, 256, 256), image_padval=-1, mask_padval=0):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = tuple(int(s) for s in output_size)
        self.image_padval = image_padval
        self.mask_padval = mask_padval

    def _fix_axis(self, size, target):
        """Return (crop_start, crop_end, pad_before, pad_after) for one axis."""
        if size >= target:
            start = (size - target) // 2
            return start, start + target, 0, 0
        pad_before = (target - size) // 2
        pad_after = (target - size) - pad_before
        return 0, size, pad_before, pad_after

    def __call__(self, sample):
        assert len(sample) == 2, "There should be an image and mask pair. So two arrays in total"
        image, mask = sample[0], sample[1]

        slices = []
        pads = []
        for axis, target in enumerate(self.output_size):
            start, end, pad_before, pad_after = self._fix_axis(image.shape[axis], target)
            slices.append(slice(start, end))
            pads.append((pad_before, pad_after))

        image = image[slices[0], slices[1], slices[2]]
        mask = mask[slices[0], slices[1], slices[2]]

        image = np.pad(image, pads, 'constant', constant_values=self.image_padval)
        mask = np.pad(mask, pads, 'constant', constant_values=self.mask_padval)

        return [image, mask]


class RandomCrop(object):

    def __init__(self, output_size, padval=-1):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size
        self.padval = padval

    def random_crop(self, image, mask):
        h, w, d = image.shape

        new_h, new_w, new_d = self.output_size

        h_diff = abs(min(h - new_h, 0))
        w_diff = abs(min(w - new_w, 0))
        d_diff = abs(min(d - new_d, 0))

        image = np.pad(image, ((0,h_diff), (0, w_diff), (0, d_diff)), 'edge')
        mask = np.pad(mask, ((0,h_diff), (0, w_diff), (0, d_diff)), 'constant', constant_values=0)
        
        h, w, d = image.shape
        top = torch.randint(0, h - new_h + 1, (1,)).item()
        left = torch.randint(0, w - new_w + 1, (1,)).item()
        bottom = torch.randint(0, d - new_d +1, (1,)).item()

        image = image[top: top + new_h,
                      left: left + new_w,
                      bottom: bottom + new_d]
        mask = mask[top: top + new_h,
                      left: left + new_w,
                      bottom: bottom + new_d]

        return [image, mask]

    def __call__(self, sample):
        assert len(sample) == 2, "There should be an image and mask pair. So two arrays in total"
        image, mask = sample[0], sample[1]

        image, mask = self.random_crop(image, mask)
        
        return [image, mask]
