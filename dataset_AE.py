"""
Dataset for Stage-1 of the ACNN approach (Oktay et al., 2018, IEEE TMI).

The AE is trained ONLY on lung-mask label maps (single channel, binary).
Both UTE and COPDgene CT volumes are loaded from the same CSV and only the
mask channel (arr[:,:,:,1]) is used.

Paper-faithful pipeline (Sec. III-B):
    1. Load and binarise the lung mask.
    2. Resize to a fixed canonical ``input_size`` (the paper crops/normalises
       to a fixed size before the FC bottleneck). Nearest-neighbour
       interpolation keeps the mask binary.
    3. (training only) Apply denoising-AE corruption to a copy:
         * voxel-flip with prob ``flip_prob`` (binary analogue of the
           paper's label-swap p=0.1)
         * additive Gaussian noise with std ``noise_std``
       The corrupted mask is the AE *input*; the clean resized mask is the
       reconstruction *target*.
    4. (validation) No corruption; input == target.
"""
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage.transform import resize as sk_resize


def _normalise_zero_one(image):
    image = image.astype(np.float32)
    minimum = float(np.min(image))
    maximum = float(np.max(image))
    if maximum > minimum:
        return (image - minimum) / (maximum - minimum)
    return image * 0.0


class ShapeAEDataset(Dataset):
    """Returns (corrupted_mask, clean_mask, filename) per sample.

    The numpy files have layout (H, W, D, 2) where channel 1 is the lung mask.
    Modality (UTE vs CT) does not matter here -- masks are binary in both cases.

    Args:
        csv_file: CSV with a single 'filepaths' column.
        transform: torchvision-style transform that operates on
            ``[corrupted_mask, clean_mask]`` (mirrors the
            ``[image, mask]`` convention of CausalityDataset). NOTE: the
            sample is ALREADY the canonical fixed size at this point.
        training: If True, applies denoising corruption to the input.
        noise_std: Std-dev of the additive Gaussian noise on the corrupted
            input (paper recommends a small value).
        flip_prob: Per-voxel probability of flipping 0<->1 on the corrupted
            input (the binary analogue of the paper's label-swap p=0.1).
        input_size: (H, W, D) tuple. The mask is resized to this size before
            any corruption -- matches the paper's fixed-size FC AE.
    """

    def __init__(
        self,
        csv_file,
        transform=None,
        training=True,
        noise_std=0.1,
        flip_prob=0.1,
        input_size=(96, 96, 96),
    ):
        df = pd.read_csv(csv_file)
        self.arr_files = df['filepaths'].values
        self.transform = transform
        self.training = training
        self.noise_std = float(noise_std)
        self.flip_prob = float(flip_prob)
        self.input_size = tuple(int(s) for s in input_size)

    def __len__(self):
        return len(self.arr_files)

    def __getitem__(self, index):
        filename = self.arr_files[index]

        arr = np.load(filename)
        mask = arr[:, :, :, 1]
        mask[mask > 0] = 1
        mask = _normalise_zero_one(mask).astype(np.float32)

        # Resize to the canonical fixed size. order=0 (nearest) keeps the
        # mask exactly binary 0/1; preserve_range=True / anti_aliasing=False
        # avoid any value rescaling.
        mask = sk_resize(
            mask,
            self.input_size,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.float32)

        if self.training:
            corrupted = mask.copy()

            if self.flip_prob > 0.0:
                flip_idx = np.random.rand(*corrupted.shape) < self.flip_prob
                corrupted[flip_idx] = 1.0 - corrupted[flip_idx]

            if self.noise_std > 0.0:
                noise = np.random.randn(*corrupted.shape).astype(np.float32) * self.noise_std
                corrupted = corrupted + noise

            corrupted = np.clip(corrupted, 0.0, 1.0).astype(np.float32)
        else:
            corrupted = mask.copy()

        sample = [corrupted, mask]

        if self.transform:
            sample = self.transform(sample)

        return sample[0], sample[1], filename


class MaskToTensor(object):
    """ToTensor variant for two single-channel mask volumes.

    Each entry is expected to be (H, W, D) numpy. Adds a leading channel dim.
    """

    def __call__(self, sample):
        assert len(sample) == 2, "Expecting [corrupted_mask, clean_mask] pair."
        corrupted, clean = sample[0], sample[1]
        corrupted = np.expand_dims(corrupted, axis=0)
        clean = np.expand_dims(clean, axis=0)
        return [torch.from_numpy(corrupted), torch.from_numpy(clean)]


class MaskVariableSpatialFix(object):
    """Pads a mask pair to multiples of ``2**num_of_double_stride_conv``.

    NOT used in the paper-faithful AE pipeline (we resize to a fixed size
    instead). Kept here for backward compatibility / debugging.
    """

    def __init__(self, num_of_double_stride_conv, padval=0):
        self.mul_factor = 2 ** int(num_of_double_stride_conv)
        self.padval = float(padval)

    def __call__(self, sample):
        assert len(sample) == 2, "Expecting [corrupted_mask, clean_mask] pair."
        corrupted, clean = sample[0], sample[1]

        h, w, d = corrupted.shape

        new_h = self.mul_factor * math.ceil(h / self.mul_factor)
        new_w = self.mul_factor * math.ceil(w / self.mul_factor)
        new_d = self.mul_factor * math.ceil(d / self.mul_factor)

        pad_h = new_h - h
        pad_w = new_w - w
        pad_d = new_d - d

        corrupted = np.pad(
            corrupted,
            ((0, pad_h), (0, pad_w), (0, pad_d)),
            'constant',
            constant_values=self.padval,
        )
        clean = np.pad(
            clean,
            ((0, pad_h), (0, pad_w), (0, pad_d)),
            'constant',
            constant_values=0,
        )
        return [corrupted, clean]
