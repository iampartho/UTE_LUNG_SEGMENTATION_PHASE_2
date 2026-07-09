"""
Dataset for the V-JEPA pretraining stage (3D lung-anatomy adaptation).

Mirrors the conventions of ``dataset_AE.py`` (the ACNN Stage-1 dataset) so the
two self-supervised pretraining stages are interchangeable:

    * Loads the lung mask only (channel 1 of the (H, W, D, 2) numpy volume).
      Modality (UTE vs CT) is irrelevant -- masks are binary in both.
    * Binarises the mask, resizes it to a fixed canonical ``input_size`` (the
      ViT patch-embed needs each dim divisible by the patch size).
    * Applies a Gaussian filter to the (binary) mask before returning it. This
      is the key change requested for the V-JEPA variant: at Stage-2 the
      encoder will see *soft* segmenter predictions, not hard 0/1 masks, so we
      pretrain on softened masks to match that distribution. This plays the
      same "input corruption" role that the denoising flip/noise played in
      ``dataset_AE.py``.

The V-JEPA masking (dropping tokens) is NOT done here -- it operates on the
*token* sequence and is generated on the fly during training
(``model_vjepa.generate_masks``). This dataset only produces the input volume.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage.transform import resize as sk_resize
from scipy.ndimage import gaussian_filter


def _normalise_zero_one(image):
    image = image.astype(np.float32)
    minimum = float(np.min(image))
    maximum = float(np.max(image))
    if maximum > minimum:
        return (image - minimum) / (maximum - minimum)
    return image * 0.0


class VJEPAShapeDataset(Dataset):
    """Returns (mask_volume, filename) per sample.

    The numpy files have layout (H, W, D, 2) where channel 1 is the lung mask.

    Args:
        csv_file: CSV with a single 'filepaths' column.
        transform: optional torchvision-style transform operating on a single
            (H, W, D) volume (kept for parity with the other datasets; usually
            just ``MaskToTensor``).
        input_size: (H, W, D) the volume is resized to. Each dim must be
            divisible by the encoder patch size (default patch 16 -> 96/16=6).
        gaussian_sigma: std-dev (in voxels) of the Gaussian filter applied to
            the binarised mask. Softens the hard 0/1 mask so it resembles the
            soft sigmoid predictions seen at Stage-2. Set 0.0 to disable.
    """

    def __init__(
        self,
        csv_file,
        transform=None,
        input_size=(96, 96, 96),
        gaussian_sigma=1.0,
    ):
        df = pd.read_csv(csv_file)
        self.arr_files = df['filepaths'].values
        self.transform = transform
        self.input_size = tuple(int(s) for s in input_size)
        self.gaussian_sigma = float(gaussian_sigma)

    def __len__(self):
        return len(self.arr_files)

    def __getitem__(self, index):
        filename = self.arr_files[index]

        arr = np.load(filename)
        mask = arr[:, :, :, 1]
        mask[mask > 0] = 1
        mask = _normalise_zero_one(mask).astype(np.float32)

        # Resize to the canonical fixed size. order=0 (nearest) keeps the mask
        # exactly binary before smoothing; preserve_range / no anti-alias avoid
        # any value rescaling.
        mask = sk_resize(
            mask,
            self.input_size,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.float32)

        # Soften the binary mask so the encoder's input distribution matches the
        # soft predictions it will be fed at Stage-2.
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
