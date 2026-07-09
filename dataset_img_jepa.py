"""
Dataset for the Phase 1, step 2 IMAGE-encoder training (cross-modal JEPA).

Goal of step 2 (see the project plan): train an image encoder ``E_img`` (+ a
predictor ``P``) so that ``P(E_img(GIN(CT_img)))`` regresses the frozen Step-1
mask-encoder latent ``E_mask(mask)``. The learned cross-modal energy that Phase 2
steers is then ``E_x(x, m) = ||P(E_img(x)) - E_mask(m)||_1``.

What this dataset returns per sample: a *paired* (image, mask, filename) triple,
both fit to the canonical 256^3 grid (native UTE size -> no resize at Phase 2).

Two design rules carried over from the plan, both implemented here:
    * Geometric shape-aug (anisotropic per-axis scale -> elongation, elastic warp
      -> extent) is applied IDENTICALLY to the image and the mask so they stay
      pixel-aligned (correspondence is the whole point of the regression target).
      We reuse the already-proven PAIRED transforms from datasets_causality.py
      (RandomAnisoScale / RandomElastic) -- the exact same shape-aug family the
      Step-1 mask encoder was trained under, so the two encoders see a symmetric
      shape distribution.
    * The MASK fed to E_mask must match the distribution E_mask was trained on:
      clean (largest comp(s) + fill holes) -> fit 256^3 (centre pad/crop, native
      1.25mm scale) -> Gaussian soften (sigma=1.0). The image instead follows the
      segmentation pipeline's intensity handling (HU clip on CT train, then
      [-1, 1] normalisation); GIN is NOT applied here -- it is a GPU module
      applied to the image only inside the training loop (train_img_jepa.py),
      mirroring causality_train.py.

Note on the image vs. mask channels: the numpy files have layout (H, W, D, 2);
channel 0 is the CT image, channel 1 is the lung mask -- same as datasets_causality.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter

# Reuse the segmentation pipeline's intensity handling, mask hygiene, and the
# PAIRED geometric augmentation (drawn once, applied to image AND mask together).
from datasets_causality import (
    normalise_hu,
    normalise_one_one,
    normalise_zero_one,
    _clean_binary,
    RandomAnisoScale,
    RandomElastic,
    PadOrCrop,
    ToTensor,
)


class MaskGaussianBlur(object):
    """Gaussian-soften the MASK only (the image is passed through untouched).

    The Step-1 mask encoder E_mask was trained on binary masks softened with a
    sigma=1.0 Gaussian (dataset_mask_jepa.GAUSSIAN_SIGMA), so its input looked
    like the soft sigmoid predictions seen downstream. To feed E_mask exactly the
    distribution it was trained on, we apply the SAME softening as the LAST step,
    after every geometric augmentation (so the blur is on the final 256^3 grid).
    Set ``sigma`` = 0 to disable.
    """

    def __init__(self, sigma=1.0):
        self.sigma = float(sigma)

    def __call__(self, sample):
        assert len(sample) == 2, "Expected an [image, mask] pair"
        image, mask = sample[0], sample[1]
        if self.sigma > 0.0:
            mask = gaussian_filter(mask, sigma=self.sigma).astype(np.float32)
            mask = np.clip(mask, 0.0, 1.0)
        return [image, mask]


class ImgJEPADataset(Dataset):
    """Returns (image, mask, filename), both fit to a fixed 256^3 grid.

    Args:
        csv_file: CSV with a single 'filepaths' column. For training this is the
            CT-only list (only_copd_1.25mm.csv); for validation the UTE list
            (only_ute_1.25mm.csv). E_img never sees UTE in training, so lowest
            UTE val L1 is an honest cross-modal generalisation criterion -- the
            same model-selection logic used for the Step-1 mask encoder.
        transform: a torchvision Compose of PAIRED transforms operating on the
            [image, mask] list (geometric aug + MaskGaussianBlur + ToTensor).
            Built in train_img_jepa.py so train/val pipelines differ only by the
            random geometric aug.
        training: if True apply HU clipping to the image before [-1, 1] norm
            (CT); if False skip the HU clip (UTE is not in HU). Mirrors
            datasets_causality.CausalityDataset.
        clean_mask / clean_keep_frac: mask hygiene (largest component(s) + fill
            holes) applied before any geometric aug; matches the Step-1 encoder.
        path_replace: optional (old, new) substring pair to remap CSV paths to
            the current machine. None = use the CSV paths verbatim.
    """

    def __init__(
        self,
        csv_file,
        transform=None,
        training=True,
        clean_mask=True,
        clean_keep_frac=0.10,
        path_replace=None,
    ):
        df = pd.read_csv(csv_file)
        self.arr_files = df["filepaths"].values
        self.transform = transform
        self.training = bool(training)
        self.clean_mask = bool(clean_mask)
        self.clean_keep_frac = float(clean_keep_frac)
        self.path_replace = path_replace

    def __len__(self):
        return len(self.arr_files)

    def _resolve(self, p):
        if self.path_replace is not None:
            p = p.replace(self.path_replace[0], self.path_replace[1])
        return p

    def __getitem__(self, index):
        filename = self.arr_files[index]
        path = self._resolve(filename)

        arr = np.load(path)
        img = arr[:, :, :, 0]   # CT image
        mask = arr[:, :, :, 1]  # lung mask
        mask[mask > 0] = 1

        # Image intensity: HU clip (CT train only) then [-1, 1] -- exactly the
        # segmentation pipeline. GIN is applied later, on the GPU, image-only.
        if self.training:
            img = normalise_hu(img)
        img = normalise_one_one(img)

        # Mask: [0, 1] then hygiene (strip the speck/hole artifact, fill holes),
        # matching the Step-1 mask-encoder input. Geometric aug + blur happen in
        # the transform pipeline so they stay paired with the image.
        mask = normalise_zero_one(mask).astype(np.float32)
        if self.clean_mask:
            mask = _clean_binary(mask, keep_frac=self.clean_keep_frac)

        sample = [img, mask]
        if self.transform:
            sample = self.transform(sample)
        else:
            sample = ToTensor()(sample)

        return sample[0], sample[1], filename
