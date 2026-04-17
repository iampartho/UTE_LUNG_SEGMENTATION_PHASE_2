import torch
import pandas as pd
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch import nn
from torch.utils.data import DataLoader
import math




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
