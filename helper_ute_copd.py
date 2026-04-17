import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
import SimpleITK as sitk
import numpy.ma as ma
from torch import nn
from torch.utils.data import DataLoader
from monai.utils import first
from torchvision import datasets
from torchvision.transforms import ToTensor
from monai.networks.nets import UNet, SwinUNETR,BasicUNet,BasicUNetPlusPlus
from monai.losses import DiceLoss, DiceCELoss, TverskyLoss,GeneralizedDiceLoss, DiceFocalLoss, GeneralizedDiceFocalLoss
from monai import metrics
from datetime import date
from monai.networks import one_hot
from skimage import exposure

today = date.today()
import math
import random

def normalise_hu(image, hu_range=[-1024.0,200.0]):#hu_range=[-1030.0,-230.0]):
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





class UTEDataset(Dataset):
    """AIIB MICCAI dataset."""

    def __init__(self, ute_csv_file, copd_csv_file, transform=None, training=True):
        df = pd.read_csv(ute_csv_file)
        self.arr_files_ute = df['filepaths'].values
        df = pd.read_csv(copd_csv_file)
        self.arr_files_copd = df["filepaths"].values

        self.transform = transform
        self.training = training
        

    def __len__(self):
        """Returns the size of the larger dataset to ensure all images are seen during training."""
        return max(len(self.arr_files_ute), len(self.arr_files_copd))


    def __getitem__(self, index):
        

        filename = self.arr_files_ute[index % len(self.arr_files_ute)]
        
        arr_ute = np.load(filename)
        

        
        arr_ute_img = arr_ute[:,:,:,0] # taking the image array


        arr_ute_img = normalise_one_one(arr_ute_img)

        arr_ute_mask = arr_ute[:,:,:,1]
        arr_ute_mask[arr_ute_mask > 0] = 1
        arr_ute_mask = normalise_zero_one(arr_ute_mask)



        
        img_name = self.arr_files_copd[index % len(self.arr_files_copd)]

        arr = np.load(img_name) # loading the preprocessed numpy array

        arr_copd_img = arr[:,:,:,0] # taking the image array
        arr_copd_mask = arr[:, :, :, 1]#np.load(f"{self.root_dir}datasets/preprocess_new_gt/{os.path.basename(img_name)}")#arr[:, :, :, 1]
        #arr_copd_mask_bd = arr[:,:,:,2]



        arr_copd_mask[arr_copd_mask > 0] = 1
        #arr_copd_mask_bd[arr_copd_mask_bd > 0] = 1

        HU_image = normalise_hu(arr_copd_img) # clipping the pixel value in -1024 to 200

        copd_image = normalise_one_one(HU_image) # normalizing the image in -1 to +1 values
        #copd_image = z_score_normalization(copd_image)
        copd_mask = normalise_zero_one(arr_copd_mask) # normalising to 0 to 1 values
        #copd_mask_bd = normalise_zero_one(arr_copd_mask_bd)
        
        
        sample = [arr_ute_img, arr_ute_mask, copd_image, copd_mask]#, filename, img_name]


        if self.transform:
            sample = self.transform(sample)
        
        if self.training:
            sample.append(filename)
            sample.append(img_name)

        return sample

class ToTensor(object):


    def __call__(self, sample):

        assert len(sample) == 4, "There should a UTE and CT image-mask pairs. So four array in total "
        
        ute_image, ute_mask, copd_image, copd_mask = sample[0], sample[1], sample[2], sample[3]

        ute_image = np.expand_dims(ute_image, axis=0)


        ute_mask = np.expand_dims(ute_mask, axis=0)
        

        copd_image = np.expand_dims(copd_image, axis=0)


        copd_mask = np.expand_dims(copd_mask, axis=0)
        

        return [torch.from_numpy(ute_image), torch.from_numpy(ute_mask), torch.from_numpy(copd_image), torch.from_numpy(copd_mask)]

        
class VariableSpatialFix(object):
    
    def __init__(self, num_of_double_stride_conv, padval=-1):

        self.mul_factor = 2 ** num_of_double_stride_conv
        self.padval = padval

    def __call__(self, sample):

        assert len(sample) == 4, "There should a UTE and CT image-mask pairs. So four array in total "
        ute_image, ute_mask, copd_image, copd_mask = sample[0], sample[1], sample[2], sample[3]



        h, w, d = ute_image.shape

        new_h = self.mul_factor * math.ceil(h/self.mul_factor)
        new_w = self.mul_factor * math.ceil(w/self.mul_factor)
        new_d = self.mul_factor * math.ceil(d/self.mul_factor)

        pad_h = new_h - h
        pad_w = new_w - w
        pad_d = new_d - d

        ute_image = np.pad(ute_image, ((0, pad_h), (0, pad_w), (0, pad_d)), 'constant', constant_values=self.padval)

        
        
        ute_mask = np.pad(ute_mask, ((0, pad_h), (0, pad_w), (0, pad_d)), 'constant', constant_values=0)
        #ute_mask_bd = np.pad(ute_mask_bd, ((0, pad_h), (0, pad_w), (0, pad_d)), 'constant', constant_values=0)


        h, w, d = copd_image.shape

        new_h = self.mul_factor * math.ceil(h/self.mul_factor)
        new_w = self.mul_factor * math.ceil(w/self.mul_factor)
        new_d = self.mul_factor * math.ceil(d/self.mul_factor)

        pad_h = new_h - h
        pad_w = new_w - w
        pad_d = new_d - d

        copd_image = np.pad(copd_image, ((0, pad_h), (0, pad_w), (0, pad_d)), 'constant', constant_values=self.padval)

        
        
        copd_mask = np.pad(copd_mask, ((0, pad_h), (0, pad_w), (0, pad_d)), 'constant', constant_values=0)
        #copd_mask_bd = np.pad(copd_mask_bd, ((0, pad_h), (0, pad_w), (0, pad_d)), 'constant', constant_values=0)
        
        return [ute_image, ute_mask, copd_image, copd_mask]

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
        
        assert len(sample) == 4, "There should a UTE and CT image-mask pairs. So four array in total "
        ute_image, ute_mask, copd_image, copd_mask = sample[0], sample[1], sample[2], sample[3]

        ute_image, ute_mask = self.random_crop(ute_image, ute_mask)
        copd_image, copd_mask = self.random_crop(copd_image, copd_mask)
        
        return [ute_image, ute_mask, copd_image, copd_mask]




