import os
import torch
import pandas as pd
import numpy as np
from torchvision import transforms, utils
import torchvision
import SimpleITK as sitk
from torch import nn
import tqdm
import math
from skimage import exposure
from skimage.morphology import binary_erosion, binary_opening
from monai.inferers import SlidingWindowInferer
import nibabel as nib
#from monai.networks.nets import UNETR

from basic_unet_disentagled import BasicUNet
from learnable_gin import LearnableGIN3D
# from swin import SwinUNETR
import glob

# from monai.networks.nets import BasicUNet,BasicUNetPlusPlus

def generate_confusion_mask(gt_path, pred_path, output_path):
    # Load the NIfTI files
    gt_nii = nib.load(gt_path)
    pred_nii = nib.load(pred_path)

    gt = gt_nii.get_fdata().astype(np.uint8)
    pred = pred_nii.get_fdata().astype(np.uint8)

    # Initialize confusion mask
    confusion = np.zeros_like(gt, dtype=np.uint8)

    # True Positive (TP): both GT and prediction are 1
    confusion[(gt == 1) & (pred == 1)] = 1

    # False Positive (FP): prediction is 1, GT is 0
    confusion[(gt == 0) & (pred == 1)] = 2

    # False Negative (FN): prediction is 0, GT is 1
    confusion[(gt == 1) & (pred == 0)] = 3

    # Save the confusion mask
    confusion_nii = nib.Nifti1Image(confusion, affine=gt_nii.affine, header=gt_nii.header)
    nib.save(confusion_nii, output_path)



def improve_contrast(image_array):
    """Improve the contrast of the image by rescaling intensity."""
    p1, p90 = np.percentile(image_array, (1, 90))
    image_array_rescaled = exposure.rescale_intensity(image_array, in_range=(p1, p90))
    return image_array_rescaled

def generate_nifty(arr_np, spacing, output_path):

    nifty = sitk.GetImageFromArray(arr_np)
    nifty.SetSpacing(spacing)
    sitk.WriteImage(nifty, output_path)

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

class ToTensor(object):


    def __call__(self, sample):
        if len(sample) == 2:
            image = sample[0]
            mask = sample[1]
        else:
            image = sample
            mask = None

        if mask is not None:
            image = np.expand_dims(image, axis=0)
            mask = np.expand_dims(mask, axis=0)
            return [torch.from_numpy(image), torch.from_numpy(mask)]
        else:
            image = np.expand_dims(image, axis=0)
            return torch.from_numpy(image)


        


class VariableSpatialFix(object):
    
    def __init__(self, num_of_double_stride_conv, padval=-1):

        self.mul_factor = 2 ** num_of_double_stride_conv
        self.padval = padval

    def __call__(self, sample):

        if len(sample) == 2:
            image = sample[0]
            mask = sample[1]
        else:
            image = sample
            mask = None
        
        h, w, d = image.shape

        new_h = self.mul_factor * math.ceil(h/self.mul_factor)
        new_w = self.mul_factor * math.ceil(w/self.mul_factor)
        new_d = self.mul_factor * math.ceil(d/self.mul_factor)

        pad_h = new_h - h
        pad_w = new_w - w
        pad_d = new_d - d

        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, pad_d)), 'constant', constant_values=self.padval)

        if mask is not None:
            mask = np.pad(mask, ((0, pad_h), (0, pad_w), (0, pad_d)), 'constant', constant_values=0)
            return [image, mask]
        else:
            return image

        

class RandomCrop(object):

    def __init__(self, output_size, padval=-1):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size
        self.padval = padval

    def __call__(self, sample):
        
        image = sample[0]
        mask = sample[1]
        
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

def resample_ref(im, ref, outside_val, interp):
     resampleFilter = sitk.ResampleImageFilter()
     resampleFilter.SetInterpolator(interp)
     resampleFilter.SetDefaultPixelValue(outside_val)
     resampleFilter.SetReferenceImage(ref)
     resampleFilter.SetOutputDirection(ref.GetDirection())
     resampleFilter.SetOutputOrigin(ref.GetOrigin())
     resampleFilter.SetOutputSpacing(ref.GetSpacing())
     resampleIm = resampleFilter.Execute(im)

     # print(resampleIm.GetSpacing())
     # print(ref.GetSpacing())

     # print(resampleIm.GetOrigin())



     return resampleIm

# Create wrapper functions for sliding window inference with modality parameter
def model_ute_wrapper(x):
    return model(x, "UTE")

def model_ct_wrapper(x):
    return model(x, "CT")

# test_df_1 = pd.read_csv('./ids/causality_test_ute_copd.csv') #pd.read_csv('./ids/test_0_fold.csv') #pd.read_csv('./ids/test.csv')
# test_df_2 = pd.read_csv('./ids/only_ute_1.25mm.csv')
# test_df = test_df_1[test_df_1["filepaths"].isin(test_df_2["filepaths"])]

# test_df1 = pd.read_csv("./ids/ute_test.csv")
# test_df2 = pd.read_csv("./ids/copd_test_1.25mm.csv")
# test_df = pd.concat([test_df1, test_df2])

if __name__ == "__main__":
    test_df = pd.read_csv('./ids/UTE_MRI_previous_numpy_without_clipping.csv') #pd.read_csv('./ids/UTE_MRI_previous_numpy.csv') #pd.read_csv('./ids/causality_test_ute_copd.csv')

    all_files = test_df['filepaths'].values
    # all_files = test_df['filepaths'].values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BasicUNet().to(device)


    checkpoint = './save_models/best_bunet_causality_paper_ct_train_UTE_test_w_tversky_wo_kl_only_gin_roughness_enforced_5_normalised_gin.pth'

    # checkpoint_gin = './save_models/best_augmentor_learnable_gin_mode_4.pth'

    # augmentor = LearnableGIN3D(out_channel=1, in_channel=1, interm_channel=2, kernel_size=3).to(device)
    # if checkpoint_gin:
    #     augmentor.load_state_dict(torch.load(checkpoint_gin, map_location=device.type))


    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location=device.type))

    transform = transforms.Compose(
                                    [
                                        # RandomCrop(output_size=(256, 256, 256), padval=0.5),
                                        VariableSpatialFix(num_of_double_stride_conv=4, padval=0.5), #CenterCrop(crop_size, padval=0.5, new_preprocess=True), #VariableSpatialFix(num_of_double_stride_conv=3, padval=0.5),
                                        ToTensor()
                                    ])

    output_nifty_dir = './prediction/temp' #'./prediction/baseline_swin' #'./test_data_prediction/baseline_swin'


    os.makedirs(output_nifty_dir, exist_ok=True)


    # Only for Random Crop
    sliding_window_inferer = SlidingWindowInferer(
            roi_size=(256, 256, 256),     # Region of Interest size - matches training crop
            sw_batch_size=1,             # Process one window at a time
            overlap=0.25,                # 25% overlap between windows for smoother results
            mode='gaussian',            # Gaussian blending for overlap regions
            sigma_scale=0.125,          # Controls Gaussian blending sharpness
            padding_mode='constant',     # Pad with zeros at volume borders
            cval=0.0                    # Constant value for padding
        )

    # save_targets = ['AnatCorrLungs_UTE_13.npy']

    for idx, each_file in tqdm.tqdm(enumerate(all_files)):


        
        

        # filename_nifty = os.path.basename(each_file).split('.npy')[0] + ".nii.gz"

        arr = np.load(each_file)

        img_np = arr[:,:,:,0]

        # img_np = improve_contrast(img_np, 1, 90) # for domain shift

        img_np = normalise_one_one(img_np)
        img_gt = arr[:,:,:,1]
        img_gt = normalise_zero_one(img_gt)


        h, w, d = img_np.shape

        # generate_nifty(img_np, (1,1,1), f"{output_nifty_dir}/{filename.split('.npy')[0]}.nii.gz")
        # generate_nifty(img_gt, (1.25,1.25,1.25), f"{output_nifty_dir}/{filename.split('.npy')[0]}_gt.nii.gz")

        #img_np = z_score_normalization(img_np)

        # img_np = improve_contrast(img_np, 1, 90)


        img_tensor, gt_tensor = transform([img_np, img_gt])
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # print("The shape of the image tensor is ", img_tensor.shape)

        
        #print(img_tensor.shape)

        with torch.no_grad():
            filename = f"{os.path.basename(each_file).split('.npy')[0]}_UTE_{idx}.npy"
            # if 'UTE' in each_file:
            #     filename = f"{os.path.basename(each_file).split('.npy')[0]}_UTE_{idx}.npy"
            #     pred = sliding_window_inferer(img_tensor, model_ute_wrapper)
            # else:
            #     filename = os.path.basename(each_file)
            #     pred = sliding_window_inferer(img_tensor, model_ct_wrapper)
            # aug_output = augmentor(img_tensor)

            # if not (filename in save_targets):
            #     continue

            pred  = model(img_tensor)#, isMRI)
            pred = torch.sigmoid(pred)
            pred[pred > 0.5] = 1
            pred[pred <= 0.5] = 0

            #print(torch.unique(pred))

            pred_np = pred.detach().squeeze().cpu().numpy().astype(np.uint8)
            # pred_np = pred_np[:h, :w, :d]
            # generate_nifty(img_np, (1.25,1.25,1.25), f"{output_nifty_dir}/{filename.split('.npy')[0]}_input_op.nii.gz")
            generate_nifty(pred_np, (1.25,1.25,1.25), f"{output_nifty_dir}/{filename.split('.npy')[0]}_pred.nii.gz")
            generate_nifty(gt_tensor.squeeze().cpu().numpy(), (1.25,1.25,1.25), f"{output_nifty_dir}/{filename.split('.npy')[0]}_gt.nii.gz")

            # generate_confusion_mask(f"{output_nifty_dir}/{filename.split('.npy')[0]}_gt.nii.gz", f"{output_nifty_dir}/{filename.split('.npy')[0]}_pred_unet.nii.gz", f"{output_nifty_dir}/{filename.split('.npy')[0]}_confusion_unet.nii.gz")