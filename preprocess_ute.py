import os
import sys
import glob
import argparse
import numpy as np
import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from skimage import exposure


def resample_iso(im, spacing, outside_val, interp):
     inputSize = im.GetSize()
     inputSpacing = im.GetSpacing()
     inputOrigin = im.GetOrigin()
     inputDirection = im.GetDirection()

     outputSpacing = [spacing]*im.GetDimension()
     outputSize = np.ceil( (np.array(inputSize) * np.array(inputSpacing)) / outputSpacing)

     outputSizeInt = [int(s) for s in outputSize]
     resampleFilter = sitk.ResampleImageFilter()
     resampleFilter.SetInterpolator(interp)
     resampleFilter.SetOutputDirection(inputDirection)
     resampleFilter.SetOutputOrigin(inputOrigin)
     resampleFilter.SetOutputSpacing(outputSpacing)
     resampleFilter.SetSize(outputSizeInt)
     resampleFilter.SetDefaultPixelValue(outside_val)
     resampleIm = resampleFilter.Execute(im)
     return resampleIm
def resample_ref(im, ref, outside_val, interp):
     resampleFilter = sitk.ResampleImageFilter()
     resampleFilter.SetInterpolator(interp)
     resampleFilter.SetDefaultPixelValue(outside_val)
     resampleFilter.SetReferenceImage(ref)
     resampleIm = resampleFilter.Execute(im)
     return resampleIm
def unpad(x, pad_width):
    slices = []
    for c in pad_width:
        e = None if c[1] == 0 else -c[1]
        slices.append(slice(c[0], e))
    return x[tuple(slices)]
def get_lung_inds(mask):

    img_shape = mask.shape
    inds = np.array(np.where(mask>0))
    min_inds = np.min(inds, axis = 1)
    max_inds = np.max(inds, axis = 1)

    upper_margin = img_shape - max_inds
    print(upper_margin)
    upper_margin[0] = 0
    print(upper_margin)
    return np.array(list(zip(min_inds, upper_margin)))

def improve_contrast(image_array):
    """Improve the contrast of the image by rescaling intensity."""
    p1, p90 = np.percentile(image_array, (1, 90))
    image_array_rescaled = exposure.rescale_intensity(image_array, in_range=(p1, p90))
    return image_array_rescaled

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir", type = str, help = "root directory")
    parser.add_argument("--csv", type = str, help = "csv filename")
    #parser.add_argument("--outdir", type = str, help = "output directory")

    args = parser.parse_args()
    rootdir = args.rootdir
    csv = args.csv
    #outdir = args.outdir

    all_img_file_dir = glob.glob("/Shared/lss_segerard/data/UTE_MRI_correct/*/*-BC.nii.gz")

    print(len(all_img_file_dir))

    assert len(all_img_file_dir) == 50, "Some problem in the file directory"

    output_dir = '/Shared/lss_segerard/parthghosh/data/UTE_previous_data_numpy_without_clipping'
    os.makedirs(output_dir, exist_ok=True)
    for each_filepath in tqdm.tqdm(all_img_file_dir):

        directory = os.path.dirname(each_filepath)
        filename = os.path.basename(each_filepath)
        # des_directory = directory.replace("/Shared/lss_segerard/data/IPF-UTE-CT-IsilonArchive/Normalized_Data_correct", output_dir)
        os.makedirs(output_dir, exist_ok=True)

        des_path = output_dir + "/" + filename
        des_path = des_path.replace(".nii.gz", ".npy")

        mask_filepath = each_filepath.replace("-BC.nii.gz", "_mask.seg.nrrd")

        

        # img_filepath = f"{each_filepath}/{filename}-BC.nii.gz"
        # mask_filepath = f"{each_filepath}/{filename}_mask.seg.nrrd"



        
        
        # print(img_filepath, mask_filepath)
        img = sitk.ReadImage(each_filepath) # reading the original nifti file (ct image volume)
        airway = sitk.ReadImage(mask_filepath) # reading the ground truth of the image file 


        dirs = img.GetDirection() # it returns a 3x3 matrix to denote if the image is rotated or sheered correspond to the patient body axes, a identity matrix represent no rotation
        print(dirs)
        dirs_diag = np.array([dirs[0], dirs[4], dirs[8]]) # takes the diagonal of the matrix
        dirs_diag = np.sign(dirs_diag) # takes the sign of the each element
        flips = [False if x==1 else True for x in dirs_diag] # to make sure if there is any flipping in any of the axes
        print(flips)
        img = sitk.Flip(img, flips) # if there was any flipping then fix the flipping on img, ground truth and on the lungs lobe segmentation
        airway = sitk.Flip(airway, flips)

        img_resample = resample_iso(img, 1.25, 0, sitk.sitkBSpline) # resample the images so that the spacing is 1 in all axes and -1024 to use for pixels outside the input image bounds after resampling.
        airway_resample = resample_ref(airway, img_resample, 0, sitk.sitkNearestNeighbor) # sitk.sitkBSpline) # since the image is resamples so the gt and lobe segmentation is also required to be resampled
        
        img_np = sitk.GetArrayFromImage(img_resample) # converting the image, gt and lung lobe to numpy array
        # img_np = improve_contrast(img_np)
        airway_np = sitk.GetArrayFromImage(airway_resample)
        airway_np[airway_np > 0] = 1



        sample = np.stack((img_np, airway_np), axis=-1) # stacking the ct image volume and the mask together in a new axis basically if the image is for example is (320,300,300) then gt will be the same and after stacking the resulting array will be (320,300,300,2)
        # print(sample.shape)

        np.save(des_path, sample)

    #crop_margin = get_lung_inds(masknp)
    #ct_np = unpad(ct_np, crop_margin)
if __name__=="__main__":
    main()
