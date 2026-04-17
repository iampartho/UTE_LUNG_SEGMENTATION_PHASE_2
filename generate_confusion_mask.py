import os
import nibabel as nib
import numpy as np
import pandas as pd
import tqdm
import SimpleITK as sitk

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


def generate_nifty_image(target_string, csv_path, output_path):
    df = pd.read_csv(csv_path)
    all_files = df['filepaths'].values
    for idx, each_file in enumerate(tqdm.tqdm(all_files)):
        filename = f"{os.path.basename(each_file).split('.npy')[0]}_UTE_{idx}"

        if filename == target_string:
            arr = np.load(each_file)
            img = arr[:,:,:,0]
            img = sitk.GetImageFromArray(img)
            img.SetSpacing([1.25, 1.25, 1.25])
            sitk.WriteImage(img, output_path)
            break
        

def process_nifti_folder():

    # test_df = pd.read_csv('./ids/test.csv')
    # all_files = test_df['img_name'].values

    # prediction_folder = './prediction/baseline_swin_tvs'



    # for each_file in tqdm.tqdm(all_files):
    #     each_file = each_file.split('/')[0]
    #     gt_path = f"{prediction_folder}/{each_file}_gt.nii.gz"
    #     pred_path = f"{prediction_folder}/{each_file}_pred.nii.gz"
    #     output_path = f"{prediction_folder}/{each_file}_confusion.nii.gz"
    #     generate_confusion_mask(gt_path, pred_path, output_path)
    pred_path = './prediction/temp/AnatCorrLungs_UTE_70_pred.nii.gz'
    gt_path = './prediction/temp/AnatCorrLungs_UTE_70_gt.nii.gz'
    output_path = './prediction/temp/AnatCorrLungs_UTE_70_confusion.nii.gz'
    generate_confusion_mask(gt_path, pred_path, output_path)




# Example usage:
generate_nifty_image('AnatCorrLungs_UTE_70', './ids/only_ute_1.25mm.csv', 'temp.nii.gz')
process_nifti_folder()
