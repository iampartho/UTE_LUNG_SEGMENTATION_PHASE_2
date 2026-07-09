import  os
import sys
import argparse
import re
import pandas as pd
import SimpleITK as sitk
import numpy as np
from connected import largest_object
from metrics import ASD
import tqdm
import numpy as np
import glob

def main():
    

    results = {"sid":[],
               "fold":[],
               "iou":[],
               "dice":[],
               "asd":[],
               "sdsd":[],
               "hd":[]}
    
    all_gt_files = glob.glob("./prediction/mri_to_ct_cyclegan_2_marissa_data/*_gt.nii.gz")
    # all_pred_files = glob.glob("/Shared/lss_segerard/data/UTE_MRI_OECLAD/seg_net/*.mask.0.nii.gz")

    print("Number of pred files", len(all_gt_files))

    for idx in tqdm.tqdm(range(len(all_gt_files))): # all_gt_files

        # pred_mask_path = all_pred_files[idx]

        # filename = os.path.basename(pred_mask_path).split('.mask.0.nii.gz')[0]
        # true_mask_path = f"/Shared/lss_segerard/data/UTE_MRI_OECLAD/MASKS_better/{filename}_mask.nii.gz"
        true_mask_path = all_gt_files[idx]

        filename = os.path.basename(true_mask_path).split('_gt.nii.gz')[0]

        pred_mask_path = true_mask_path.replace('_gt.nii.gz', '_pred.nii.gz')

        print("true_mask_path", true_mask_path, "\n\n")
        print("pred_mask_path", pred_mask_path, "\n\n")



        sid = filename
        

        
        # gt = sitk.ReadImage(true_mask_path)

        gt = sitk.Cast(sitk.BinaryThreshold(sitk.ReadImage(true_mask_path), lowerThreshold=1, upperThreshold=255, insideValue=1, outsideValue=0), sitk.sitkUInt16)
        
    
        
        pred = sitk.Cast(sitk.ReadImage(pred_mask_path), sitk.sitkUInt16)



        pred = sitk.Cast(largest_object(pred, 2), sitk.sitkUInt16)

        segeval = sitk.LabelOverlapMeasuresImageFilter()
        segeval.Execute(gt, pred)
        dice = segeval.GetDiceCoefficient(1)
        iou = segeval.GetJaccardCoefficient(1)


        asd, sdsd, hd = ASD(gt,pred)
        results["sid"].append(sid)
        results["fold"].append(0)
        results["iou"].append(iou)
        results["dice"].append(dice)
        results["asd"].append(asd)
        results["sdsd"].append(sdsd)
        results["hd"].append(hd)
        print(results)
    df = pd.DataFrame(results)
    os.makedirs("./test_result_csv", exist_ok=True)
    df.to_csv("./test_result_csv/mri_to_ct_cyclegan_no_mri_leakage_crop_192_volmatched_on_basicunet_td1_marissa_data_.csv",index=False)




if __name__=="__main__":
    main()

