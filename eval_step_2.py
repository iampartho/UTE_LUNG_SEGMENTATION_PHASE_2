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
    
    all_gt_files = glob.glob("./prediction/temp/*_gt.nii.gz")

    for idx in tqdm.tqdm(range(len(all_gt_files))):

        
        true_mask_path = all_gt_files[idx]

        filename = os.path.basename(true_mask_path).split('_gt.nii.gz')[0]

        pred_mask_path = true_mask_path.replace('_gt.nii.gz', '_pred.nii.gz')

        print("true_mask_path", true_mask_path, "\n\n")
        print("pred_mask_path", pred_mask_path, "\n\n")



        sid = filename
        

        
        # gt = sitk.ReadImage(true_mask_path)
        gt = sitk.Cast(sitk.ReadImage(true_mask_path), sitk.sitkUInt16)
        
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
    df.to_csv("./test_result_csv/td1_roughness_enforced_5_normalised_gin_prev_data.csv",index=False)




if __name__=="__main__":
    main()

