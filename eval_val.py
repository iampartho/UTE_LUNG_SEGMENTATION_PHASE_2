import  os
import sys
import argparse
import re
import pandas as pd
import SimpleITK as sitk
import numpy as np
from lungute.common.connected import largest_object
from lungute.eval.metrics import ASD

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infn", type = str, help = "Input Filename")
    #parser.add_argument("--outfn", type = str, help = "Filename")
    args = parser.parse_args()
    infn = args.infn
    #outfn = args.outfn

    print("Input filename: {}".format(infn), flush = True)

    results = {"sid":[],
               "fold":[],
               "iou":[],
               "dice":[],
               "asd":[],
               "sdsd":[],
               "hd":[]}
    filenames = pd.read_csv(infn, header = None)
    num_files = len(filenames)

    rootdir="/Shared/lss_segerard/data/UTE_MRI/"
    for idx in range(num_files):


        imfn = os.path.join(rootdir,
                             filenames.iloc[idx, 0])
        gtfn = os.path.join(rootdir,
                               filenames.iloc[idx, 1])
        fold = filenames.iloc[idx, 2]


        sid = os.path.basename(os.path.dirname(imfn))
        print(imfn)
        print(gtfn)
        outdir = os.path.join(os.path.dirname(gtfn),'seg_net')
        outbase = os.path.basename(gtfn)
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        im = sitk.ReadImage(imfn)
        gt = sitk.ReadImage(gtfn)
        gt = sitk.Cast(sitk.BinaryThreshold(gt, 1, 255), sitk.sitkUInt16)
        probfn = os.path.join(outdir, outbase.replace("_mask.seg.nrrd","_prob.nii.gz"))
        outfn = os.path.join(outdir, outbase.replace("_mask.seg.nrrd","_mask.nii.gz"))

        prob = sitk.Cast(sitk.ReadImage(probfn), sitk.sitkUInt16)
        pred = sitk.Cast(sitk.ReadImage(outfn), sitk.sitkUInt16)



        pred = sitk.Cast(largest_object(pred, 2), sitk.sitkUInt16)

        segeval = sitk.LabelOverlapMeasuresImageFilter()
        segeval.Execute(gt, pred)
        dice = segeval.GetDiceCoefficient(1)
        iou = segeval.GetJaccardCoefficient(1)


        asd, sdsd, hd = ASD(gt,pred)
        results["sid"].append(sid)
        results["fold"].append(fold)
        results["iou"].append(iou)
        results["dice"].append(dice)
        results["asd"].append(asd)
        results["sdsd"].append(sdsd)
        results["hd"].append(hd)
        print(results)
    df = pd.DataFrame(results)
    df.to_csv("results/results_val.csv",index=False)




if __name__=="__main__":
    main()

