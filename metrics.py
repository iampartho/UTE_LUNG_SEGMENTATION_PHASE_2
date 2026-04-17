import SimpleITK as sitk
import numpy as np

def ASD(truth,pred):
    truth_contour = sitk.BinaryContour(truth)
    pred_contour = sitk.BinaryContour(pred)

    truth_dist = sitk.SignedMaurerDistanceMap(truth_contour,squaredDistance=False,useImageSpacing=True)
    pred_dist = sitk.SignedMaurerDistanceMap(pred_contour,squaredDistance=False,useImageSpacing=True)

    truth_contour_np = sitk.GetArrayFromImage(truth_contour)
    pred_contour_np = sitk.GetArrayFromImage(pred_contour)

    truth_dist_np = sitk.GetArrayFromImage(truth_dist)
    pred_dist_np= sitk.GetArrayFromImage(pred_dist)

    dists_to_truth = truth_dist_np[pred_contour_np>0]
    dists_to_pred = pred_dist_np[truth_contour_np>0]
    dists_to_truth[dists_to_truth<0]=0
    dists_to_pred[dists_to_pred<0]=0

    comb_dists = np.hstack([dists_to_pred, dists_to_truth])
    asd = np.mean(comb_dists)
    sdsd = np.std(comb_dists)
    maxsd = np.max(comb_dists)
    return asd, sdsd, maxsd


