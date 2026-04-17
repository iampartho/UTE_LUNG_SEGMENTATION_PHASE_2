import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch import nn
from torch.utils.data import DataLoader
#from monai.networks.nets import UNETR
from monai.losses import DiceLoss, DiceCELoss, TverskyLoss,GeneralizedDiceLoss, DiceFocalLoss, GeneralizedDiceFocalLoss, HausdorffDTLoss
from monai.inferers import SlidingWindowInferer
#from monai import metrics
from datetime import date
today = date.today()
from datasets_causality import CausalityDataset, ToTensor, VariableSpatialFix, RandomCrop
import torch.nn.functional as F
import SimpleITK as sitk



#For Mixed Precision

#from GDFloss import GeneralizedDiceFocalLoss
import torch
from torch.cuda.amp import autocast


# For Data Parallel
import torch.nn.parallel


#from unet_ute import UNet
# from swin import SwinUNETR
from basic_unet_disentagled import BasicUNet
from gin_with_log_capability import CausalityAugmentation3D

#from unet_2_dec import UNet



global_loss = np.inf
global_train_loss = np.inf
epoch_since_loss_didnt_improve = 0
LAMBDA_DIV = 1


def log_augmentation_weights(csv_path, filename, loss_value, params):
    row_data = {'filename': filename}
    for i, p in enumerate(params):
        row_data[f'kernel_size_{i}'] = p['k']
        row_data[f'kernel_{i}'] = p['w'].cpu().numpy().flatten().tolist()

    row_data['loss_value'] = float(loss_value)

    columns = ['filename']
    for i in range(len(params)):
        columns.append(f'kernel_size_{i}')
        columns.append(f'kernel_{i}')
    columns.append('loss_value')

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(columns=columns)

    if filename in df['filename'].values:
        existing_loss = df.loc[df['filename'] == filename, 'loss_value'].values[0]
        if loss_value < existing_loss:
            idx = df.index[df['filename'] == filename].tolist()[0]
            for col, val in row_data.items():
                if isinstance(val, list):
                    df.at[idx, col] = str(val)
                else:
                    df.at[idx, col] = val
            df.to_csv(csv_path, index=False)
    else:
        row_ready = {}
        for k, v in row_data.items():
            row_ready[k] = str(v) if isinstance(v, list) else v
        new_row_df = pd.DataFrame([row_ready])
        df = pd.concat([df, new_row_df], ignore_index=True)
        df.to_csv(csv_path, index=False)


def save_nifty(array, spacing, save_path):
    out_img = sitk.GetImageFromArray(array.astype(np.float32))
    out_img.SetSpacing(spacing)
    sitk.WriteImage(out_img, save_path)
    print(f"Saved: {save_path}")

def calculate_iou_precision(pred_masks, true_masks, threshold=0.5):
    # Convert prediction masks to binary masks based on threshold
    pred_mask = (pred_masks > threshold).astype(np.uint8)
    true_mask = (true_masks > threshold).astype(np.uint8)
    

    # Iterate over each sample in the batch (not necessary for single sample)
    #for pred_mask, true_mask in zip(pred_masks_binary, true_masks_binary):
    # Calculate intersection and union
    intersection = np.sum(np.logical_and(pred_mask, true_mask))
    union = np.sum(np.logical_or(pred_mask, true_mask))
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0.0
    

    # Calculate Precision
    precision = intersection / (np.sum(pred_mask) + 1e-6)  # Add epsilon to avoid division by zero


    
    return [iou, precision]



class BinarySoftDiceLoss(nn.Module):
    """
    Computes the Soft Dice Loss for single-channel binary segmentation.
    Expects inputs to be logits (before sigmoid).
    """
    def __init__(self, smooth=1.0):
        super(BinarySoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # 1. Apply Sigmoid to get probabilities [0, 1]
        probs = torch.sigmoid(logits)
        
        # 2. Flatten for efficient calculation
        # (N, 1, D, H, W) -> (N, -1)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        
        # 3. Calculate intersection and union
        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        
        # 4. Compute Dice Score
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # 5. Return Dice Loss (1 - Dice Score)
        return 1.0 - dice_score.mean()

def binary_kl_consistency_loss(pred1_logits, pred2_logits, lambda_div=10.0):
    """
    Computes KL Divergence for single-channel binary outputs.
    We construct the full distribution [P(bg), P(fg)] for both predictions 
    to use PyTorch's kl_div.
    
    Args:
        pred1_logits: Logits from view 1 (N, 1, D, H, W)
        pred2_logits: Logits from view 2 (N, 1, D, H, W)
    """
    # 1. Construct Log-Probabilities for View 1 (Input to KL Div)
    # log(p) and log(1-p) using log_sigmoid for numerical stability
    log_prob_fg_1 = F.logsigmoid(pred1_logits)      # log(p)
    log_prob_bg_1 = F.logsigmoid(-pred1_logits)     # log(1-p)
    
    # Concatenate to shape (N, 2, D, H, W)
    log_probs_1 = torch.cat([log_prob_bg_1, log_prob_fg_1], dim=1)

    # 2. Construct Probabilities for View 2 (Target for KL Div)
    prob_fg_2 = torch.sigmoid(pred2_logits)         # p
    prob_bg_2 = 1.0 - prob_fg_2                     # 1-p
    
    # Concatenate to shape (N, 2, D, H, W)
    probs_2 = torch.cat([prob_bg_2, prob_fg_2], dim=1)

    # 3. Compute KL Divergence
    # D(P1 || P2) -> We want P1 to be consistent with P2
    # Note: The paper minimizes D(p(y|aug1) || p(y|aug2)) + D(p(y|aug2) || p(y|aug1)) implicitly 
    # or just one direction. PyTorch KLDivLoss is sum(target * (log_target - input)).
    # Standard consistency usually uses the symmetric KL or just one way. 
    # Here we stick to the standard implementation pattern: input=log_probs, target=probs.
    
    kl_loss = F.kl_div(log_probs_1, probs_2, reduction='batchmean')
    
    return lambda_div * kl_loss











def train(dataloader, model, optimizer, epoch, bce_criterion, dice_criterion, augmentor, df, scaler=None):
    ''' 
        dataloader => it is the pytorch dataloader created from a pytorch dataset
        model => it is the actual model where we will give the images for forward propagation
        optimizer => the optimizer to use to weight update
        epoch => the current number of the epoch
        net => the name of the network (string)
        df => an empty daframe containing two columns of train loss and test loss

    '''
    print(f"\n\n========================== Training Epoch {epoch}===================\n\n")
    size = len(dataloader.dataset)
    model.train()
    augmentor.eval()

    log_csv_path = './log/augmentation_weights_log_causality_train_td1_roughness_enforced_5_normalised_gin.csv'
    if not os.path.exists('./log'):
        os.makedirs('./log')

    train_metrics = {"loss":0, "IOU": 0,"precision":0,}
    idx = 0

    #accumulation_step = 8
    


    for batch, (X, y, fname) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        use_gin = np.random.rand() < 0.8
        gin_params = None

        with torch.no_grad():
            # aug_view1, aug_view2 = augmentor(X)
            if use_gin:
                aug_view1 = augmentor(X)
                gin_params = augmentor.get_gin_params()
            else:
                aug_view1 = X

        # save_nifty(X.squeeze().detach().cpu().numpy(), (1.25,1.25,1.25), './gin_output/original.nii.gz')
        # save_nifty(aug_view1.squeeze().detach().cpu().numpy(), (1.25,1.25,1.25), './gin_output/aug_view1.nii.gz')

        
        # original_np = X.squeeze().detach().cpu().numpy()
        # aug_view1_np = aug_view1.squeeze().detach().cpu().numpy()
        # aug_view2_np = aug_view2.squeeze().detach().cpu().numpy()
        # gt_np = y.squeeze().detach().cpu().numpy()

        # save_nifty(original_np, (1.25,1.25,1.25), './gin_output/original_ute.nii.gz')
        # save_nifty(aug_view1_np, (1.25,1.25,1.25), './gin_output/aug_view1_ute.nii.gz')
        # save_nifty(aug_view2_np, (1.25,1.25,1.25), './gin_output/aug_view2_ute.nii.gz')
        # save_nifty(gt_np, (1.25,1.25,1.25), './gin_output/gt_ute.nii.gz')

        # raise RuntimeError("Stop here")
        
        if scaler: # it means use mixed precision
            with torch.cuda.amp.autocast():
                #print(X.shape)

                pred_1 = model(aug_view1)
                pred_2 = model(aug_view2)
                

                # --- C. Calculate Supervised Losses (BCE + Dice) ---
                # Loss for View 1
                loss_bce_1 = bce_criterion(pred_1, y)
                loss_dice_1 = dice_criterion(pred_1, y)
                total_seg_loss_1 = loss_bce_1 + loss_dice_1
                # total_seg_loss_1 = dice_criterion(pred_1, y)
                
                # Loss for View 2
                loss_bce_2 = bce_criterion(pred_2, y)
                loss_dice_2 = dice_criterion(pred_2, y)
                total_seg_loss_2 = loss_bce_2 + loss_dice_2
                
                # --- D. Calculate Consistency Loss (KL Divergence) ---
                # Enforce invariance: Prediction on View 1 should be close to View 2 [cite: 332]
                # We calculate bidirectional consistency for stability
                kl_dist_1_to_2 = binary_kl_consistency_loss(pred_1, pred_2, lambda_div=10.0)
                kl_dist_2_to_1 = binary_kl_consistency_loss(pred_2, pred_1, lambda_div=10.0)
                consistency_loss = (kl_dist_1_to_2 + kl_dist_2_to_1) / 2.0
                
                # --- E. Total Loss ---
                # Paper Eq. 3: Seg Loss(View1) + Seg Loss(View2) + Consistency
                total_loss = total_seg_loss_1 + total_seg_loss_2 + consistency_loss
                # total_loss = total_seg_loss_1


                
            scaler.scale(total_loss).backward()
            scaler.step(optimizer) 
            scaler.update()
            optimizer.zero_grad()

            


            
            

        else: # do not use mixed precision
            
            pred_1 = model(aug_view1)
            # pred_2 = model(aug_view2)
            
            # --- C. Calculate Supervised Losses (BCE + Dice) ---
            # Loss for View 1
            # loss_bce_1 = bce_criterion(pred_1, y)
            loss_dice_1 = dice_criterion(pred_1, y)
            total_seg_loss_1 = loss_dice_1 #loss_bce_1 + loss_dice_1
            # total_seg_loss_1 = dice_criterion(pred_1, y)
            # Loss for View 2
            # loss_bce_2 = bce_criterion(pred_2, y)
            # loss_dice_2 = dice_criterion(pred_2, y)
            # total_seg_loss_2 = loss_bce_2 + loss_dice_2
            
            # --- D. Calculate Consistency Loss (KL Divergence) ---
            # Enforce invariance: Prediction on View 1 should be close to View 2 [cite: 332]
            # We calculate bidirectional consistency for stability
            # kl_dist_1_to_2 = binary_kl_consistency_loss(pred_1, pred_2, lambda_div=10.0)
            # kl_dist_2_to_1 = binary_kl_consistency_loss(pred_2, pred_1, lambda_div=10.0)
            # consistency_loss = (kl_dist_1_to_2 + kl_dist_2_to_1) / 2.0
            
            # --- E. Total Loss ---
            # Paper Eq. 3: Seg Loss(View1) + Seg Loss(View2) + Consistency
            total_loss = total_seg_loss_1 #+ total_seg_loss_2 #+ consistency_loss
            # total_loss = total_seg_loss_1

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if use_gin and gin_params is not None:
            f_name = fname[0] if isinstance(fname, (tuple, list)) else fname
            log_augmentation_weights(log_csv_path, f_name, total_loss.item(), gin_params)

        pred_np = torch.sigmoid(pred_1).squeeze().detach().cpu().numpy()
        label_np = y.squeeze().detach().cpu().numpy()

        pred_np[pred_np > 0.5] = 1
        pred_np[pred_np <= 0.5] = 0
        # save_nifty(pred_np, (1.25,1.25,1.25), './gin_output/pred_mask.nii.gz')

        # raise RuntimeError("Stop here")

        metrics = calculate_iou_precision(pred_np, label_np)

        train_metrics['IOU'] += metrics[0]
        train_metrics['precision'] += metrics[1]

        train_metrics["loss"] += total_loss.item()

        
        idx = idx+1 # is it necessary to use this, we can just simply use the batch variable here
        print(f"\n Epoch {epoch} Batch {idx}/{len(dataloader)} loss: {total_loss.item():.4f} ({train_metrics['loss']/idx:1.4f}) \
            Precision: {metrics[1]:.4f} ({train_metrics['precision']/idx:.5f}) \
            IOU: {metrics[0]:.4f} ({train_metrics['IOU']/idx:.5f})")


        # Empty cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # if idx == 9:
        #     break
    
    df.at[epoch, 'epoch'] = epoch
    df.at[epoch, 'loss'] = train_metrics["loss"] / idx
    df.at[epoch, 'IOU'] = train_metrics['IOU'] / idx
    df.at[epoch, 'precision'] = train_metrics['precision'] / idx

    #loss, current = loss.item(), (batch + 1) * len(X)
    print(f"\n\n\n\n Epoch {epoch} Training Ended \n")
    print('\n\n Current Learning rate is : ', optimizer.param_groups[0]['lr'], '\n\n\n')
    print(f"Epoch {epoch} train loss: {train_metrics['loss']/idx:>7f} \
        Precision: {train_metrics['precision']/idx:>7f} \
        IOU: {train_metrics['IOU']/idx:>7f}")

    print("\n\n\n\n\n")
    

def test(dataloader, model, epoch, bce_criterion, dice_criterion, df, scheduler=None):

    print(f"\n\n\n ======================= Validation Epoch {epoch} ====================\n\n\n")
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    print("Using Sliding Window Inference with ROI size (192, 192, 96)")
    
    model.eval()
    test_loss, correct = 0, 0
    idx = 0
    test_metrics = {"running_loss":0, "IOU": 0,"precision":0,}
    
    
    
    # Initialize single sliding window inferer for both UTE and CT data
    # ROI size (192, 192, 96) matches training crop size for consistency
    # Uses Gaussian blending for smooth transitions between overlapping windows
    # sliding_window_inferer = SlidingWindowInferer(
    #     roi_size=(192, 192, 96),     # Region of Interest size - matches training crop
    #     sw_batch_size=1,             # Process one window at a time
    #     overlap=0.25,                # 25% overlap between windows for smoother results
    #     mode='gaussian',            # Gaussian blending for overlap regions
    #     sigma_scale=0.125,          # Controls Gaussian blending sharpness
    #     padding_mode='constant',     # Pad with zeros at volume borders
    #     cval=0.0                    # Constant value for padding
    # )
    
    log_csv_path = './log/augmentation_weights_log_causality_train_td1_roughness_enforced_5_normalised_gin.csv'
    save_on_best_test_log_csv_path = './log/augmentation_weights_log_causality_train_td1_roughness_enforced_5_normalised_gin_saved_on_best_test.csv'

    with torch.no_grad():
        for X, y, _ in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)#sliding_window_inferer(inputs=X, network=model)
            test_loss = dice_criterion(pred, y).item() #+ bce_criterion(pred, y).item())/2

            pred_np = torch.sigmoid(pred).squeeze().detach().cpu().numpy()
            label_np = y.squeeze().detach().cpu().numpy()

            metrics = calculate_iou_precision(pred_np, label_np)

            test_metrics['IOU'] += metrics[0]
            test_metrics['precision'] += metrics[1]

            test_metrics["running_loss"] += test_loss


            
            idx = idx+1

            


            print(f"Epoch {epoch} Batch {idx}/{len(dataloader)} loss: {test_loss:.4f} ({test_metrics['running_loss']/idx:.5f}) \
                Precision: {metrics[1]:.4f} ({test_metrics['precision']/idx:.5f}) \
                IOU: {metrics[0]:.4f} ({test_metrics['IOU']/idx:.5f})")

            # if idx == 9:
            #     break



        df.at[epoch, 'epoch'] = epoch
        df.at[epoch, 'loss'] = test_metrics["running_loss"] / idx
        df.at[epoch, 'IOU'] = test_metrics['IOU'] / idx
        df.at[epoch, 'precision'] = test_metrics['precision'] / idx

        #final_iou = test_metrics['IOU'] / idx

        print(f"\n\n\n\n Epoch {epoch} Validation Ended \n")
        print(f"Epoch {epoch} loss: {test_metrics['running_loss']/idx:>7f} \
             Test  Precision: {test_metrics['precision']/idx:>7f} Test  IOU: {test_metrics['IOU']/idx:>7f} ")

        print("\n\n\n\n\n")

        val_loss = test_metrics["running_loss"] / idx
        if scheduler:
            scheduler.step(val_loss)

        global global_loss, epoch_since_loss_didnt_improve
        # if final_iou >= global_target_iou:
        #     #print(f"\nLoss improved from {global_loss} to {val_loss} . Saving the current weight\n\n\n\n\n")
        #     print(f"\nFinal IOU improved from {global_target_iou} to {final_iou} . Saving the current weight\n\n\n\n\n")
        #     global_target_iou = final_iou
        #     epoch_since_loss_didnt_improve=0
        #     torch.save(model.state_dict(), './save_models/'+ 'best_ute_swin_HD_loss_pretrained_join_train_12.pth')
        if global_loss >= val_loss:
            print(f"\nLoss improved from {global_loss} to {val_loss} . Saving the current weight\n\n\n\n\n")
            #print(f"\nTargeted IOU improved from {global_target_iou} to {target_iou} . Saving the current weight\n\n\n\n\n")
            global_loss = val_loss
            epoch_since_loss_didnt_improve=0
            torch.save(model.state_dict(), './save_models/'+ 'best_bunet_causality_paper_ct_train_UTE_test_w_tversky_wo_kl_only_gin_roughness_enforced_5_normalised_gin.pth')
            print("Model saved successfully")
            if os.path.exists(log_csv_path):
                shutil.copy(log_csv_path, save_on_best_test_log_csv_path)

        else:
            epoch_since_loss_didnt_improve += 1



def main():

    net = "unet"
    print(net)
    crop_size = (256,128,256)
    training_data = CausalityDataset(csv_file='./ids/only_copd_1.25mm.csv', #causality_train_ute_copd.csv',#only_copd_1.25mm.csv`',   
                               transform = transforms.Compose(
                                [
                                    #RandomCrop(output_size=(192, 192, 96), padval=0.5), 
                                    VariableSpatialFix(num_of_double_stride_conv=4),
                                    ToTensor()
                                ]),
                               training = True,
                               )

    test_data = CausalityDataset(   csv_file='./ids/only_ute_1.25mm.csv',#causality_test_ute_copd.csv',#only_ute`_1.25mm.csv`',
                               transform = transforms.Compose(
                                [ 
                                    VariableSpatialFix(num_of_double_stride_conv=4),
                                    ToTensor()
                                ]),
                               training=False,
                            )


    print(device)


    print("bunet")
    batch_size = 1
    gen = BasicUNet()



        # Check if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

        # Wrap the model with DataParallel
        gen = nn.DataParallel(gen)
    
    gen = gen.to(device)


    model_checkpoint = ''
    if model_checkpoint:
        gen.load_state_dict(torch.load(model_checkpoint))
        # gen.freeze_normalization_layers()
        print("Model Loaded successfully and Normalization layers frozen")

    scaler = None #torch.cuda.amp.GradScaler()

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    print('data loaded')



    optimizer = torch.optim.Adam(gen.parameters(), lr = 1e-3)#(param_groups) #(gen.parameters(), lr = 1e-4)

    scheduler = None #ReduceLROnPlateau(optimizer, mode='min', patience=300, factor=0.1)
    
    print('training data')
    

    epochs = 400
    df_train = pd.DataFrame(columns=['epoch','loss'])
    df_test = pd.DataFrame(columns=['epoch','loss','IOU','precision'])

    # Initialize Losses
    bce_criterion = nn.BCEWithLogitsLoss()
    dice_criterion = TverskyLoss(sigmoid=True)#BinarySoftDiceLoss(smooth=1e-5)
    # loss_fn = TverskyLoss(sigmoid=True)
    augmentor = CausalityAugmentation3D(in_channels=1).to(device)

    for t in range(epochs):

        
        # test(test_dataloader, gen, t, net, df_test, scheduler)
        train(train_dataloader, gen, optimizer, t, bce_criterion, dice_criterion, augmentor, df_train, scaler)

        test(test_dataloader, gen, t, bce_criterion, dice_criterion, df_test, scheduler) # 
        
        # df_train.at[t, 'lr'] = optimizer.param_groups[0]['lr']
        # df_test.at[t, 'lr'] = optimizer.param_groups[0]['lr']
        
        df_train.to_csv('./log/train_metrics_' + net + '_causality_paper_ct_train_UTE_test_w_tversky_wo_kl_only_gin_td1_roughness_enforced_5_normalised_gin.csv', index=False)
        df_test.to_csv('./log/test_metrics_' + net + '_causality_papery_ct_train_UTE_test_w_tversky_wo_kl_only_gin_td1_roughness_enforced_5_normalised_gin.csv', index=False)
        
        sys.stdout.flush()

        
        if epoch_since_loss_didnt_improve > 100:
            print(f"\n\n\nIt has been more than {epoch_since_loss_didnt_improve} epoch since the Validation loss improved. So, terminating training.\n\n\n")
            #torch.save(gen.state_dict(), './save_models/'+ f'epoch_{t}_ute_HD_loss_swin_join_train_12.pth')
            break

    print("Done!")

    print('Saved Pytorch model to model.pth')

if __name__=="__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    main()



