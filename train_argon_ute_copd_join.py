import os
import sys
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets
from torchvision.transforms import ToTensor
#from monai.networks.nets import UNETR
from monai.losses import DiceLoss, DiceCELoss, TverskyLoss,GeneralizedDiceLoss, DiceFocalLoss, GeneralizedDiceFocalLoss, HausdorffDTLoss
from monai.inferers import SlidingWindowInferer
#from monai import metrics
from datetime import date
today = date.today()
from helper_ute_copd import UTEDataset, ToTensor, VariableSpatialFix, RandomCrop
import argparse

# from causality_paper_augmentation_2 import CausalityAugmentation3D
from gin_with_log_capability import CausalityAugmentation3D
import itertools


#For Mixed Precision

#from GDFloss import GeneralizedDiceFocalLoss
import torch
from torch.cuda.amp import autocast


# For Data Parallel
import torch.nn.parallel


#from unet_ute import UNet
# from swin import SwinUNETR
from basic_unet_disentagled import BasicUNet


#from unet_2_dec import UNet
import csv



global_loss = np.inf
#global_target_iou = 0.5813
epoch_since_loss_didnt_improve = 0


def calculate_iou_precision(pred_masks_ute, true_masks_ute, pred_masks_copd, true_masks_copd, threshold=0.5):
    # Convert prediction masks to binary masks based on threshold
    pred_masks_binary_ute = (pred_masks_ute > threshold).astype(np.uint8)
    pred_masks_binary_copd = (pred_masks_copd > threshold).astype(np.uint8)
    true_masks_binary_ute = (true_masks_ute > threshold).astype(np.uint8)
    true_masks_binary_copd = (true_masks_copd > threshold).astype(np.uint8)
    

    # Initialize lists to store IoU and Precision values for each sample
    iou_list = []
    precision_list = []

    # Iterate over each sample in the batch (not necessary for single sample)
    #for pred_mask_ute, true_mask_ute, pred_mask_copd, true_mask_copd in zip(pred_masks_binary_ute, true_masks_binary_ute, pred_masks_binary_copd, true_masks_binary_copd):
        # Calculate intersection and union
    intersection_ute = np.sum(np.logical_and(pred_masks_binary_ute, true_masks_binary_ute))
    union_ute = np.sum(np.logical_or(pred_masks_binary_ute, true_masks_binary_ute))
    intersection_copd = np.sum(np.logical_and(pred_masks_binary_copd, true_masks_binary_copd))
    union_copd = np.sum(np.logical_or(pred_masks_binary_copd, true_masks_binary_copd))
        
    # Calculate IoU
    iou_ute = intersection_ute / union_ute if union_ute > 0 else 0.0
    iou_copd = intersection_copd / union_copd if union_copd > 0 else 0.0
    iou_list.append(iou_ute)
    iou_list.append(iou_copd)

    # Calculate Precision
    precision_ute = intersection_ute / (np.sum(pred_masks_binary_ute) + 1e-6)  # Add epsilon to avoid division by zero
    precision_copd = intersection_copd / (np.sum(pred_masks_binary_copd) + 1e-6)  # Add epsilon to avoid division by zero
    precision_list.append(precision_ute)
    precision_list.append(precision_copd)


    # Calculate average IoU and Precision across the batch
    avg_iou = np.mean(iou_list)
    avg_precision = np.mean(precision_list)
    
    
    return [avg_iou, avg_precision]



def log_augmentation_weights(csv_path, filename, loss_value, params):
    """
    Logs augmentation parameters if the new loss is lower than the existing record.
    Structure: filename, kernel_size_0, kernel_0, ..., kernel_size_n, kernel_n, loss_value
    """
    
    # Flatten weights to string for CSV storage
    # Note: Kernels can be large, this might create very large CSVs.
    row_data = {'filename': filename}
    for i, p in enumerate(params):
        if i == p['idx']:
            row_data[f'kernel_size_{i}'] = p['k']
            # Convert tensor to flat list string to fit in one CSV cell
            # Using numpy to summarize or list to store full data
            row_data[f'kernel_{i}'] = p['w'].numpy().flatten().tolist()
    
    row_data['loss_value'] = float(loss_value)
    
    # Define columns
    columns = ['filename']
    for i in range(len(params)):
        columns.append(f'kernel_size_{i}')
        columns.append(f'kernel_{i}')
    columns.append('loss_value')

    # Load existing log if it exists
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(columns=columns)

    # Check logic
    if filename in df['filename'].values:
        # Get existing loss
        existing_loss = df.loc[df['filename'] == filename, 'loss_value'].values[0]
        
        if loss_value < existing_loss:
            # Update row
            # We iterate columns to ensure we update everything correctly
            idx = df.index[df['filename'] == filename].tolist()[0]
            for col, val in row_data.items():
                # Convert list to string representation for storage if it's a list
                if isinstance(val, list):
                    df.at[idx, col] = str(val)
                else:
                    df.at[idx, col] = val
            df.to_csv(csv_path, index=False)
    else:
        # Insert new row
        # Convert lists to strings for the dataframe
        row_ready = {}
        for k, v in row_data.items():
            row_ready[k] = str(v) if isinstance(v, list) else v
            
        new_row_df = pd.DataFrame([row_ready])
        # Concatenate and save
        df = pd.concat([df, new_row_df], ignore_index=True)
        df.to_csv(csv_path, index=False)











def train(dataloader, model, augmentor, loss_fn, optimizer, epoch, net, df, scaler=None):
    print(f"\n\n========================== Training Epoch {epoch} ===================\n\n")
    size = len(dataloader.dataset)
    model.train()
    augmentor.eval() # GIN is random conv, no training needed
    
    # Path for the weights log
    log_csv_path = './log/augmentation_weights_log_on_ute.csv'
    if not os.path.exists('./log'):
        os.makedirs('./log')
    
    train_metrics = {"running_loss1":0,"running_loss2":0, "IOU": 0,"precision":0}
    idx = 0

    # Modified unpacking to include filenames
    # Assuming dataloader now yields: X_ute, y_ute, X_copd, y_copd, fname_ute, fname_copd
    for batch, (X_ute, y_ute, X_copd, y_copd, fname_ute, fname_copd) in enumerate(dataloader):
        X_ute, y_ute, X_copd, y_copd = X_ute.to(device), y_ute.to(device), X_copd.to(device), y_copd.to(device)

        use_gin = np.random.random() < 0.8

        # ---------------------
        # Train Segmentation Model
        # ---------------------
        optimizer.zero_grad()
        
        # Temp variables to store params if GIN is used
        gin_params_ute = None
        gin_params_copd = None

        if use_gin:
            with torch.no_grad():
                # Forward UTE
                X_ute_aug = augmentor(X_ute)
                # Capture params immediately after forward pass for UTE
                # Note: We must clone/copy because the next forward pass (COPD) will overwrite internal state
                # but since we modified GIN to return detached CPU tensors, we just grab them now.
                gin_params_ute = augmentor.get_gin_params()

                # Forward COPD
                # X_copd_aug = augmentor(X_copd)
                # Capture params for COPD
                # gin_params_copd = augmentor.get_gin_params()
        else:
            X_ute_aug = X_ute
        
        X_copd_aug = X_copd

        pred_ute = model(X_ute_aug)
        loss_seg_ute = loss_fn(pred_ute, y_ute)
        
        pred_copd = model(X_copd_aug)
        loss_seg_copd = loss_fn(pred_copd, y_copd)
        
        total_loss = loss_seg_ute + loss_seg_copd
        total_loss.backward()
        optimizer.step()

        # --- Logging Block ---
        if use_gin:
            # Handle batch size > 1:
            # fname_ute is likely a tuple or list. We iterate.
            # However, GIN implementation usually applies different random weights per batch item.
            # Our `get_gin_params` returns the whole weight tensor [Batch*Ch_out, ...].
            # For simplicity, assuming Batch Size = 1 as per your main() function:
            
            # Extract single filename string if it's a tuple/list
            f_ute = fname_ute[0] if isinstance(fname_ute, (tuple, list)) else fname_ute
            #f_copd = fname_copd[0] if isinstance(fname_copd, (tuple, list)) else fname_copd
            
            # Log UTE
            log_augmentation_weights(log_csv_path, f_ute, loss_seg_ute.item(), gin_params_ute)
            
            # Log COPD
            #log_augmentation_weights(log_csv_path, f_copd, loss_seg_copd.item(), gin_params_copd)
        # ---------------------

        # Metrics
        loss1 = loss_seg_ute
        loss2 = loss_seg_copd
        
        pred_np_ute = torch.sigmoid(pred_ute).squeeze().detach().cpu().numpy()
        label_np_ute = y_ute.squeeze().detach().cpu().numpy()

        pred_np_copd = torch.sigmoid(pred_copd).squeeze().detach().cpu().numpy()
        label_np_copd = y_copd.squeeze().detach().cpu().numpy()

        metrics = calculate_iou_precision(pred_np_ute, label_np_ute, pred_np_copd, label_np_copd)

        train_metrics['IOU'] += metrics[0]
        train_metrics['precision'] += metrics[1]
        train_metrics["running_loss1"] += loss1.item()
        train_metrics["running_loss2"] += loss2.item()
        
        idx = idx+1 
        print(f"\n Epoch {epoch} Batch {idx}/{len(dataloader)} loss1: {loss1.item():.4f} ({train_metrics['running_loss1']/idx:1.4f}) \
            loss2: {loss2.item():.4f} ({train_metrics['running_loss2']/idx:1.4f}) \
            Precision: {metrics[1]:.4f} ({train_metrics['precision']/idx:.5f}) \
            IOU: {metrics[0]:.4f} ({train_metrics['IOU']/idx:.5f})")

        # Empty cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ... (Rest of function remains same) ...
    df.at[epoch, 'epoch'] = epoch
    df.at[epoch, 'loss1'] = train_metrics["running_loss1"] / idx
    df.at[epoch, 'loss2'] = train_metrics["running_loss2"] / idx
    df.at[epoch, 'IOU'] = train_metrics['IOU'] / idx
    df.at[epoch, 'precision'] = train_metrics['precision'] / idx


    print(f"\n\n\n\n Epoch {epoch} Training Ended \n")
    print('\n\n Current Learning rate is : ', optimizer.param_groups[0]['lr'], '\n\n\n')
    print(f"Epoch {epoch} train loss1: {train_metrics['running_loss1']/idx:>7f} loss2: {train_metrics['running_loss2']/idx:>7f} \
        Train Precision: {train_metrics['precision'] / idx:.5f} Train IOU: {train_metrics['IOU'] / idx:.5f} ")

    print("\n\n\n\n\n")
    

def test(dataloader, model, loss_fn, epoch, augmentor, df, scheduler=None):

    print(f"\n\n\n ======================= Validation Epoch {epoch} ====================\n\n\n")
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    print("Using Sliding Window Inference with ROI size (192, 192, 96)")
    
    model.eval()
    test_loss, correct = 0, 0
    idx = 0
    test_metrics = {"running_loss1":0, "running_loss2":0, "IOU": 0,"precision":0,}
    
    
    
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
    
    with torch.no_grad():
        for X_ute, y_ute, X_copd, y_copd in dataloader:
            X_ute, y_ute, X_copd, y_copd = X_ute.to(device), y_ute.to(device), X_copd.to(device), y_copd.to(device)

            pred_ute = model(X_ute) #sliding_window_inferer(inputs=X_ute, network=model)
            test_loss1 = loss_fn(pred_ute, y_ute).item()
            pred_copd = model(X_copd) #sliding_window_inferer(inputs=X_copd, network=model)
            test_loss2 = loss_fn(pred_copd, y_copd).item()

            pred_np_ute = torch.sigmoid(pred_ute).squeeze().detach().cpu().numpy()
            label_np_ute = y_ute.squeeze().detach().cpu().numpy()

            pred_np_copd = torch.sigmoid(pred_copd).squeeze().detach().cpu().numpy()
            label_np_copd = y_copd.squeeze().detach().cpu().numpy()

            metrics = calculate_iou_precision(pred_np_ute, label_np_ute, pred_np_copd, label_np_copd)


            test_metrics['IOU'] += metrics[0]
            test_metrics['precision'] += metrics[1]


            test_metrics["running_loss1"] += test_loss1
            test_metrics["running_loss2"] += test_loss2


            
            idx = idx+1

            


            print(f"Epoch {epoch} Batch {idx}/{len(dataloader)} loss1: {test_loss1:.4f} ({test_metrics['running_loss1']/idx:.5f}) \
                loss2: {test_loss2:.4f} ({test_metrics['running_loss2']/idx:.5f}) \
                Precision: {metrics[1]:.4f} ({test_metrics['precision']/idx:.5f}) \
                IOU: {metrics[0]:.4f} ({test_metrics['IOU']/idx:.5f})")

            # if idx == 9:
            #     break



        df.at[epoch, 'epoch'] = epoch
        df.at[epoch, 'loss1'] = test_metrics["running_loss1"] / idx
        df.at[epoch, 'loss2'] = test_metrics["running_loss2"] / idx
        df.at[epoch, 'IOU'] = test_metrics['IOU'] / idx
        df.at[epoch, 'precision'] = test_metrics['precision'] / idx

        #final_iou = test_metrics['IOU'] / idx

        print(f"\n\n\n\n Epoch {epoch} Validation Ended \n")
        print(f"Epoch {epoch} loss1: {test_metrics['running_loss1']/idx:>7f} \
             loss2: {test_metrics['running_loss2']/idx:>7f} \
             Test  Precision: {test_metrics['precision']/idx:>7f} Test  IOU: {test_metrics['IOU']/idx:>7f} ")

        print("\n\n\n\n\n")

        val_loss = test_metrics["running_loss1"] / idx #(test_metrics["running_loss1"] / idx + test_metrics["running_loss2"] / idx) / 2
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
            torch.save(model.state_dict(), './save_models/'+ 'best_bunet_joint_train_with_gin_on_ute_with_logging.pth')
            print("Model saved successfully")
            

        else:
            epoch_since_loss_didnt_improve += 1



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type = str, default="bunet",help = "network")

    args = parser.parse_args()
    net = args.net


    #net = "unet"
    print(net)
    crop_size = (256,128,256)
    training_data = UTEDataset(ute_csv_file='./ids/ute_train.csv',
                               copd_csv_file='./ids/copd_train_1.25mm.csv',
                               transform = transforms.Compose(
                                [
                                    VariableSpatialFix(num_of_double_stride_conv=4),#RandomCrop(output_size=(192, 192, 96), padval=0.5), 
                                    ToTensor()
                                ]),
                               training = True,
                               )

    test_data = UTEDataset(ute_csv_file='./ids/ute_test.csv',
                               copd_csv_file='./ids/copd_test_1.25mm.csv',
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
    augmentor = CausalityAugmentation3D().to(device)

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    print('data loaded')


    #loss_fn = GeneralizedDiceFocalLoss(sigmoid=True)
    loss_fn = TverskyLoss(sigmoid=True)
    
    # Optimizer for Segmentation Model
    optimizer = torch.optim.Adam(gen.parameters(), lr = 1e-3)
    
    scheduler = None #ReduceLROnPlateau(optimizer, mode='min', patience=300, factor=0.1)
    
    print('training data')
    

    epochs = 200
    df_train = pd.DataFrame(columns=['epoch','loss1','loss2', 'IOU','precision'])
    df_test = pd.DataFrame(columns=['epoch','loss1','loss2', 'IOU','precision'])
    for t in range(epochs):

        
        # test(test_dataloader, gen, loss_fn, t, net, df_test, scheduler)
        train(train_dataloader, gen, augmentor, loss_fn, optimizer, t, net, df_train, scaler)

        test(test_dataloader, gen, loss_fn, t, augmentor, df_test, scheduler) # 
        
        # df_train.at[t, 'lr'] = optimizer.param_groups[0]['lr']
        # df_test.at[t, 'lr'] = optimizer.param_groups[0]['lr']
        
        df_train.to_csv('./log/train_metrics_' + net + '_joint_train_with_gin_on_ute_with_logging.csv', index=False)
        df_test.to_csv('./log/test_metrics_' + net + '_joint_train_with_gin_on_ute_with_logging.csv', index=False)
        
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
