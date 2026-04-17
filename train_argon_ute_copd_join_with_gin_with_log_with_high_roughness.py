import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from monai.losses import TverskyLoss
from datetime import date
from helper_ute_copd import UTEDataset, ToTensor, VariableSpatialFix
import argparse
import shutil
from gin_with_log_capability import CausalityAugmentation3D
from basic_unet_disentagled import BasicUNet

global_loss = np.inf
global_train_loss = np.inf
epoch_since_loss_didnt_improve = 0

def calculate_iou_precision(pred_masks_ute, true_masks_ute, pred_masks_copd, true_masks_copd, threshold=0.5):
    pred_masks_binary_ute = (pred_masks_ute > threshold).astype(np.uint8)
    pred_masks_binary_copd = (pred_masks_copd > threshold).astype(np.uint8)
    true_masks_binary_ute = (true_masks_ute > threshold).astype(np.uint8)
    true_masks_binary_copd = (true_masks_copd > threshold).astype(np.uint8)
    
    iou_list = []
    precision_list = []

    intersection_ute = np.sum(np.logical_and(pred_masks_binary_ute, true_masks_binary_ute))
    union_ute = np.sum(np.logical_or(pred_masks_binary_ute, true_masks_binary_ute))
    intersection_copd = np.sum(np.logical_and(pred_masks_binary_copd, true_masks_binary_copd))
    union_copd = np.sum(np.logical_or(pred_masks_binary_copd, true_masks_binary_copd))
        
    iou_ute = intersection_ute / union_ute if union_ute > 0 else 0.0
    iou_copd = intersection_copd / union_copd if union_copd > 0 else 0.0
    iou_list.append(iou_ute)
    iou_list.append(iou_copd)

    precision_ute = intersection_ute / (np.sum(pred_masks_binary_ute) + 1e-6)
    precision_copd = intersection_copd / (np.sum(pred_masks_binary_copd) + 1e-6)
    precision_list.append(precision_ute)
    precision_list.append(precision_copd)

    return [np.mean(iou_list), np.mean(precision_list)]

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

def train(dataloader, model, augmentor, loss_fn, optimizer, epoch, net, df, scaler=None):
    print(f"\n\n========================== Training Epoch {epoch} ===================\n\n")
    model.train()
    augmentor.eval() 
    
    log_csv_path = './log/augmentation_weights_log_on_ute_with_roughness_enforced_td3.csv'
    if not os.path.exists('./log'):
        os.makedirs('./log')
    
    train_metrics = {"running_loss1":0,"running_loss2":0, "IOU": 0,"precision":0}
    idx = 0

    for batch, (X_ute, y_ute, X_copd, y_copd, fname_ute, fname_copd) in enumerate(dataloader):
        X_ute, y_ute, X_copd, y_copd = X_ute.to(device), y_ute.to(device), X_copd.to(device), y_copd.to(device)

        use_gin = np.random.random() < 0.8
        
        optimizer.zero_grad()
        gin_params_ute = None

        if use_gin:
            with torch.no_grad():
                # Augmentor natively applies exact target roughness per layer now!
                X_ute_aug = augmentor(X_ute)
                gin_params_ute = augmentor.get_gin_params()
                X_copd_aug = augmentor(X_copd)
                gin_params_copd = augmentor.get_gin_params()
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

        if use_gin:
            f_ute = fname_ute[0] if isinstance(fname_ute, (tuple, list)) else fname_ute
            f_copd = fname_copd[0] if isinstance(fname_copd, (tuple, list)) else fname_copd
            log_augmentation_weights(log_csv_path, f_ute, loss_seg_ute.item(), gin_params_ute)
            log_augmentation_weights(log_csv_path, f_copd, loss_seg_copd.item(), gin_params_copd)
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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
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

    train_loss = train_metrics["running_loss1"] / idx
    global global_train_loss, epoch_since_loss_didnt_improve
    if global_train_loss >= train_loss:
        print(f"\nLoss improved from {global_train_loss} to {train_loss} . Saving the current weight\n\n\n\n\n")
        global_train_loss = train_loss
        epoch_since_loss_didnt_improve=0
        torch.save(model.state_dict(), './save_models/'+ 'best_train_bunet_joint_train_with_gin_on_ute_with_logging_roughness_enforced_td3.pth')
        print("Model saved successfully")
    else:
        epoch_since_loss_didnt_improve += 1

def test(dataloader, model, loss_fn, epoch, augmentor, df, scheduler=None):
    print(f"\n\n\n ======================= Validation Epoch {epoch} ====================\n\n\n")
    model.eval()
    test_metrics = {"running_loss1":0, "running_loss2":0, "IOU": 0,"precision":0,}
    idx = 0

    log_csv_path = './log/augmentation_weights_log_on_ute_with_roughness_enforced_td3.csv'
    save_on_best_test_log_csv_path = './log/augmentation_weights_log_on_ute_with_roughness_enforced_saved_on_best_test_td3.csv'
    
    with torch.no_grad():
        for X_ute, y_ute, X_copd, y_copd in dataloader:
            X_ute, y_ute, X_copd, y_copd = X_ute.to(device), y_ute.to(device), X_copd.to(device), y_copd.to(device)

            pred_ute = model(X_ute)
            test_loss1 = loss_fn(pred_ute, y_ute).item()
            pred_copd = model(X_copd)
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

        df.at[epoch, 'epoch'] = epoch
        df.at[epoch, 'loss1'] = test_metrics["running_loss1"] / idx
        df.at[epoch, 'loss2'] = test_metrics["running_loss2"] / idx
        df.at[epoch, 'IOU'] = test_metrics['IOU'] / idx
        df.at[epoch, 'precision'] = test_metrics['precision'] / idx

        print(f"\n\n\n\n Epoch {epoch} Validation Ended \n")
        print(f"Epoch {epoch} loss1: {test_metrics['running_loss1']/idx:>7f} \
             loss2: {test_metrics['running_loss2']/idx:>7f} \
             Test  Precision: {test_metrics['precision']/idx:>7f} Test  IOU: {test_metrics['IOU']/idx:>7f} ")
        print("\n\n\n\n\n")

        val_loss = test_metrics["running_loss1"] / idx 
        if scheduler:
            scheduler.step(val_loss)

        global global_loss, epoch_since_loss_didnt_improve
        if global_loss >= val_loss:
            print(f"\nLoss improved from {global_loss} to {val_loss} . Saving the current weight\n\n\n\n\n")
            global_loss = val_loss
            epoch_since_loss_didnt_improve=0
            torch.save(model.state_dict(), './save_models/'+ 'best_test_bunet_joint_train_with_gin_on_ute_with_logging_roughness_enforced_td3.pth')
            print("Model saved successfully")
            shutil.copy(log_csv_path, save_on_best_test_log_csv_path)
        else:
            epoch_since_loss_didnt_improve += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type = str, default="bunet",help = "network")
    args = parser.parse_args()
    net = args.net

    print(net)
    training_data = UTEDataset(ute_csv_file='./ids/UTE_MRI_previous_numpy.csv',
                               copd_csv_file='./ids/copd_train_1.25mm.csv',
                               transform = transforms.Compose([
                                    VariableSpatialFix(num_of_double_stride_conv=4),
                                    ToTensor()
                               ]),
                               training = True)

    test_data = UTEDataset(ute_csv_file='./ids/only_ute_1.25mm.csv',
                               copd_csv_file='./ids/copd_test_1.25mm.csv',
                               transform = transforms.Compose([ 
                                    VariableSpatialFix(num_of_double_stride_conv=4),
                                    ToTensor()
                               ]),
                               training=False)

    print(device)
    batch_size = 1
    gen = BasicUNet()

    if torch.cuda.device_count() > 1:
        gen = torch.nn.DataParallel(gen)
    
    gen = gen.to(device)

    augmentor = CausalityAugmentation3D().to(device)

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    loss_fn = TverskyLoss(sigmoid=True)
    optimizer = torch.optim.Adam(gen.parameters(), lr = 1e-3)
    scheduler = None 
    
    epochs = 200
    df_train = pd.DataFrame(columns=['epoch','loss1','loss2', 'IOU','precision'])
    df_test = pd.DataFrame(columns=['epoch','loss1','loss2', 'IOU','precision'])
    
    for t in range(epochs):
        train(train_dataloader, gen, augmentor, loss_fn, optimizer, t, net, df_train, None)
        test(test_dataloader, gen, loss_fn, t, augmentor, df_test, scheduler)
        
        df_train.to_csv('./log/train_metrics_' + net + '_joint_train_with_gin_on_ute_with_logging_roughness_enforced_td3.csv', index=False)
        df_test.to_csv('./log/test_metrics_' + net + '_joint_train_with_gin_on_ute_with_logging_roughness_enforced_td3.csv', index=False)
        
        sys.stdout.flush()
        
        if epoch_since_loss_didnt_improve > 100:
            print(f"\n\n\nIt has been more than {epoch_since_loss_didnt_improve} epoch since the Validation loss improved. So, terminating training.\n\n\n")
            break

    print("Done!")

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()