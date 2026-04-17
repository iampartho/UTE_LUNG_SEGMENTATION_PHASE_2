import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
from monai.losses import TverskyLoss
from monai.inferers import SlidingWindowInferer
from datetime import date
from datasets_causality import CausalityDataset, ToTensor, VariableSpatialFix, RandomCrop
import argparse
import itertools
from basic_unet_disentagled import BasicUNet

# Global variables for tracking progress
global_loss = np.inf
epoch_since_loss_didnt_improve = 0

def calculate_iou_precision(pred_masks, true_masks, threshold=0.5):
    pred_masks_binary = (pred_masks > threshold).astype(np.uint8)
    true_masks_binary = (true_masks > threshold).astype(np.uint8)
    
    intersection = np.sum(np.logical_and(pred_masks_binary, true_masks_binary))
    union = np.sum(np.logical_or(pred_masks_binary, true_masks_binary))
    
    iou = intersection / union if union > 0 else 0.0
    precision = intersection / (np.sum(pred_masks_binary) + 1e-6)
    
    return [iou, precision]

def train(dataloader, model, loss_fn, optimizer, epoch, net, df, scaler=None):
    print(f"\n\n========================== Training Epoch {epoch} ===================\n\n")
    model.train()
    
    
    train_metrics = {"running_loss": 0, "IOU": 0, "precision": 0}
    idx = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        

        optimizer.zero_grad()
        
        

        pred = model(X)
        loss = loss_fn(pred, y)
        
        loss.backward()
        optimizer.step()

        # Metrics
        pred_np = torch.sigmoid(pred).squeeze().detach().cpu().numpy()
        label_np = y.squeeze().detach().cpu().numpy()

        iou, precision = calculate_iou_precision(pred_np, label_np)

        train_metrics['IOU'] += iou
        train_metrics['precision'] += precision
        train_metrics["running_loss"] += loss.item()
        
        idx += 1 
        print(f"\r Epoch {epoch} Batch {idx}/{len(dataloader)} loss: {loss.item():.4f} ({train_metrics['running_loss']/idx:1.4f}) \
            Precision: {precision:.4f} ({train_metrics['precision']/idx:.5f}) \
            IOU: {iou:.4f} ({train_metrics['IOU']/idx:.5f})", end="")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    df.at[epoch, 'epoch'] = epoch
    df.at[epoch, 'loss'] = train_metrics["running_loss"] / idx
    df.at[epoch, 'IOU'] = train_metrics['IOU'] / idx
    df.at[epoch, 'precision'] = train_metrics['precision'] / idx

    print(f"\n\n Epoch {epoch} Training Ended \n")
    print('Current Learning rate: ', optimizer.param_groups[0]['lr'])
    print(f"Epoch {epoch} train loss: {train_metrics['running_loss']/idx:>7f} \
        Train Precision: {train_metrics['precision'] / idx:.5f} Train IOU: {train_metrics['IOU'] / idx:.5f} ")

def test(dataloader, model, loss_fn, epoch, df, scheduler=None):
    print(f"\n\n\n ======================= Validation Epoch {epoch} ====================\n\n\n")
    model.eval()
    test_metrics = {"running_loss": 0, "IOU": 0, "precision": 0}
    idx = 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y).item()

            pred_np = torch.sigmoid(pred).squeeze().detach().cpu().numpy()
            label_np = y.squeeze().detach().cpu().numpy()

            iou, precision = calculate_iou_precision(pred_np, label_np)

            test_metrics['IOU'] += iou
            test_metrics['precision'] += precision
            test_metrics["running_loss"] += loss
            
            idx += 1
            print(f"\r Epoch {epoch} Batch {idx}/{len(dataloader)} loss: {loss:.4f} ({test_metrics['running_loss']/idx:.5f}) \
                Precision: {precision:.4f} ({test_metrics['precision']/idx:.5f}) \
                IOU: {iou:.4f} ({test_metrics['IOU']/idx:.5f})", end="")

        df.at[epoch, 'epoch'] = epoch
        df.at[epoch, 'loss'] = test_metrics["running_loss"] / idx
        df.at[epoch, 'IOU'] = test_metrics['IOU'] / idx
        df.at[epoch, 'precision'] = test_metrics['precision'] / idx

        print(f"\n\n Epoch {epoch} Validation Ended \n")
        val_loss = test_metrics["running_loss"] / idx
        print(f"Epoch {epoch} loss: {val_loss:>7f} \
             Test Precision: {test_metrics['precision']/idx:>7f} Test IOU: {test_metrics['IOU']/idx:>7f} ")

        if scheduler:
            scheduler.step(val_loss)

        global global_loss, epoch_since_loss_didnt_improve
        if global_loss >= val_loss:
            print(f"\nLoss improved from {global_loss} to {val_loss}. Saving model...\n")
            global_loss = val_loss
            epoch_since_loss_didnt_improve = 0
            torch.save(model.state_dict(), './save_models/best_unet_td1.pth')
            print("Model saved successfully")
        else:
            epoch_since_loss_didnt_improve += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, default="bunet", help="network")
    args = parser.parse_args()
    net = args.net
    print(f"Network: {net}")

    # Dataset setup
    training_data = CausalityDataset(csv_file='./ids/only_copd_1.25mm.csv',
                                    transform=transforms.Compose([
                                        VariableSpatialFix(num_of_double_stride_conv=4),
                                        ToTensor()
                                    ]),
                                    training=True)

    test_data = CausalityDataset(csv_file='./ids/only_ute_1.25mm.csv',
                                transform=transforms.Compose([
                                    VariableSpatialFix(num_of_double_stride_conv=4),
                                    ToTensor()
                                ]),
                                training=False)

    batch_size = 1
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    print('Data loaded')

    # Model setup
    model = BasicUNet()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    
    loss_fn = TverskyLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 200
    df_train = pd.DataFrame(columns=['epoch', 'loss', 'IOU', 'precision'])
    df_test = pd.DataFrame(columns=['epoch', 'loss', 'IOU', 'precision'])

    if not os.path.exists('./save_models'):
        os.makedirs('./save_models')
    if not os.path.exists('./log'):
        os.makedirs('./log')

    for t in range(epochs):
        train(train_dataloader, model, loss_fn, optimizer, t, net, df_train)
        test(test_dataloader, model, loss_fn, t, df_test)
        
        df_train.to_csv(f'./log/train_metrics_{net}_td1.csv', index=False)
        df_test.to_csv(f'./log/test_metrics_{net}_td1.csv', index=False)
        
        sys.stdout.flush()

        if epoch_since_loss_didnt_improve > 100:
            print(f"\nEarly stopping at epoch {t}")
            break

    print("Training Done!")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
