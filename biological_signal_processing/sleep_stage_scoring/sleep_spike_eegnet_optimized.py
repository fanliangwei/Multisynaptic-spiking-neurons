#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Implementing Spiking EEGNet for sleep stage scoring
"""

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import scipy
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader
from collections import OrderedDict
from timm.utils import update_summary
from sklearn.metrics import confusion_matrix

from EEGNet import SpikeEEGNetModel

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(42)

# Select device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Sleep EEG data classification training program')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./eeg_sleep_data', help='Data directory path')
    
    # Model hyperparameters
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--fold', type=int, default=10, help='Number of cross-validation folds')
    parser.add_argument('--classes', type=int, default=5, help='Number of classes')
    
    # Model parameters
    parser.add_argument('--chans', type=int, default=1, help='Number of channels')
    parser.add_argument('--time_points', type=int, default=30, help='Number of time points')
    parser.add_argument('--pk2', type=int, default=100, help='Size of the pooling layer')
    parser.add_argument('--activation_type', type=str, default='MSF', choices=['LIF', 'MSF'], help='Activation function type: LIF or MSF')
    parser.add_argument('--surro_gate', type=str, default='rectangular', choices=['rectangular', 'sigmoid', 'atan', 'gaussian'], help='Surrogate gate type for MSF activation function')
    
    # Output parameters
    parser.add_argument('--save_dir', type=str, default='./output/', help='Model save directory')
    parser.add_argument('--model_name', type=str, default=None, 
                        help='Model name, auto-generated if not specified')
    
    args = parser.parse_args()
    
    # Auto-generate model name if not specified
    if args.model_name is None:
        args.model_name = f'spike_eegnet_epoch{args.epochs}-fold{args.fold}-{args.activation_type}'
    
    return args


def plot_confusion_matrix(y_pred, y_true, classes, savefile=None):
    """
    Plot confusion matrix
    
    Parameters:
        y_pred: Predicted labels
        y_true: True labels
        classes: List of classes
        savefile: Path to save the figure
    """
    cf_matrix = confusion_matrix(y_true, y_pred)
    cf_matrix = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]

    df_cm = pd.DataFrame(cf_matrix*100., index=classes, columns=classes)

    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='.2f')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    if savefile:
        plt.savefig(savefile)
    plt.show()


def test(model, test_loader=None, criterion=None):
    """
    Test model performance
    
    Parameters:
        model: Model
        test_loader: Test data loader
        criterion: Loss function
    
    Returns:
        test_acc: Test accuracy
        all_predicted: All predicted labels
        test_loss_avg: Average test loss
    """
    test_acc = 0.
    sum_sample = 0.
    test_loss_sum = 0.
    model.eval()
    all_predicted = []
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.float().to(device)
            labels = labels.view((-1)).long().to(device)
            predictions = model(images)
            
            _, predicted = torch.max(predictions.data, 1)
            test_loss = criterion(predictions, labels)
            test_loss_sum += test_loss.item()
            
            labels = labels.cpu()
            predicted = predicted.cpu().t()
            all_predicted.append(predicted)
            test_acc += (predicted == labels).sum()
            sum_sample += predicted.numel()
            
    all_predicted = np.concatenate(all_predicted)
    return test_acc.data.cpu().numpy()/sum_sample, all_predicted, test_loss_sum / len(test_loader)


def trans_data(data):
    """
    Convert data to N*C*T format and extract labels
      
    Returns:
        Sleepmat: Processed data
        labels: Corresponding labels
    """
    Sleepmat = data['epoch_data'].squeeze()     
    labels = np.array(data['labels']).squeeze()
    useless_num = sum(labels==-1)
    useful_idx = np.where(labels!=-1)[0].squeeze()

    Sleepmat = Sleepmat[useful_idx]
    labels = labels[useful_idx]
    return Sleepmat, labels


def train(model, epochs, criterion, optimizer, scheduler=None, 
          train_loader=None, test_loader=None, fold=0, save_path=None, model_name=None):
    """
    Train model
    
    Parameters:
        model: Model
        epochs: Number of training epochs
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        train_loader: Training data loader
        test_loader: Testing data loader
        fold: Current cross-validation fold
        save_path: Save path
        model_name: Model name
    
    Returns:
        acc_list: Dictionary recording training and validation accuracies
    """
    best_acc = 0
    
    # Create save directory
    fold_dir = f'{model_name}/fold{fold}/'
    save_dir = os.path.join(save_path, fold_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f'>>>>>>>>>>>>>>>>Data will be saved to {save_dir}')
    
    # Initialize training records
    acc_list = {'train_acc': np.zeros(epochs), 'test_acc': np.zeros(epochs)}
    
    for epoch in range(epochs):
        train_acc = 0
        sum_sample = 0
        train_loss_sum = 0 
        model.train()
        t0 = time.time()
        
        # Training process
        for i, (images, labels) in enumerate(train_loader):
            images = images.float().to(device)
            labels = labels.view((-1)).long().to(device)
            
            optimizer.zero_grad()
            predictions = model(images)
            _, predicted = torch.max(predictions.data, 1) 
            
            train_loss = criterion(predictions, labels)
            train_loss.backward()
            
            train_loss_sum += train_loss.item()
            optimizer.step()
            
            labels = labels.cpu()
            predicted = predicted.cpu().t()
            
            train_acc += (predicted == labels).sum()
            sum_sample += predicted.numel()
            
            if i % 10 == 0:
                print(f">>>{i}/{len(train_loader)}", end='')
        
        t1 = time.time()
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        # Calculate training accuracy
        train_acc = train_acc.data.cpu().numpy() / sum_sample
        
        # Validation process
        valid_acc, _, valid_loss = test(model, test_loader, criterion=criterion)
        
        # Save metrics
        train_metrics = OrderedDict([('loss', train_loss_sum/len(train_loader))])
        eval_metrics = OrderedDict([('loss', valid_loss), ('acc', valid_acc)])
        update_summary(epoch, train_metrics, eval_metrics, 
                      os.path.join(save_dir, f'fold{fold}-summary.csv'))
        
        acc_list['train_acc'][epoch] = train_acc
        acc_list['test_acc'][epoch] = valid_acc
        
        # Save best model
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), os.path.join(save_path, fold_dir, 'best_model.pth'))
        
        print(f'Epoch: {epoch:3d}, Train Loss: {train_loss_sum/len(train_loader):.4f}, '
              f'Train Accuracy: {train_acc:.4f}, Validation Accuracy: {valid_acc:.4f}, Time: {t1-t0:.3f}s', 
              flush=True)
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]}')
    
    # Save training results
    max_acc = acc_list['test_acc'].max()
    save_file = os.path.join(save_path, fold_dir, f'acc_list-MaxAcc={round(max_acc,3)}.mat')
    scipy.io.savemat(save_file, {'test_list': acc_list['test_acc'], 'train_list': acc_list['train_acc']})
    
    return acc_list


def main(args):
    """
    Main function
    
    Parameters:
        args: Command line arguments
    """
    # Create model save path
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Load data
    file_list = os.listdir(args.data_dir)
    
    all_data = []
    all_label = []
    all_test_labels = []
    all_predicted_label = []
    
    # Load datasets
    for i in range(len(file_list)):
        file_name = file_list[i]
        Sdata = scipy.io.loadmat(os.path.join(args.data_dir, file_name))
        datas, labels = trans_data(Sdata)
        all_data.append(datas)
        all_label.append(labels)
        print(f"Successfully loaded {os.path.join(args.data_dir, file_name)}!")
    
    # Merge data
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label)
    
    # Cross-validation
    num_files = len(all_label)
    indices = np.arange(num_files)
    np.random.shuffle(indices)
    folds = np.array_split(indices, args.fold)
    
    for i in range(args.fold):
        # Split training and test sets
        test_index = folds[i]
        train_index = np.concatenate([folds[j] for j in range(args.fold) if j!=i])
        nb_of_train_sample = len(train_index)
        nb_of_test_sample = len(test_index)        

        print(f'Training samples: {nb_of_train_sample}, Test samples: {nb_of_test_sample}')

        # Create data loaders
        train_data = TensorDataset(
            torch.from_numpy(all_data[train_index] * 1.),
            torch.from_numpy(all_label[train_index] * 1.)
        )
        train_loader = DataLoader(
            train_data, shuffle=True, batch_size=args.batch_size, drop_last=False
        )
        
        test_data = TensorDataset(
            torch.from_numpy(all_data[test_index] * 1.),
            torch.from_numpy(all_label[test_index] * 1.)
        )
        test_loader = DataLoader(
            test_data, shuffle=False, batch_size=args.batch_size, drop_last=False
        )
        
        all_test_labels.append(all_label[test_index])
        
        # Create model
        model = SpikeEEGNetModel(
            chans=args.chans, 
            feaure_dim=3000//args.time_points,
            time_points=args.time_points,
            pk2=args.pk2,
            classes=args.classes,
            activation_type=args.activation_type,
            surro_gate=args.surro_gate
        ).to(device)
        
        # Create optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        print(f"Device: {device}")
        
        # Train model
        print(f"\n===== Starting training fold {i+1}/{args.fold} =====")
        acc_list = train(
            model=model,
            epochs=args.epochs,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            test_loader=test_loader,
            fold=i,
            save_path=args.save_dir,
            model_name=args.model_name
        )
        
        # Load best model and test
        fold_dir = f'{args.model_name}/fold{i}/'
        model.load_state_dict(torch.load(os.path.join(args.save_dir, fold_dir, 'best_model.pth')))
        test_acc, predicted_labels, _ = test(model, test_loader, criterion=criterion)
        print(f'Final test accuracy for fold {i+1}: {test_acc:.4f}')
        
        all_predicted_label.append(predicted_labels)
    
    # Combine all test results
    all_predicted_label = np.concatenate(all_predicted_label)
    all_test_labels = np.concatenate(all_test_labels)
    avg_test_acc = (all_predicted_label == all_test_labels).sum() / len(all_test_labels)
    
    print(f'Final average test accuracy: {avg_test_acc:.4f}')
    
    # Save prediction results and confusion matrix
    result_dir = os.path.join(args.save_dir, args.model_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    scipy.io.savemat(
        os.path.join(result_dir, 'predict-label.mat'),
        {'predicted': all_predicted_label, 'label': all_test_labels}
    )
    
    plot_confusion_matrix(
        all_predicted_label, 
        all_test_labels, 
        list(range(args.classes)), 
        savefile=os.path.join(result_dir, 'confusion_matrix.png')
    )


if __name__ == "__main__":
    args = parse_arguments()
    main(args) 