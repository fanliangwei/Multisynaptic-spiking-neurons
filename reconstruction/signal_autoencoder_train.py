# Signal Autoencoder Training for Spiking Neural Networks
# This file implements spiking autoencoders for signal reconstruction
# including both LIF and MSF neurons

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from spikingjelly.clock_driven import layer, functional, surrogate
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import pandas as pd
import os


class ActFun(torch.autograd.Function):
    """Custom activation function for LIF neurons"""
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, input, thresh=0.5, alpha=0.5):
        """
        Forward pass: Applies step function at threshold
        """
        ctx.save_for_backward(input)
        ctx.thresh = thresh
        ctx.alpha = alpha
        return input.ge(thresh).float()

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        """
        Backward pass: Applies rectangular surrogate gradient
        """
        (input,) = ctx.saved_tensors
        thresh = ctx.thresh
        alpha = ctx.alpha
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < alpha
        temp = temp / (2 * alpha)
        return grad_input * temp.float(), None, None

def act_fun(input, thresh=0.5, alpha=0.5):
    return ActFun.apply(input, thresh, alpha)

class mem_update(nn.Module):
    """LIF (Leaky Integrate-and-Fire) Layer implementation"""
    def __init__(self, decay=0.25, thresh=0.5, alpha=0.5):
        super(mem_update, self).__init__()
        self.decay = decay
        self.thresh = thresh
        self.alpha = alpha

    def forward(self, x):
        time_window = x.size()[0] ### set timewindow
        mem = torch.zeros_like(x[0]).to(x.device)
        spike = torch.zeros_like(x[0]).to(x.device)
        output = torch.zeros_like(x)
        mem_old = 0
        for i in range(time_window):
            if i >= 1:
                # Update membrane potential
                mem = mem_old * self.decay * (1 - spike.detach()) + x[i]
            else:
                mem = x[i]
            # Generate spike if membrane potential exceeds threshold
            spike = act_fun(mem, self.thresh, self.alpha)
            mem_old = mem.clone()
            output[i] = spike
        return output


# Surrogate gradient functions
def g_window(x,alpha):
    """Rectangular surrogate gradient"""
    temp = abs(x) < alpha
    return temp / (2 * alpha)

def g_sigmoid(x,alpha):
    """Sigmoid surrogate gradient"""
    sgax = (alpha*x).sigmoid()
    return alpha * (1-sgax) * sgax

def g_atan(x,alpha):
    """Arctangent surrogate gradient"""
    return alpha / (2 * (1 + ((np.pi / 2) * alpha * x)**2))

def g_gaussian(x,alpha):
    """Gaussian surrogate gradient"""
    return (1 / np.sqrt(2 * np.pi * alpha**2)) * torch.exp(-x**2 / (2 * alpha**2))


# Multi-synaptic activation functions with different surrogate gradients
class ActFun_rectangular(torch.autograd.Function):
    """Multi-synaptic activation function with rectangular surrogate gradient"""
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, input, init_thre=0.5, D=4, alpha=0.5):
        """
        Forward pass: Multi-synaptic spike generation
        Args:
            input: Input tensor
            init_thre: Initial threshold value
            D: Number of synapses
            alpha: Surrogate gradient parameter
        """
        ctx.save_for_backward(input)
        ctx.init_thre = init_thre
        ctx.D = D
        ctx.alpha = alpha
        
        # Create multiple thresholds
        thresholds = torch.arange(D, device=input.device).float() + init_thre
        out = input.ge(thresholds[0]).float() + input.ge(thresholds[1]).float() + input.ge(thresholds[2]).float() + input.ge(thresholds[3]).float()
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        """Backward pass with rectangular surrogate gradient"""
        (input,) = ctx.saved_tensors
        init_thre = ctx.init_thre
        D = ctx.D
        alpha = ctx.alpha
        grad_input = grad_output.clone()
        
        thresholds = torch.arange(D, device=input.device).float() + init_thre
        # Apply surrogate gradient for each threshold
        grad_x = grad_input * (g_window(input-thresholds[0],alpha)+g_window(input-(thresholds[1]),alpha)+g_window(input-(thresholds[2]),alpha)+g_window(input-(thresholds[3]),alpha))
 
        return grad_x, None, None, None

def act_fun_rectangular(input, init_thre=0.5, D=4, alpha=0.5):
    return ActFun_rectangular.apply(input, init_thre, D, alpha)

class ActFun_sigmoid(torch.autograd.Function):
    """Multi-synaptic activation function with sigmoid surrogate gradient"""
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, input, init_thre=0.5, D=4, alpha=4.0):
        ctx.save_for_backward(input)
        ctx.init_thre = init_thre
        ctx.D = D
        ctx.alpha = alpha
        
        thresholds = torch.arange(D, device=input.device).float() + init_thre
        out = input.ge(thresholds[0]).float() + input.ge(thresholds[1]).float() + input.ge(thresholds[2]).float() + input.ge(thresholds[3]).float()
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        init_thre = ctx.init_thre
        D = ctx.D
        alpha = ctx.alpha
        
        grad_input = grad_output.clone()
        thresholds = torch.arange(D, device=input.device).float() + init_thre
        grad_x = grad_input * (g_sigmoid(input-thresholds[0],alpha)+g_sigmoid(input-thresholds[1],alpha)+g_sigmoid(input-thresholds[2],alpha)+g_sigmoid(input-thresholds[3],alpha))
 
        return grad_x, None, None, None    
    
def act_fun_sigmoid(input, init_thre=0.5, D=4, alpha=4.0):
    return ActFun_sigmoid.apply(input, init_thre, D, alpha)

class ActFun_atan(torch.autograd.Function):
    """Multi-synaptic activation function with arctangent surrogate gradient"""
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, input, init_thre=0.5, D=4, alpha=2.0):
        ctx.save_for_backward(input)
        ctx.init_thre = init_thre
        ctx.D = D
        ctx.alpha = alpha
        
        thresholds = torch.arange(D, device=input.device).float() + init_thre
        out = input.ge(thresholds[0]).float() + input.ge(thresholds[1]).float() + input.ge(thresholds[2]).float() + input.ge(thresholds[3]).float()
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        init_thre = ctx.init_thre
        D = ctx.D
        alpha = ctx.alpha
        
        grad_input = grad_output.clone()
        thresholds = torch.arange(D, device=input.device).float() + init_thre
        grad_x = grad_input * (g_atan(input-thresholds[0],alpha)+g_atan(input-thresholds[1],alpha)+g_atan(input-thresholds[2],alpha)+g_atan(input-thresholds[3],alpha))
 
        return grad_x, None, None, None    
    
def act_fun_atan(input, init_thre=0.5, D=4, alpha=2.0):
    return ActFun_atan.apply(input, init_thre, D, alpha)

class ActFun_gaussian(torch.autograd.Function):
    """Multi-synaptic activation function with Gaussian surrogate gradient"""
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, input, init_thre=0.5, D=4, alpha=0.4):
        ctx.save_for_backward(input)
        ctx.init_thre = init_thre
        ctx.D = D
        ctx.alpha = alpha
        
        thresholds = torch.arange(D, device=input.device).float() + init_thre
        out = input.ge(thresholds[0]).float() + input.ge(thresholds[1]).float() + input.ge(thresholds[2]).float() + input.ge(thresholds[3]).float()
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        init_thre = ctx.init_thre
        D = ctx.D
        alpha = ctx.alpha
        
        grad_input = grad_output.clone()
        thresholds = torch.arange(D, device=input.device).float() + init_thre
        grad_x = grad_input * (g_gaussian(input-thresholds[0],alpha)+g_gaussian(input-thresholds[1],alpha)+g_gaussian(input-thresholds[2],alpha)+g_gaussian(input-thresholds[3],alpha))
 
        return grad_x, None, None, None    
    
def act_fun_gaussian(input, init_thre=0.5, D=4, alpha=0.4):
    return ActFun_gaussian.apply(input, init_thre, D, alpha)

class mem_update_MSF(nn.Module):
    """MSF Layer implementation"""
    def __init__(self, decay=0.25, init_thre=0.5, D=4, surro_gate='rectangular'):
        """
        Args:
            decay: Membrane potential decay factor
            init_thre: Initial firing threshold
            D: Number of synapses
            surro_gate: Type of surrogate gradient ('rectangular', 'sigmoid', 'atan', 'gaussian')
        """
        super(mem_update_MSF, self).__init__()
        self.decay = decay
        self.init_thre = init_thre
        self.D = D
        self.surro_gate = surro_gate
        
        # Dictionary of available activation functions
        self.act_fun_dict = {
            'rectangular': act_fun_rectangular,
            'sigmoid': act_fun_sigmoid,
            'atan': act_fun_atan,
            'gaussian': act_fun_gaussian
        }

    def forward(self, x):
        """
        Forward pass through MSF layer
        Returns:
            output: Spike trains
        """
        time_window = x.size()[0] ### set timewindow
        mem = torch.zeros_like(x[0]).to(x.device)
        spike = torch.zeros_like(x[0]).to(x.device)
        output = torch.zeros_like(x)
        mem_old = 0
        
        # Select the activation function based on surrogate gate type
        act_fun = self.act_fun_dict.get(self.surro_gate, act_fun_rectangular)
        
        for i in range(time_window):
            if i >= 1:
                # Update membrane potential
                mask = spike > 0
                mem = mem_old * self.decay * (1 - mask.float()) + x[i]
            else:
                mem = x[i]
            # Multi-threshold firing function
            spike = act_fun(mem, self.init_thre, self.D)
            mem_old = mem.clone()
            output[i] = spike
        return output


# Signal generation functions
def generate_variable_sine_wave(A_func, T_func, phase, timepoint=100):
    """
    Generate sine signals with continuously varying periods and amplitudes
    
    Args:
        A_func: Function that returns amplitude A(x) at corresponding position
        T_func: Function that returns period T(x) at corresponding position
        phase: Signal phase
        timepoint: Total number of time points
    
    Returns: Generated sine wave signal
    """
    x = np.arange(0, timepoint)
    
    # Calculate amplitude and period that change over time
    A_values = A_func(x)
    T_values = T_func(x)
    
    # Generate sine signal
    y = A_values * np.sin(2 * np.pi * 1/T_values * x + phase)
    
    return torch.tensor(y, dtype=torch.float32).unsqueeze(0)

# Example parameter functions for varying signals
A_func = lambda x: 1 + 3 * np.sin(0.1 * x)  # Amplitude varies between 1 and 4
T_func = lambda x: 10 + 5 * np.cos(0.05 * x)  # Period varies between 10 and 15


def generate_variable_square_wave(T_func, phase, timepoint=100):
    """
    Generate square wave signals with continuously varying periods
    
    Args:
        T_func: Function that returns period T(x) at corresponding position
        phase: Signal phase
        timepoint: Total number of time points
    
    Returns: Generated square wave signal
    """
    x = np.arange(0, timepoint)
    
    # Calculate period that changes over time
    T_values = T_func(x)
    
    # Generate periodic square wave signal using sign of sine wave
    y = np.sign(np.sin(2 * np.pi * 1/T_values * x + phase))
    
    # Adjust amplitude to range [0, 1]
    y = (y + 1) / 2
    
    return torch.tensor(y, dtype=torch.float32).unsqueeze(0)

# Example parameter functions (same as above)
A_func = lambda x: 1 + 3 * np.sin(0.1 * x)  # Amplitude varies between 1 and 4
T_func = lambda x: 10 + 5 * np.cos(0.05 * x)  # Period varies between 10 and 15


def generate_sine_wave(A, T, phase, timepoint=100):
    """
    Generate sine signal with fixed amplitude A, period T, and phase
    
    Args:
        A: Amplitude
        T: Period
        phase: Phase shift
        timepoint: Number of time points
    
    Returns: Generated sine wave signal
    """
    x = np.arange(0, timepoint)
    y = A * np.sin(2 * np.pi * 1/T * x + phase)
    return torch.tensor(y, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

def generate_square_wave(A, T, phase, timepoint=100):
    """
    Generate periodic square wave signal with fixed amplitude A, period T, and phase
    
    Args:
        A: Amplitude
        T: Period
        phase: Phase shift
        timepoint: Number of time points
    
    Returns: Generated square wave signal
    """
    x = np.arange(0, timepoint)
    
    # Generate square wave using sign of sine wave
    y = np.sign(np.sin(2 * np.pi * 1/T * x + phase))
    
    # Scale amplitude and normalize to [0, A] range
    y = A * (y + 1) / 2
    
    return torch.tensor(y, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

class Snn_Autoencoder(nn.Module):
    """
    Spiking Autoencoder
    Implements encoder-decoder architecture with spiking neurons for signal reconstruction
    """
    def __init__(self, input_size, hidden_size, activation_type='MSF'):
        """
        Initialize SNN Autoencoder
        
        Args:
            input_size: Input dimension size
            hidden_size: Hidden layer size (bottleneck dimension)
            activation_type: Type of spiking activation ('LIF' or 'MSF')
        """
        super(Snn_Autoencoder, self).__init__()
        self.fc1 = layer.SeqToANNContainer(nn.Linear(input_size, hidden_size))  # Encoder layer
        
        # Choose activation function based on parameter
        if activation_type == 'LIF':
            self.act = mem_update()  # LIF neurons
        else:  # Default to MSF
            self.act = mem_update_MSF()  # MSF neurons
            
        self.fc2 = layer.SeqToANNContainer(nn.Linear(hidden_size, input_size))  # Decoder layer
    
    def forward(self, x):
        encoded = self.act(self.fc1(x))  # Encoding with spiking activation
        decoded = self.fc2(encoded)  # Decoding
        return decoded


def train_model(model, train_data, test_data, epochs, batch_size, learning_rate, device='cpu', model_prefix='', experiment_name=''):
    """
    Train the autoencoder model
    
    Args:
        model: SNN autoencoder model
        train_data: Training dataset
        test_data: Testing dataset
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        device: Training device ('cpu' or 'cuda')
        model_prefix: Prefix for saved model files
        experiment_name: Experiment name for creating save directories
    """
    model.to(device)
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)  # Cosine annealing scheduler
    
    # Record training and testing losses
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    best_epoch = 0
    
    # Create model directory
    model_dir = os.path.join(experiment_name, 'models') if experiment_name else 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_losses = []
        
        # Create mini-batches with random shuffling
        indices = np.random.permutation(len(train_data))
        for start_idx in range(0, len(train_data), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            # Prepare batch data
            batch = torch.stack([train_data[i] for i in batch_indices])
            batch = batch.unsqueeze(-1).permute(2, 0, 1, 3).to(device)
            
            # Forward propagation
            optimizer.zero_grad()
            output = model(batch)
            
            # Calculate reconstruction loss
            loss = criterion(output, batch)
            epoch_losses.append(loss.item())
            
            # Backward propagation and optimization
            loss.backward()
            optimizer.step()
        
        # Update learning rate
        scheduler.step()
        
        # Record average training loss
        avg_train_loss = np.mean(epoch_losses)
        train_losses.append(avg_train_loss)
        
        # Testing phase
        test_loss = evaluate_model(model, test_data, criterion, device)
        test_losses.append(test_loss)

        # Check if current model is the best so far
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch
            # Save the best model
            best_model_path = os.path.join(model_dir, f'{model_prefix}_best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch+1}: Found new best model, test loss: {test_loss:.6f}")

        print(f'Epoch [{epoch+1}/{epochs}], Training loss: {avg_train_loss:.6f}, Test loss: {test_loss:.6f}, Best loss: {best_test_loss:.6f} (Epoch {best_epoch+1})')              
    
    # Save the final model
    last_model_path = os.path.join(model_dir, f'{model_prefix}_last_model.pth')
    torch.save(model.state_dict(), last_model_path)
    
    print(f"Training completed. Best model (Epoch {best_epoch+1}) saved to {best_model_path}")
    print(f"Final model saved to {last_model_path}")
    
    return train_losses, test_losses, best_model_path, last_model_path, best_epoch

def evaluate_model(model, test_data, criterion, device='cpu'):
    """
    Evaluate model performance on test data
    
    Args:
        model: Trained model
        test_data: Test dataset
        criterion: Loss function
        device: Evaluation device
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for test_signal in test_data:
            # Prepare test input
            test_input = test_signal.unsqueeze(-1).unsqueeze(0).permute(2, 0, 1, 3).to(device)
            output = model(test_input)
            loss = criterion(output, test_input).item()
            total_loss += loss
    
    return total_loss / len(test_data)

def test_model(model, test_signals, device='cpu'):
    """
    Test model and calculate reconstruction error for each signal
    
    Args:
        model: Trained model
        test_signals: List of test signals
        device: Testing device
    
    """
    model.eval()
    criterion = nn.MSELoss()
    reconstruction_errors = []
    reconstructed_signals = []
    
    with torch.no_grad():
        for test_signal in test_signals:
            # Prepare test data
            test_input = test_signal.unsqueeze(-1).unsqueeze(0).permute(2, 0, 1, 3).to(device)
            
            # Reconstruct signal using model
            reconstructed = model(test_input)
            
            # Calculate reconstruction error
            error = criterion(reconstructed, test_input).item()
            reconstruction_errors.append(error)
            
            # Save reconstructed signal
            reconstructed_signals.append(reconstructed.cpu().squeeze())

    return reconstructed_signals, reconstruction_errors

def plot_training_history(train_losses, test_losses, best_epoch, signal_type, experiment_name=''):
    """
    Visualize training and testing loss curves
    
    Args:
        train_losses: List of training losses per epoch
        test_losses: List of testing losses per epoch
        best_epoch: Epoch number of best model
        signal_type: Type of signal ('sine' or 'square')
        experiment_name: Experiment name for saving plots
    """
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Model (Epoch {best_epoch+1})')
    plt.title(f'{signal_type.capitalize()} Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    
    # Set save path
    save_path = os.path.join(experiment_name, f'{signal_type}_training_history.png') if experiment_name else f'{signal_type}_training_history.png'
    plt.savefig(save_path)
    plt.show()


def generate_datasets(n_train=100, n_test=100, signal_type='sine'):
    """
    Generate training and testing datasets with different parameter ranges
    
    Args:
        n_train: Number of training samples per parameter combination
        n_test: Number of testing samples per parameter combination
        signal_type: 'sine' for sine waves only, 'square' for square waves only
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Training data parameter ranges
    A_train = [random.uniform(1.0, 4.0) for _ in range(n_train)]  # Amplitude range
    T_train = list(range(8, 65))  # Period range
    phase_train = [random.uniform(0, 2*np.pi) for _ in range(8)]  # Phase range
    
    # Testing data parameter ranges - use different parameters to test generalization
    A_test = [random.uniform(1.0, 4.0) for _ in range(n_test)]
    T_test = [8,16,24,32,40,48,56,64]  # Specific test periods
    phase_test = [np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]  # Specific test phases
    
    # Generate training data
    train_data = []
    for A in A_train:
        for T in T_train:
            for phase in phase_train:
                if signal_type == 'sine':
                    signal = generate_sine_wave(A, T, phase, timepoint=300)
                else:  # signal_type == 'square'
                    signal = generate_square_wave(1, T, phase, timepoint=300)
                train_data.append(signal)
    
    # Generate testing data
    test_data = []
    test_params = []  # Record parameters of test signals
    
    # Fixed parameter test signals
    for A in A_test:
        for T in T_test:
            for phase in phase_test:
                if signal_type == 'sine':
                    # Generate sine wave
                    signal = generate_sine_wave(A, T, phase, timepoint=300)
                    test_data.append(signal)
                    test_params.append(('sine', A, T, phase))
                else:  # signal_type == 'square'
                    # Generate square wave
                    signal = generate_square_wave(1, T, phase, timepoint=300)
                    test_data.append(signal)
                    test_params.append(('square', 1, T, phase))
    
    return train_data, test_data, test_params


def plot_varying_examples(signal_type, test_model, device='cpu', experiment_name=''):
    """
    Test model on signals with continuously varying parameters and save results
    
    Args:
        signal_type: Type of signal to test ('sine' or 'square')
        test_model: Trained model for testing
        device: Testing device
        experiment_name: Experiment name for saving results
    """
    # Define parameter variation functions
    a_scale = 3
    t_scale = 5
    A_func = lambda x: 1 + a_scale * np.sin(0.1 * x)  # Varying amplitude
    T_func = lambda x: 80 + t_scale * np.cos(0.05 * x)  # Varying period
        
    if signal_type == 'sine':
        # Generate sine wave with varying parameters
        varying_signal = generate_variable_sine_wave(A_func, T_func, phase=0, timepoint=300)
    else:  # signal_type == 'square'
        # Generate square wave with varying parameters
        varying_signal = generate_variable_square_wave(T_func, phase=np.pi*3/2, timepoint=300)

    with torch.no_grad():
        test_signal = varying_signal.reshape(300,1,1,1)
        reconstructed_signal = test_model(test_signal.to(device))
    
    original = test_signal.numpy().flatten()
    reconstructed = reconstructed_signal.cpu().numpy().flatten()
    
    # Calculate reconstruction error at each time point (squared error)
    point_errors = (original - reconstructed) ** 2
    
    # Calculate overall MSE error
    mse_error = np.mean(point_errors)
    
    # Create dataframe and save to CSV
    df = pd.DataFrame({
        'time_point': range(len(original)),
        'original_signal': original,
        'reconstructed_signal': reconstructed,
        'squared_error': point_errors
    })
    
    # Set CSV save path
    csv_dir = experiment_name if experiment_name else '.'
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    csv_path = os.path.join(csv_dir, f'{signal_type}_varying_signals.csv')
    df.to_csv(csv_path, index=False)
    print(f'Varying parameter signal data saved to {csv_path}')
    print(f'Reconstruction MSE error: {mse_error:.6f}')

    # Plot reconstruction results
    plt.figure(figsize=(10, 5))
    plt.plot(original, label="Original Signal")
    plt.plot(reconstructed, label="Reconstructed Signal", linestyle='dashed')
    plt.legend()
    plt.title(f"Original vs Reconstructed Signal (MSE = {mse_error:.6f})")
    
    # Set save path
    save_path = os.path.join(experiment_name, f'{signal_type}_varying_reconstruction.png') if experiment_name else f'{signal_type}_varying_reconstruction.png'
    plt.savefig(save_path)
    plt.show()

def plot_reconstruction_examples(original_signals, reconstructed_signals, test_params, n_examples=5, experiment_name=''):
    """
    Visualize reconstruction results for selected test examples
    
    Args:
        original_signals: List of original test signals
        reconstructed_signals: List of reconstructed signals
        test_params: List of signal parameters for each test case
        n_examples: Number of examples to display
        experiment_name: Experiment name for saving plots
    """
    # Randomly select n_examples examples
    indices = np.random.choice(len(original_signals), n_examples, replace=False)
    
    plt.figure(figsize=(15, 3*n_examples))
    for i, idx in enumerate(indices):
        plt.subplot(n_examples, 1, i+1)
        
        original = original_signals[idx].numpy().flatten()
        reconstructed = reconstructed_signals[idx].numpy().flatten()
        
        # Get signal parameters for title
        signal_type, *params = test_params[idx]
        if signal_type == 'sine' or signal_type == 'square':
            A, T, phase = params
            title = f"{signal_type.capitalize()}: A={A:.2f}, T={T}, phase={phase:.2f}"
        else:
            a_scale, t_scale = params
            title = f"{signal_type.capitalize()}: a_scale={a_scale:.2f}, t_scale={t_scale:.2f}"
        
        plt.plot(original, label="Original")
        plt.plot(reconstructed, label="Reconstructed", linestyle='dashed')
        plt.title(title)
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    # Set save path
    save_path = os.path.join(experiment_name, 'reconstruction_examples.png') if experiment_name else 'reconstruction_examples.png'
    plt.savefig(save_path)
    plt.show()


def main(signal_type='sine', experiment_name='', input_size=1, hidden_size=1, epochs=30, batch_size=32, learning_rate=0.001, activation_type='MSF'):
    """
    Main function for training and testing signal reconstruction models
    
    Args:
        signal_type: 'sine' for sine wave, 'square' for square wave
        experiment_name: Experiment name, used to create folders for saving results
        input_size: Input dimension size
        hidden_size: Hidden layer size (encoding bottleneck)
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        activation_type: Type of spiking activation function: 'LIF' or 'MSF'
    """
    # Set random seed to ensure reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Set computing device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create experiment directory
    if experiment_name and not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
        print(f"Created experiment directory: {experiment_name}")
    
    # Display model parameters
    print(f"Model parameters: input_size={input_size}, hidden_size={hidden_size}, epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}, activation_type={activation_type}")
    
    if signal_type == 'sine':
        # Train sine wave model
        print("\n=== Training Sine Wave Model ===")
        # Generate sine wave dataset
        print("Generating sine wave dataset...")
        train_data, test_data, test_params = generate_datasets(n_train=50, n_test=10, signal_type='sine')
        print(f"Training data: {len(train_data)} samples")
        print(f"Testing data: {len(test_data)} samples")
        
        # Create sine wave model
        model = Snn_Autoencoder(input_size, hidden_size, activation_type)
        print(f"Created sine wave SNN autoencoder: input size={input_size}, hidden size={hidden_size}, activation={activation_type}")
        
        # Train sine wave model, test simultaneously and save the best model
        print("Starting sine wave model training...")
        train_losses, test_losses, best_model_path, last_model_path, best_epoch = train_model(
            model, train_data, test_data, epochs, batch_size, learning_rate, device, 
            model_prefix='sine', experiment_name=experiment_name)
        
        # Visualize training history
        plot_training_history(train_losses, test_losses, best_epoch, 'sine', experiment_name)
        
        # Load the best model for final testing
        print("Loading best model for final testing...")
        best_model = Snn_Autoencoder(input_size, hidden_size, activation_type)
        best_model.load_state_dict(torch.load(best_model_path))
        best_model.to(device)
        
        # Test the best model
        print("Testing best sine wave model...")
        reconstructed_signals, reconstruction_errors = test_model(best_model, test_data, device)
        
        # Calculate sine wave model average reconstruction error
        avg_error = np.mean(reconstruction_errors)
        print(f"Sine wave best model average reconstruction error (MSE): {avg_error:.6f}")
        
        # Visualize sine wave model reconstruction results
        plot_reconstruction_examples(test_data, reconstructed_signals, test_params, n_examples=5, experiment_name=experiment_name)
             
        # Show varying parameter signal reconstruction results
        plot_varying_examples(signal_type, best_model, device, experiment_name)
        
    elif signal_type == 'square':
        # Train square wave model
        print("\n=== Training Square Wave Model ===")
        # Generate square wave dataset
        print("Generating square wave dataset...")
        train_data, test_data, test_params = generate_datasets(n_train=50, n_test=10, signal_type='square')
        print(f"Training data: {len(train_data)} samples")
        print(f"Testing data: {len(test_data)} samples")
        
        # Create square wave model
        model = Snn_Autoencoder(input_size, hidden_size, activation_type)
        print(f"Created square wave SNN autoencoder: input size={input_size}, hidden size={hidden_size}, activation={activation_type}")
        
        # Train square wave model, test simultaneously and save the best model
        print("Starting square wave model training...")
        train_losses, test_losses, best_model_path, last_model_path, best_epoch = train_model(
            model, train_data, test_data, epochs, batch_size, learning_rate, device, 
            model_prefix='square', experiment_name=experiment_name)
        
        # Visualize training history
        plot_training_history(train_losses, test_losses, best_epoch, 'square', experiment_name)
        
        # Load the best model for final testing
        print("Loading best model for final testing...")
        best_model = Snn_Autoencoder(input_size, hidden_size, activation_type)
        best_model.load_state_dict(torch.load(best_model_path))
        best_model.to(device)
        
        # Test the best model
        print("Testing best square wave model...")
        reconstructed_signals, reconstruction_errors = test_model(best_model, test_data, device)
        
        # Calculate square wave model average reconstruction error
        avg_error = np.mean(reconstruction_errors)
        print(f"Square wave best model average reconstruction error (MSE): {avg_error:.6f}")
        
        # Visualize square wave model reconstruction results
        plot_reconstruction_examples(test_data, reconstructed_signals, test_params, n_examples=5, experiment_name=experiment_name)
        
        # Show varying parameter signal reconstruction results
        plot_varying_examples(signal_type, best_model, device, experiment_name)
        
    else:
        print(f"Error: Unsupported signal type '{signal_type}'. Please use 'sine' or 'square'.")


if __name__ == "__main__":
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Train SNN signal autoencoder')
    parser.add_argument('--signal_type', type=str, default='sine', choices=['sine', 'square'],
                        help='Signal type to train: sine or square')
    parser.add_argument('--experiment_name', type=str, default='sine_snn_autoencoder_h32-MSF',
                        help='Experiment name, used to create folders for saving results')
    # Model parameter command line arguments
    parser.add_argument('--input_size', type=int, default=1,
                        help='Input dimension size')
    parser.add_argument('--hidden_size', type=int, default=32,
                        help='Hidden layer size')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer')
    # Spiking activation function type
    parser.add_argument('--activation_type', type=str, default='MSF', choices=['LIF', 'MSF'],
                        help='Spiking activation function type: LIF or MSF')
    
    args = parser.parse_args()
    
    main(
        signal_type=args.signal_type, 
        experiment_name=args.experiment_name,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        activation_type=args.activation_type
    )