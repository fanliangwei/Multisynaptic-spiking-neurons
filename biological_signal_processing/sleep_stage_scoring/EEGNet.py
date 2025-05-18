import torch
import torch.nn as nn
from spikingjelly.clock_driven import layer, functional, surrogate
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActFun(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, input, thresh=1.0, alpha=0.5):
        ctx.save_for_backward(input)
        ctx.thresh = thresh
        ctx.alpha = alpha
        return input.ge(thresh).float()

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        thresh = ctx.thresh
        alpha = ctx.alpha
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < alpha
        temp = temp / (2 * alpha)
        return grad_input * temp.float(), None, None

def act_fun(input, thresh=1.0, alpha=0.5):
    return ActFun.apply(input, thresh, alpha)

class mem_update(nn.Module):
    # LIF Layer
    def __init__(self, decay=0.25, thresh=1.0, alpha=0.5):
        super(mem_update, self).__init__()
        self.decay = decay
        self.thresh = thresh
        self.alpha = alpha

    def forward(self, x):
        time_window = x.size()[0] ### set timewindow
        mem = torch.zeros_like(x[0]).to(device)
        spike = torch.zeros_like(x[0]).to(device)
        output = torch.zeros_like(x)
        mem_old = 0
        for i in range(time_window):
            if i >= 1:
                mem = mem_old * self.decay * (1 - spike.detach()) + x[i]
            else:
                mem = x[i]
            spike = act_fun(mem, self.thresh, self.alpha)
            mem_old = mem.clone()
            output[i] = spike
        return output


def g_window(x,alpha):
    temp = abs(x) < alpha
    return temp / (2 * alpha)

def g_sigmoid(x,alpha):
    sgax = (alpha*x).sigmoid()
    return alpha * (1-sgax) * sgax

def g_atan(x,alpha):
    return alpha / (2 * (1 + ((np.pi / 2) * alpha * x)**2))

def g_gaussian(x,alpha):
    return (1 / np.sqrt(2 * np.pi * alpha**2)) * torch.exp(-x**2 / (2 * alpha**2))


class ActFun_rectangular(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, input, init_thre=1.0, D=4, alpha=0.5):
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
        grad_x = grad_input * (g_window(input-thresholds[0],alpha)+g_window(input-(thresholds[1]),alpha)+g_window(input-(thresholds[2]),alpha)+g_window(input-(thresholds[3]),alpha))
 
        return grad_x, None, None, None

def act_fun_rectangular(input, init_thre=1.0, D=4, alpha=0.5):
    return ActFun_rectangular.apply(input, init_thre, D, alpha)

class ActFun_sigmoid(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, input, init_thre=1.0, D=4, alpha=4.0):
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
    
def act_fun_sigmoid(input, init_thre=1.0, D=4, alpha=4.0):
    return ActFun_sigmoid.apply(input, init_thre, D, alpha)

class ActFun_atan(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, input, init_thre=1.0, D=4, alpha=2.0):
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
    
def act_fun_atan(input, init_thre=1.0, D=4, alpha=2.0):
    return ActFun_atan.apply(input, init_thre, D, alpha)

class ActFun_gaussian(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, input, init_thre=1.0, D=4, alpha=0.4):
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
    
def act_fun_gaussian(input, init_thre=1.0, D=4, alpha=0.4):
    return ActFun_gaussian.apply(input, init_thre, D, alpha)

class mem_update_MSF(nn.Module):
    # MSF Layer
    def __init__(self, decay=0.25, init_thre=1.0, D=4, surro_gate='rectangular'):
        super(mem_update_MSF, self).__init__()
        self.decay = decay
        self.init_thre = init_thre
        self.D = D
        self.surro_gate = surro_gate
        
        self.act_fun_dict = {
            'rectangular': act_fun_rectangular,
            'sigmoid': act_fun_sigmoid,
            'atan': act_fun_atan,
            'gaussian': act_fun_gaussian
        }

    def forward(self, x):
        time_window = x.size()[0] ### set timewindow
        mem = torch.zeros_like(x[0]).to(device)
        spike = torch.zeros_like(x[0]).to(device)
        output = torch.zeros_like(x)
        mem_old = 0
        
        # select the activation function
        act_fun = self.act_fun_dict.get(self.surro_gate, act_fun_rectangular)
        
        for i in range(time_window):
            if i >= 1:
                mask = spike > 0
                mem = mem_old * self.decay * (1 - mask.float()) + x[i]
            else:
                mem = x[i]
            # multi-threshold firing function
            spike = act_fun(mem, self.init_thre, self.D)
            mem_old = mem.clone()
            output[i] = spike
        return output


class SpikeEEGNetModel(nn.Module): # EEGNET-8,2
    def __init__(self, chans=22, classes=4, feaure_dim=100, time_points=30, temp_kernel=32,
                 f1=16, f2=32, d=2, pk2=16, dropout_rate=0.5, max_norm1=1, max_norm2=0.25,
                 activation_type='MSF', surro_gate='rectangular', decay=0.25, init_thre=1.0, D=4, alpha=0.5):
        super(SpikeEEGNetModel, self).__init__()
        self.feaure_dim = feaure_dim
        self.time_points = time_points
        # Calculating FC input features
        linear_size = (feaure_dim//(pk2))*f2*time_points

        # Choose activation function based on parameter
        if activation_type == 'LIF':
            self.act = mem_update(decay=decay, thresh=init_thre, alpha=alpha)  # LIF 
        else:  # Default to MSF
            self.act = mem_update_MSF(decay=decay, init_thre=init_thre, D=D, surro_gate=surro_gate)  # MSF

        # Temporal Filters
        self.block1 = nn.Sequential(
            layer.SeqToANNContainer(nn.Conv2d(1, f1, (1, temp_kernel), padding='same', bias=False)),
            layer.SeqToANNContainer(nn.BatchNorm2d(f1)),
            self.act
        )
        # Spatial Filters
        self.block2 = nn.Sequential(
            layer.SeqToANNContainer(nn.Conv2d(f1, d * f1, (chans, 1), groups=f1, bias=False)), # Depthwise Conv
            layer.SeqToANNContainer(nn.BatchNorm2d(d * f1)),
            self.act,
            nn.Dropout(dropout_rate)
        )
        self.block3 = nn.Sequential(
            layer.SeqToANNContainer(nn.Conv2d(d * f1, f2, (1, 16), bias=False, padding='same')),
            layer.SeqToANNContainer(nn.BatchNorm2d(f2)),
            self.act,
            nn.AvgPool3d((1, 1, pk2)),
            nn.Dropout(dropout_rate)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(linear_size, classes)

        # Apply max_norm constraint to the depthwise layer in block2
        self._apply_max_norm(self.block2[0], max_norm1)

        # Apply max_norm constraint to the linear layer
        self._apply_max_norm(self.fc, max_norm2)

    def _apply_max_norm(self, layer, max_norm):
        for name, param in layer.named_parameters():
            if 'weight' in name:
                param.data = torch.renorm(param.data, p=2, dim=0, maxnorm=max_norm)

    def forward(self, x):
        b,t = x.shape
        x = x.reshape(b,1,1,self.time_points,self.feaure_dim).permute(3,0,1,2,4)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x) # T,B,C,1,1
        x = x.permute(1,0,2,3,4)
        x = self.flatten(x)
        x = self.fc(x)
        return x