from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
# from .._internally_replaced_utils import load_state_dict_from_url
# from ..utils import _log_api_usage_once
from attention import CSA

from spikingjelly.clock_driven.neuron import *
from spikingjelly.clock_driven import layer, functional, surrogate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thresh = 1.0

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
    def forward(ctx, input, init_thre=1.0, D=8, alpha=0.5):
        ctx.save_for_backward(input)
        ctx.init_thre = init_thre
        ctx.D = D
        ctx.alpha = alpha
        
        thresholds = torch.arange(D, device=input.device).float() + init_thre
        out = input.ge(thresholds[0]).float() + input.ge(thresholds[1]).float() + input.ge(thresholds[2]).float() + input.ge(thresholds[3]).float() + input.ge(thresholds[4]).float() + input.ge(thresholds[5]).float() + input.ge(thresholds[6]).float() + input.ge(thresholds[7]).float()
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
        grad_x = grad_input * (g_window(input-thresholds[0],alpha)+g_window(input-(thresholds[1]),alpha)+g_window(input-(thresholds[2]),alpha)+g_window(input-(thresholds[3]),alpha)+g_window(input-thresholds[4],alpha)+g_window(input-(thresholds[5]),alpha)+g_window(input-(thresholds[6]),alpha)+g_window(input-(thresholds[7]),alpha))
 
        return grad_x, None, None, None

def act_fun_rectangular(input, init_thre=1.0, D=8, alpha=0.5):
    return ActFun_rectangular.apply(input, init_thre, D, alpha)

class ActFun_sigmoid(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, input, init_thre=1.0, D=8, alpha=4.0):
        ctx.save_for_backward(input)
        ctx.init_thre = init_thre
        ctx.D = D
        ctx.alpha = alpha
        
        thresholds = torch.arange(D, device=input.device).float() + init_thre
        out = input.ge(thresholds[0]).float() + input.ge(thresholds[1]).float() + input.ge(thresholds[2]).float() + input.ge(thresholds[3]).float() + input.ge(thresholds[4]).float() + input.ge(thresholds[5]).float() + input.ge(thresholds[6]).float() + input.ge(thresholds[7]).float()
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
        grad_x = grad_input * (g_sigmoid(input-thresholds[0],alpha)+g_sigmoid(input-thresholds[1],alpha)+g_sigmoid(input-thresholds[2],alpha)+g_sigmoid(input-thresholds[3],alpha)+g_sigmoid(input-thresholds[4],alpha)+g_sigmoid(input-thresholds[5],alpha)+g_sigmoid(input-thresholds[6],alpha)+g_sigmoid(input-thresholds[7],alpha))
 
        return grad_x, None, None, None    
    
def act_fun_sigmoid(input, init_thre=1.0, D=8, alpha=4.0):
    return ActFun_sigmoid.apply(input, init_thre, D, alpha)

class ActFun_atan(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, input, init_thre=1.0, D=8, alpha=2.0):
        ctx.save_for_backward(input)
        ctx.init_thre = init_thre
        ctx.D = D
        ctx.alpha = alpha
        
        thresholds = torch.arange(D, device=input.device).float() + init_thre
        out = input.ge(thresholds[0]).float() + input.ge(thresholds[1]).float() + input.ge(thresholds[2]).float() + input.ge(thresholds[3]).float() + input.ge(thresholds[4]).float() + input.ge(thresholds[5]).float() + input.ge(thresholds[6]).float() + input.ge(thresholds[7]).float()
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
        grad_x = grad_input * (g_atan(input-thresholds[0],alpha)+g_atan(input-thresholds[1],alpha)+g_atan(input-thresholds[2],alpha)+g_atan(input-thresholds[3],alpha)+g_atan(input-thresholds[4],alpha)+g_atan(input-thresholds[5],alpha)+g_atan(input-thresholds[6],alpha)+g_atan(input-thresholds[7],alpha))
 
        return grad_x, None, None, None    
    
def act_fun_atan(input, init_thre=1.0, D=8, alpha=2.0):
    return ActFun_atan.apply(input, init_thre, D, alpha)

class ActFun_gaussian(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, input, init_thre=1.0, D=8, alpha=0.4):
        ctx.save_for_backward(input)
        ctx.init_thre = init_thre
        ctx.D = D
        ctx.alpha = alpha
        
        thresholds = torch.arange(D, device=input.device).float() + init_thre
        out = input.ge(thresholds[0]).float() + input.ge(thresholds[1]).float() + input.ge(thresholds[2]).float() + input.ge(thresholds[3]).float() + input.ge(thresholds[4]).float() + input.ge(thresholds[5]).float() + input.ge(thresholds[6]).float() + input.ge(thresholds[7]).float()
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
        grad_x = grad_input * (g_gaussian(input-thresholds[0],alpha)+g_gaussian(input-thresholds[1],alpha)+g_gaussian(input-thresholds[2],alpha)+g_gaussian(input-thresholds[3],alpha)+g_gaussian(input-thresholds[4],alpha)+g_gaussian(input-thresholds[5],alpha)+g_gaussian(input-thresholds[6],alpha)+g_gaussian(input-thresholds[7],alpha))
 
        return grad_x, None, None, None    
    
def act_fun_gaussian(input, init_thre=1.0, D=8, alpha=0.4):
    return ActFun_gaussian.apply(input, init_thre, D, alpha)

class mem_update_MSF(nn.Module):
    # MSF Layer
    def __init__(self, decay=0.25, init_thre=1.0, D=8, surro_gate='rectangular'):
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



class batch_norm_2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(batch_norm_2d, self).__init__()
        self.bn = BatchNorm3d1(num_features)  # input (N,C,D,H,W) C-dimension batch norm on (N,D,H,W) slice. spatio-temporal Batch Normalization

    def forward(self, input):
        y = (
            input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
        )  
        y = self.bn(y)
        return (
            y.contiguous().transpose(0, 1).contiguous().transpose(0, 2)
        )  


class batch_norm_2d1(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(batch_norm_2d1, self).__init__()
        self.bn = BatchNorm3d2(num_features)

    def forward(self, input):
        y = input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
        y = self.bn(y)
        return y.contiguous().transpose(0, 1).contiguous().transpose(0, 2)


class BatchNorm3d1(torch.nn.BatchNorm3d):
    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, thresh)
            nn.init.zeros_(self.bias)


class BatchNorm3d2(torch.nn.BatchNorm3d):
    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, 0.2 * thresh)
            # nn.init.constant_(self.weight, 0)
            nn.init.zeros_(self.bias)


class Bottle2neck(nn.Module):

    expansion: int = 2 

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        scale = 4,
        stype='normal'
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.snn_conv1 = nn.Sequential(
            mem_update_MSF(),
            layer.SeqToANNContainer(
                nn.Conv2d(
                    in_channels=inplanes,
                    out_channels=width*scale,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    bias=False,
                )
            ),            
            batch_norm_2d(width*scale),
        )        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool3d(kernel_size=(1,3,3), stride = (1,stride,stride), padding=(0,1,1))
        convs = []        
        for i in range(self.nums):
            convs.append(
                nn.Sequential(
                   mem_update_MSF(),
                    layer.SeqToANNContainer(
                        nn.Conv2d(
                            in_channels=width,
                            out_channels=width,
                            kernel_size=3,
                            padding=1,
                            stride=stride,
                            groups=groups,
                            bias=False,
                        )
                    ),                        
                    batch_norm_2d1(width),
                )               
            )
        self.convs = nn.ModuleList(convs)

        self.snn_conv3 = nn.Sequential(
            mem_update_MSF(),
            layer.SeqToANNContainer(
                nn.Conv2d(
                    in_channels=width*scale,
                    out_channels=planes*Bottle2neck.expansion,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    bias=False,
                )
            ),               
            batch_norm_2d1(planes * Bottle2neck.expansion),
            CSA(1, planes * Bottle2neck.expansion, c_ratio=8),
        )     

        self.downsample = downsample
        self.stride = stride
        self.stype = stype
        self.scale = scale
        self.width  = width


    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.snn_conv1(x) # torch.Size([1, 1, 256, 120, 152])

        spx = torch.split(out, self.width, 2)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i] # torch.Size([1, 1, 64, 120, 152])
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp) #torch.Size([1, 1, 64, 60, 76])
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 2) # torch.Size([1, 1, 128, 60, 76])
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),2)
        elif self.scale != 1 and self.stype=='stage':
          out1 = self.pool(spx[self.nums].permute((1,2,0,3,4)))
          out = torch.cat((out, out1.permute((2,0,1,3,4))),2)
        
        out = self.snn_conv3(out) #torch.Size([1, 1, 128, 60, 76])

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity

        return out

class MHSANet(nn.Module):
    def __init__(
        self,
        block: Type[Union[Bottle2neck]],
        layers: List[int],
        baseWidth: int = 26,
        scale: int = 4,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        T: int = 1,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.baseWidth = baseWidth
        self.scale = scale
        self.T = T
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Sequential(
                layer.SeqToANNContainer(
                    nn.Conv2d(
                        in_channels=3,
                        out_channels=self.inplanes,
                        kernel_size=7,
                        padding=3,
                        stride=2,
                        bias=False,
                    )
                ),
                batch_norm_2d(self.inplanes)
        )

        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3),stride=(1,2,2),padding=(0,1,1))   ### timewindow>1时  multistep maxpool

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
 

    def _make_layer(
        self,
        block: Type[Union[Bottle2neck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                layer.SeqToANNContainer(
                    nn.Conv2d(
                        in_channels=self.inplanes,
                        out_channels=planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    )
                ),
                batch_norm_2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.baseWidth, previous_dilation, norm_layer, scale=self.scale, stype='stage',
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.baseWidth,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    scale=self.scale,
                )
            )

        return nn.Sequential(*layers)


    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        # torch.autograd.set_detect_anomaly(True)
        outputs= []
        input = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)

        output = self.conv1(input) #input [1,1,3,240,304]  output[1,1,64,120,152]

        output = self.maxpool(output.permute((1,2,0,3,4)))  ## timewindow>1时的 multistep maxpool
        output = output.permute((2,0,1,3,4))

        output = self.layer1(output) #torch.Size([1, 1, 128, 60, 76])
        output = self.layer2(output) #torch.Size([1, 1, 256, 30, 38])
        outputs.append(output)
        output = self.layer3(output) #torch.Size([1, 1, 512, 15, 19])
        outputs.append(output)
        output = self.layer4(output) #torch.Size([1, 1, 1024, 8, 10])
        outputs.append(output)

        return outputs

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)



def MHSANet50_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MHSANet(Bottle2neck, [3, 4, 6, 3], baseWidth = 26, scale = 4, **kwargs)

    return model


def MHSANet50_4x24w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs["groups"] = 4

    model = MHSANet(Bottle2neck, [3, 4, 6, 3], baseWidth = 24, scale = 4, **kwargs)

    return model