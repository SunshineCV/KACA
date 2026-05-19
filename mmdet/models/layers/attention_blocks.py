import warnings

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F

from mmdet.registry import MODELS
from .fastkan import FastKANLayer
import torch


from torch.nn.parameter import Parameter
@MODELS.register_module()
class GCTBlock(nn.Module):
    def __init__(self, in_channels, c=2, alpha=3, beta=1, is_learnable=False):
        super(GCTBlock, self).__init__()
        #GCT-B0: is_learnable=False:
        #GCT-B1: is_learnable=True: 
        #c: standard deviation
        #alpha and beta control the range of c
        if is_learnable:
            self.theta=nn.Parameter(torch.zeros(1))
            self.alpha = alpha
            self.beta = beta
            self.sig = nn.Sigmoid()
        self.c = c
        self.is_learnable = is_learnable
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = y - y.mean(dim=1, keepdim=True)
        std = y.std(dim=1, keepdim=True) + 1e-5
        y = y / std
        y = torch.pow(y, 2)
        if self.is_learnable:
            self.c = self.alpha*self.sig(self.theta) + self.beta
        y = torch.exp(-y / (2*self.c*self.c))
        out = y * x
        return out


@MODELS.register_module()
class LCTBlock(nn.Module):
    def __init__(self, in_channels, groups=32):
        super(LCTBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.groups = groups
        self.channels =  in_channels // groups
        self.weight = nn.Parameter(torch.zeros(1, self.channels))
        self.bias = nn.Parameter(torch.ones(1, self.channels))
        self.sig = nn.Sigmoid()


    def forward(self, x):
        b, c, h, w = x.size()
        y= self.avg_pool(x).view(b*self.groups, -1)
        y = self.weight * y + self.bias
        y = self.sig(y).view(b, c, 1, 1)
        out = x * y
        return out



@MODELS.register_module()
class ECABlock(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, in_channels, k_size=3):
        super(ECABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


def kaiming_init(module: nn.Module,
                 a: float = 0,
                 mode: str = 'fan_out',
                 nonlinearity: str = 'relu',
                 bias: float = 0,
                 distribution: str = 'normal') -> None:
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

@MODELS.register_module()
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


@MODELS.register_module()
class KACABlock(BaseModule):
    def __init__(self, in_channels, groups=64, denominator=0.5, grid_min=-2, grid_max=2, num_grids=12, use_layernorm=False, fusion='mul', pooling_type='avg'):
        super(KACABlock, self).__init__()
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg = nn.AdaptiveAvgPool2d(1)
        self.pooling_type = pooling_type
        self.channels = in_channels // groups
        self.fusion = fusion
        self.groups = groups
        self.kan_layer = FastKANLayer(self.channels, self.channels, grid_min=grid_min, grid_max=grid_max, num_grids=num_grids, use_layernorm=use_layernorm, use_base_update=True, \
                base_activation=torch.sigmoid, denominator=denominator)
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg(x)
        return context

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.spatial_pool(x).view(b*self.groups, self.channels)
        y = self.kan_layer(y)
        y = y.view(b, c, 1, 1)
        if self.fusion == 'mul':
            out = x * torch.sigmoid(y)
        elif self.fusion == 'add':
            out = x + y
        return out
