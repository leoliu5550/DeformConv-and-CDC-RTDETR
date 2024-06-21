
import copy
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision.ops import deform_conv2d

from .utils import get_activation

from src.core import register
import math
import logging
import logging.config
logging.config.fileConfig('logging.conf')
logtracker = logging.getLogger(f"model.adapter.{__name__}")
__all__ = ['Adapter','CDCadapter','Deformadapter']


class DeformConvBlock(nn.Module):
    def __init__(self,ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act='relu'):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias
            )
        self.conv_offset = nn.Conv2d(
            ch_in, 
            (kernel_size**2)*2, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias
            )
        # nn.init.constant_(self.conv_offset.weight,0)
        nn.init.trunc_normal_(self.conv_offset.weight)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        offset = self.conv_offset(x)
        # print(f"offset shape = {offset.shape}")
        # mask = self.sig(self.conv_mask(x)) 

        out = deform_conv2d(
            input=x, offset=offset, 
            weight=self.conv.weight, 
            mask=None, padding=(1, 1))

        
        return out
    

class Conv2d_cdiffBlock(nn.Module):
    def __init__(self,
                ch_in, 
                ch_out, 
                kernel_size, 
                stride, 
                padding=None, 
                bias=False, 
                act='relu',
                theta=0.7):

        super().__init__() 
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        nn.init.normal_(self.conv.weight)
        self.theta = theta
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        out_normal = self.conv(x)
        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            # [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)

            kernel_diff = kernel_diff[:, :, None, None]

            out_diff = F.conv2d(
                input=x, 
                weight=kernel_diff, 
                bias=self.conv.bias, 
                stride=self.conv.stride, 
                padding=0
                )
            out = out_normal - self.theta * out_diff
            out = self.act(out)
            return out
        
class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

@register
class Adapter(nn.Module):
    def __init__(self,
                in_channels=[512, 1024, 2048],
                hidden_dim=256,
                act='silu'):
        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in enumerate(in_channels) :
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False,act=act),
                    nn.BatchNorm2d(hidden_dim)
                )
            )

    def forward(self, feats):
        # yolo_feats = self.yolov1(ori_x)
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        return proj_feats

@register
class CDCadapter(nn.Module):
    def __init__(self,
                in_channels=[512, 1024, 2048],
                hidden_dim=256,
                act='silu'):
        # channel projection
        self.input_proj = nn.ModuleList()
        for layer_idx,in_channel in enumerate(in_channels) :
            if layer_idx == len(in_channels)-1:
                self.input_proj.append(
                    nn.Sequential(
                        # let deform conv remain same size after deformconv
                        Conv2d_cdiffBlock(in_channel, hidden_dim, kernel_size=3, stride=1, padding=1,bias=False,act=act),
                        nn.BatchNorm2d(hidden_dim)
                    )
                )
            else:
                self.input_proj.append(
                    nn.Sequential(
                        nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                        nn.BatchNorm2d(hidden_dim)
                    )
                )

    def forward(self, feats):
        # yolo_feats = self.yolov1(ori_x)
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        return proj_feats
    
@register
class Deformadapter(nn.Module):
    def __init__(self,
                in_channels=[512, 1024, 2048],
                hidden_dim=256,
                act='silu',):
        # channel projection
        self.input_proj = nn.ModuleList()
        for layer_idx,in_channel in enumerate(in_channels) :
            if layer_idx == len(in_channels)-1:
                self.input_proj.append(
                    nn.Sequential(
                        # let deform conv remain same size after deformconv
                        DeformConvBlock(in_channel, hidden_dim, kernel_size=3, stride=1, padding=1,bias=False,act=act),
                        nn.BatchNorm2d(hidden_dim)
                    )
                )
            else:
                self.input_proj.append(
                    nn.Sequential(
                        nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                        nn.BatchNorm2d(hidden_dim)
                    )
                )

    def forward(self, feats):
        # yolo_feats = self.yolov1(ori_x)
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        return proj_feats
    
