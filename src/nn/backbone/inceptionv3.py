import torch
import torch.nn as nn 
import torch.nn.functional as F 

from collections import OrderedDict

from .common import get_activation, ConvNormLayer, ConvTranspose

from src.core import register

__all__ = ['inceptionv3']

from torchvision.models import inception_v3


@register
class inceptionv3(nn.Module):
    standard_model = inception_v3(weights = "IMAGENET1K_V1")
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            self.standard_model.Conv2d_1a_3x3,
            self.standard_model.Conv2d_2a_3x3,
            self.standard_model.Conv2d_2b_3x3,
            self.standard_model.maxpool1,
            self.standard_model.Conv2d_3b_1x1,
            self.standard_model.Conv2d_4a_3x3,
            self.standard_model.maxpool2,
            self.standard_model.Mixed_5b,
            self.standard_model.Mixed_5c,
            self.standard_model.Mixed_5d
        )
        self.block2 = nn.Sequential(
            self.standard_model.Mixed_6a,
            self.standard_model.Mixed_6b,
            self.standard_model.Mixed_6c,
            self.standard_model.Mixed_6d,
            self.standard_model.Mixed_6e
        )
        self.block3 = nn.Sequential(
            self.standard_model.Mixed_7a,
            self.standard_model.Mixed_7b,
            self.standard_model.Mixed_7c
        )
        self.tconv = nn.ModuleList()

        self.tconv.append(
            nn.Sequential(
                ConvNormLayer(
                    ch_in = 288,ch_out = 256,kernel_size = 2,stride = 1,act = 'silu'),
                ConvTranspose(
                    ch_in = 256,ch_out = 256,kernel_size = 5,stride = 1,act = 'silu')

                )
        )
        
        self.tconv.append(
            nn.Sequential(
                ConvNormLayer(
                    ch_in = 768,ch_out = 256,kernel_size = 3,stride = 1,act = 'silu'),
                ConvTranspose(
                    ch_in = 256,ch_out = 256,kernel_size = 3,stride = 1,act = 'silu')
                )
        )
        
        self.tconv.append(
            nn.Sequential(
                ConvNormLayer(
                    ch_in = 2048,ch_out = 256,kernel_size = 3,stride = 1,act = 'silu'),
                ConvTranspose(
                    ch_in = 256,ch_out = 256,kernel_size = 3,stride = 1,act = 'silu')
                )
        )
        
        self.drop = nn.ModuleList(
            [nn.Dropout(p=0.8,inplace=False) for _ in range(3)])
        
    def forward(self,x):
        feat1 = self.drop[0](self.block1(x)) # [1, 288, 77, 77]
        feat2 = self.drop[1](self.block2(feat1)) # [1, 768, 38, 38]
        feat3 = self.drop[2](self.block3(feat2)) # [1, 2048, 18, 18]
        
        
        return [self.tconv[0](feat1),
                self.tconv[1](feat2),
                self.tconv[2](feat3)]
        