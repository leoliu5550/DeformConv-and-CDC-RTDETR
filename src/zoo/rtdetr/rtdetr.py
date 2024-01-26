"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 

from src.core import register

# region
import logging
import logging.config
logging.config.fileConfig('logging.conf')
logtracker = logging.getLogger(f"model.{__name__}")
# endregion

__all__ = ['RTDETR', ]


@register
class RTDETR(nn.Module):
    __inject__ = ['backbone','subbone','catmodul', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module,subbone:nn.Module,catmodul:nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.subbone = subbone
        self.catmodul = catmodul
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        
    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])
            
        b_x = self.backbone(x)
        sub_x = self.subbone(x)
        att_x = self.catmodul(b_x,sub_x)
        logtracker.debug("after att_x")
        x = self.encoder(att_x )      

        x = self.decoder(x, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
