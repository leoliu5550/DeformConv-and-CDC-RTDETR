import torch 
import torch.nn as nn
import torch.nn.functional as F 

from .backbone import Backbone
from .hybrid_encoder import HybridEncoder
from .decoder import RTDETRTransformer as decoder

import numpy as np
import dynamic_yaml


class rtdetr(nn.Module):
    def __init__(self,yaml_file):
        super().__init__()
        # load model setting
        with open(yaml_file) as fileobj:
            cfg = dynamic_yaml.load(fileobj)


        # backbone part
        backbone_cfg = cfg.backbone
        self.backbone = Backbone(
            backbone= backbone_cfg.backbone,
            norm_layer= backbone_cfg.norm_layer
        )
        # hybirdencoder part
        hybird_cfg = cfg.hybirdencoder
        self.HybirdEncoder = HybridEncoder(
            in_channels = hybird_cfg.in_channels,
            feat_strides = hybird_cfg.feat_strides,
            hidden_dim = hybird_cfg.hidden_dim,
            nhead = hybird_cfg.nhead,
            dim_feedforward = hybird_cfg.dim_feedforward,
            dropout = hybird_cfg.dropout,
            enc_act = hybird_cfg.enc_act,
            use_encoder_idx = hybird_cfg.use_encoder_idx,
            num_encoder_layers = hybird_cfg.num_encoder_layers,
            pe_temperature = hybird_cfg.pe_temperature,
            expansion = hybird_cfg.expansion,
            depth_mult = hybird_cfg.depth_mult,
            act = hybird_cfg.act,
            eval_spatial_size = hybird_cfg.eval_size
        )

        self.decoder = decoder(
            
        )
        # 

    def forward(self,x):

        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])

            
        x = self.backbone(x)
        x = self.HybirdEncoder(x)
        return x
    


class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        
    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])
            
        x = self.backbone(x)
        x = self.encoder(x)        
        x = self.decoder(x, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 