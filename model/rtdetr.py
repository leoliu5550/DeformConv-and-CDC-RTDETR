import torch.nn as nn
import torch.nn.functional as F 

from .backbone import Backbone
from .hybrid_encoder import HybridEncoder
from .decoder import RTDETRTransformer as decoder

import dynamic_yaml


class rtdetr(nn.Module):
    def __init__(self,yaml_file):
        super().__init__()
        # load model setting
        with open(yaml_file) as fileobj:
            cfg = dynamic_yaml.load(fileobj)


        # backbone part
        backbone_cfg = cfg.model.backbone
        self.backbone = Backbone(
            backbone= backbone_cfg.backbone,
            norm_layer= backbone_cfg.norm_layer
        )
        # hybirdencoder part
        hybird_cfg = cfg.model.hybirdencoder
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

        
        # decoder part
        decoder_cfg = cfg.model.RTDETRTransformer
        self.decoder = decoder(
            num_classes=decoder_cfg.num_classes,
            hidden_dim=decoder_cfg.hidden_dim,
            num_queries=decoder_cfg.num_queries,
            position_embed_type=decoder_cfg.position_embed_type,
            feat_channels=decoder_cfg.feat_channels,
            feat_strides=decoder_cfg.feat_strides,
            num_levels=decoder_cfg.num_levels,
            num_decoder_points=decoder_cfg.num_decoder_points,
            nhead=decoder_cfg.nhead,
            num_decoder_layers=decoder_cfg.num_decoder_layers,
            dim_feedforward=decoder_cfg.dim_feedforward,
            dropout=decoder_cfg.dropout,
            activation=decoder_cfg.activation,
            num_denoising=decoder_cfg.num_denoising,
            label_noise_ratio=decoder_cfg.label_noise_ratio,
            box_noise_scale= decoder_cfg.box_noise_scale,
            learnt_init_query=decoder_cfg.learnt_init_query,
            eval_spatial_size=decoder_cfg.eval_spatial_size,
            eval_idx=decoder_cfg.eval_idx,
            eps=decoder_cfg.eps, 
            aux_loss=decoder_cfg.aux_loss
        )

    def forward(self,x):
        x = self.backbone(x)
        x = self.HybirdEncoder(x)
        x = self.decoder(x)
        return x
    


# class RTDETR(nn.Module):
#     __inject__ = ['backbone', 'encoder', 'decoder', ]

#     def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
#         super().__init__()
#         self.backbone = backbone
#         self.decoder = decoder
#         self.encoder = encoder
#         self.multi_scale = multi_scale
        
#     def forward(self, x, targets=None):
#         if self.multi_scale and self.training:
#             sz = np.random.choice(self.multi_scale)
#             x = F.interpolate(x, size=[sz, sz])
            
#         x = self.backbone(x)
#         x = self.encoder(x)        
#         x = self.decoder(x, targets)

#         return x
    
#     def deploy(self, ):
#         self.eval()
#         for m in self.modules():
#             if hasattr(m, 'convert_to_deploy'):
#                 m.convert_to_deploy()
#         return self 