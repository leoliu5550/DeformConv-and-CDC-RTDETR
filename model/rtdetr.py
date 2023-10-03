import torch 
import torch.nn as nn
import dynamic_yaml
from .backbone import Backbone
from .hybrid_encoder import HybridEncoder
from .decoder import RTDETRTransformer


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
            eval_size = hybird_cfg.eval_size
        )

        # 

    def forward(self,x):
        x = self.backbone(x)
        x = self.HybirdEncoder(x)
        return x