import sys,pytest
sys.path.append(".")
import torch
import torch.nn as nn
import random
from model.decoder import RTDETRTransformer
from model.backbone import *
from model.hybrid_encoder import *

device = 'cpu'#'cuda:0'
x = torch.ones(1,3,800,800)
bmodel = Backbone(
            backbone='resnet50',
            norm_layer=None
        )


hybird = HybridEncoder(
    in_channels=[512, 1024, 2048],
    feat_strides=[8, 16, 32],
    hidden_dim=1024,
    nhead=8,
    dim_feedforward = 1024,
    dropout=0.0,
    enc_act='gelu',
    use_encoder_idx=[2],
    num_encoder_layers=1,
    pe_temperature=10000,
    expansion=1.0,
    depth_mult=1.0,
    act='silu',
    eval_size=None
)



model = RTDETRTransformer(                
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        position_embed_type='sine',
        feat_channels=[1024, 1024, 1024],# here exist problem
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_decoder_points=4,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.,
        activation="relu",
        num_denoising=0,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
        eval_spatial_size=None,
        eval_idx=-1,
        eps=1e-2, 
        aux_loss=True).to(device)

out = bmodel(x)
out = hybird(out)
print("#"*80)
print("before decoder")

for it in out:
    print(it.shape)
print("#"*80)
out = model(out)


print()
print("#"*80)
print(out.keys())