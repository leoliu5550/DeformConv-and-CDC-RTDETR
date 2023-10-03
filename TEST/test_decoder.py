import sys,pytest
sys.path.append(".")
import torch
import torch.nn as nn
import random
from model.decoder import *
from model.backbone import *
from model.hybrid_encoder import *

class Testdecoder:
    device = 'cuda:1'

    def test_MLP(self):
        model = MLP(
            input_dim = 256,
            hidden_dim = 512,
            output_dim = 1024,
            num_layers= 2,
            act='gelu'
        ).to(self.device)
        x = torch.ones([16,256]).to(self.device)
        assert model(x).shape == torch.Size([16,1024])

    @pytest. mark. skip(reason="passnow because i dont have any choice")
    def test_MSDeformableAttention(self):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        batch_szie = 4
        query_length = 256
        value_length = 1024
        classify = 1024
        n_levels = 4


        querysize = torch.ones([batch_szie,query_length,classify]).to(self.device)
        reference_point = torch.ones([batch_szie,query_length,n_levels,2]).to(self.device)
        value = torch.ones([batch_szie,value_length,classify]).to(self.device)
        value_spatial_shapes = [(random.random(),random.random()) for _ in range(n_levels) ]
        value_mask = None

        model = MSDeformableAttention(
            embed_dim = 256 ,
            num_heads = 8 ,
            num_levels = n_levels ,
            num_points = 4
        ).to(self.device)

        output = model(
            query = querysize,
            reference_points =reference_point,
            value =value,
            value_spatial_shapes = value_spatial_shapes,
            value_mask =value_mask
        ).to(self.device)

        print()
        print(output)
        assert 1 ==2
                # value,
                # value_spatial_shapes,
                # value_mask=None):
    @pytest. mark. skip(reason="passnow because i dont have any choice too")
    def test_TransformerDecoderLayer(self):
        D_MODELs = 512
        nhead = 8
        model = TransformerDecoderLayer(
            d_model=D_MODELs,
            n_head=nhead,
            dim_feedforward = 1024,
            activation= "silu"
        ).to(self.device)
        x = torch.ones([9,3,D_MODELs]).to(self.device)
        output = model(x)
        print()

        print(output.shape)
        assert 1 == 22



    def test_RTDETRTransformer(self):

        x = torch.ones(1,3,800,800).to(self.device)

        bmodel = Backbone(
            backbone='resnet50',
            norm_layer=None
        ).to(self.device)


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
        ).to(self.device)

        model = RTDETRTransformer(                
                num_classes=80,
                hidden_dim=256,
                num_queries=300,
                position_embed_type='sine',
                feat_channels=[512, 1024, 2048],
                feat_strides=[8, 16, 32],
                num_levels=3,
                num_decoder_points=4,
                nhead=8,
                num_decoder_layers=6,
                dim_feedforward=1024,
                dropout=0.,
                activation="relu",
                num_denoising=100,
                label_noise_ratio=0.5,
                box_noise_scale=1.0,
                learnt_init_query=False,
                eval_spatial_size=None,
                eval_idx=-1,
                eps=1e-2, 
                aux_loss=True).to(self.device)

        out = bmodel(x)
        out = hybird(out)
        print()
        print("#"*80)
        # print(out)
        for it in out:
            print(it.shape)
        print("here")
        out = model(out)


        print()
        print("#"*80)
        print(out.shape)
        assert 11==22
