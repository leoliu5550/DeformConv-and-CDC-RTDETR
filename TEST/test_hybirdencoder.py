import sys,pytest
sys.path.append(".")
import torch
import torch.nn as nn
from model.hybrid_encoder import *
from model.backbone import Backbone
from model.comm.common import FrozenBatchNorm2d
import dynamic_yaml 



class Testencoder:
    with open("model_config.yaml") as file:
        cfg = dynamic_yaml.load(file)
    device = cfg['device']
    D_MODEL = 512
    def test_ConvNormlayer(self):
        x = torch.ones([2, 512, 100, 100]).to(self.device)
        model = ConvNormLayer(
            ch_in=512,
            ch_out=512,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True).to(self.device)
        assert model(x).shape == x.shape

    def test_RepVggblock(self):
        x = torch.ones([2, 512, 100, 100]).to(self.device)
        model = RepVggBlock(
            ch_in=512,
            ch_out=512,
            act='silu').to(self.device)
        assert model(x).shape == x.shape

    def test_CSPRepLayer(self):
        x = torch.ones([2, 512, 100, 100]).to(self.device)
        model = CSPRepLayer(
            in_channels=512,
            out_channels=512,
            num_blocks=3,
            bias= True,
            act='gelu'
        ).to(self.device)
        assert model(x).shape == x.shape
    
    def test_TransformerEncoderLayer(self):
        x = torch.ones([9,3,self.D_MODEL]).to(self.device)
        model = TransformerEncoderLayer(
            d_model=self.D_MODEL,
            nhead=16,
            dim_feedforward=2048
        ).to(self.device)
        assert model(x).shape == torch.Size([9,3,self.D_MODEL])

    def test_TransformerEncoder(self):
        encoder = TransformerEncoderLayer(
            d_model=self.D_MODEL,
            nhead=16,
            dim_feedforward=512
        )
        
        model = TransformerEncoder(
            encoder_layer=encoder,
            num_layers=3
        ).to(self.device)
        x = torch.ones([9,64,self.D_MODEL]).to(self.device)
        assert model(x).shape == torch.Size([9,64,self.D_MODEL])


    def test_HybirdEncoder(self):
        x = torch.ones([2,3,800,800]).to(self.device)
        Backbone_model = Backbone(
            backbone='resnet50',
            norm_layer=FrozenBatchNorm2d
        ).to(self.device)
        # Backbone_model.to(self.device)


        output = Backbone_model(x)
        hidden_dim = 256
        # b_out = torch.stack((output['feat1'],output['feat2'],output['feat3']),dim=0)
        model = HybridEncoder(
            in_channels=[512, 1024, 2048],
            feat_strides=[8, 16, 32],
            hidden_dim=hidden_dim,
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
            eval_spatial_size = None
        ).to(self.device)
        out = model(output)

        wh = [100,50,25]
        for i,feat in enumerate(out):
            # print(i,feat.shape)
            assert feat.shape == torch.Size([2,hidden_dim,wh[i],wh[i]])
            # 0 torch.Size([2, 256, 100, 100])
            # 1 torch.Size([2, 256, 50, 50])
            # 2 torch.Size([2, 256, 25, 25])

# class Testbackbone:
#     device= 'cuda:1'
#     backbone='resnet50'
#     out_channel = 1024
#     norm_layer=None



#     x = torch.ones([2,3,800,800]).to(device)
#     # @pytest. mark. skip(reason="no way of currently testing this")
#     def test_FRbackbone(self):
#         Backbone_model = Backbone(
#             backbone='resnet50',
#             out_channel = 1024,
#             norm_layer=FrozenBatchNorm2d
#         ).to(self.device)
#         output = list(Backbone_model(self.x).items())
#         assert output[0][1].shape == torch.Size([2, 512, 100, 100])#512
#         assert output[1][1].shape == torch.Size([2, 1024, 50, 50]) #1024
#         assert output[2][1].shape == torch.Size([2, 2048, 25, 25])

#     @pytest. mark. skip(reason="change to with out FPN")
#     def test_BNbackbone2(self):
#         Backbone_model = Backbone(
#             backbone='resnet50',
#             out_channel = 1024,
#             norm_layer=None
#         ).to(self.device)
#         output = list(Backbone_model(self.x).items())
#         assert output[0][1].shape == torch.Size([2, 2048, 100, 100])#512
#         assert output[1][1].shape == torch.Size([2, 2048, 50, 50]) #1024
#         assert output[2][1].shape == torch.Size([2, 2048, 25, 25])
