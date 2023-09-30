import sys,pytest
sys.path.append(".")
import torch
import torch.nn as nn
from model.decoder import *


class Testdecoder:
    device = 'cuda:1'

    def test_MLP(self):
        model = MLP(
            input_dim = 256,
            hidden_dim = 512,
            output_dim = 1024,
            num_layers= 2,
            act='gelu'
        )
        x = torch.ones([16,256])
        assert model(x).shape == torch.Size([16,1024])


    def test_MSDeformableAttention(self):
        model = MSDeformableAttention(
            embed_dim = 256 ,
            num_heads = 8 ,
            num_levels = 4 ,
            num_points = 4
        )