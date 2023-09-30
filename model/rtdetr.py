from backbone import  Backbone
from .hybrid_encoder import HybridEncoder
from .decoder import RTDETRTransformer
import torch 
import torch.nn as nn

class RTDETR(nn.Module):
    def __init__(self,yaml_file):
        super().__init__()
        self.backbone = Backbone
        self.HybirdEncoder = HybridEncoder
        self.RTDETRTransformer = RTDETRTransformer

    def forward(image):

        return image