import torch 
import torch.nn as nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
# from torchvision.ops import FeaturePyramidNetwork
from .comm.common import FrozenBatchNorm2d

class Backbone(nn.Module):
    def __init__(self,
                backbone:str,
                weights=None,
                dilation=False,
                norm_layer= False
):
        super().__init__()
        if norm_layer == False:
            norm_layer = FrozenBatchNorm2d
        else:
            norm_layer = None
        origin_model = getattr(torchvision.models, backbone)(
            replace_stride_with_dilation=[False, False, dilation],
            weights=weights,
            norm_layer=norm_layer
        )
        self.model = IntermediateLayerGetter(origin_model,{'layer2': 'feat1','layer3':'feat2', 'layer4': 'feat3'})
        # self.FPNmodel = FeaturePyramidNetwork([512,1024,2048], 2048)

    def forward(self,x):
        out = self.model(x)
        # out =[ans['feat1'],ans['feat2'],ans['feat3']]
        # out = [ item[1] for item in out]
        # # out = self.FPNmodel(out)
        return out

# device = 'cuda:1'
# Backbone_model = Backbone(
#             backbone='resnet50',
#             out_channel = 1024,
#             norm_layer=None
#         ).to(device)
# x = torch.ones([1,3,100,100]).to(device)
# output = Backbone_model(x)
# print(output)