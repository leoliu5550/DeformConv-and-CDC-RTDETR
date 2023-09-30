import torch 
import torch.nn as nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import FeaturePyramidNetwork
# from .common import FrozenBatchNorm2d
class Backbone(nn.Module):
    def __init__(self,
                backbone:str,
                # frozenbatch=False,
                weights=None,
                dilation=False,
                norm_layer= None
):
        super().__init__()
        origin_model = getattr(torchvision.models, backbone)(
            replace_stride_with_dilation=[False, False, dilation],
            weights=weights,
            norm_layer=norm_layer
        )
        self.model = IntermediateLayerGetter(origin_model,{'layer2': 'feat1','layer3':'feat2', 'layer4': 'feat3'})
        self.FPNmodel = FeaturePyramidNetwork([512,1024,2048], 2048)

    def forward(self,x):
        out = self.model(x)
        return out
    
device = 'cuda:0'

x = torch.ones([2,3,800,800]).to(device)
Backbone_model = Backbone(
    backbone='resnet50'
).to(device)
output = Backbone_model(x)

# sq = nn.Sequential(nn.Conv2d(1024, 16, kernel_size=1, bias=False),
#             nn.BatchNorm2d(16)
#             ).to(device)

# print(sq(output['feat2']))

# for t,i in enumerate(output):
#     print(output[i].type())
# torch.cuda.FloatTensor
# torch.cuda.FloatTensor
# torch.cuda.FloatTensor

class net2(nn.Module):
    def __init__(self,
                in_channels = [512, 1024, 2048]):
        super().__init__()
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, 16, kernel_size=1, bias=False),
                    nn.BatchNorm2d(16)
                )
            )
    def forward(self,feats):
        proj_feats = [self.input_proj[i](feats[feat]) for i, feat in enumerate(feats)]
        return proj_feats
    
net_ = net2().to(device)
output2 = net_(output)
print(output2)