import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import FeaturePyramidNetwork
torch.cuda.empty_cache()

device = "cuda:1" if torch.cuda.is_available() else "cpu"
weights = None
net = models.resnet50(weights=weights)
x = torch.ones([60,3,800,800]).to(device)
new_net = IntermediateLayerGetter(net,{'layer2':'feat2', 'layer3': 'feat3','layer4': 'feat4'}).to(device)
# print(new_net(x).items())
print([(k, v.shape) for k, v in new_net(x).items()])
print(list(new_net(x).items())[0][1])
# print(new_net(x).shape)
# print("#"*80)
# m = FeaturePyramidNetwork([512,1024,2048], 2048)
# output = m(new_net(x))
# print([(k, v.shape) for k, v in output.items()])