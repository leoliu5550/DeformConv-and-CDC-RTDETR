import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import FeaturePyramidNetwork
torch.cuda.empty_cache()

device = "cuda:1" if torch.cuda.is_available() else "cpu"
weights = None
net = models.resnet50(weights=weights)
x = torch.ones([2,3,4,4])
new_net = IntermediateLayerGetter(net,{'layer2':'feat2', 'layer3': 'feat3','layer4': 'feat4'})
# print([(k, v.shape[1]) for k, v in new_net(x).items()])
print(new_net(x))
# print("#"*80)
# m = FeaturePyramidNetwork([512,1024,2048], 2048)
# output = m(new_net(x))
# print([(k, v.shape) for k, v in output.items()])