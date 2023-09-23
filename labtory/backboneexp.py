import torch
import torch.nn as nn
import torchvision.models as models
torch.cuda.empty_cache()

device = "cuda:1" if torch.cuda.is_available() else "cpu"
# print(torch.cuda.memory_allocated(device))
# for key,value in net.named_parameters():
#     print(key,value)
# print(net)
# models.resnet50(weights=weights)
# models.resnet18(weights=weights)
# models.resnet101(weights=weights)


# x = torch.ones([2,3,800,800]).to(device)
# net = models.densenet161().to(device)
# out = net(x)


# model = nn.Sequential(*features)
# model = model.to(device)
# x = torch.rand([1,3,244,244]).to(device)
# out = model(x)

# if USE_FPN: # 多层输出
#     for key,value in out.items():
#         print(key, value.shape)
import torchvision
print("#"*80)
m = torchvision.models.resnet18()
# extract layer1 and layer3, giving as names `feat1` and feat2`
new_m = torchvision.models._utils.IntermediateLayerGetter(m,{'layer1': 'feat1','layer2':'feat2', 'layer3': 'feat3'})
out = new_m(torch.rand(2, 3, 800, 800))
print([(k, v.shape) for k, v in out.items()])

m = torchvision.ops.FeaturePyramidNetwork([64,128,256], 64)
output = m(out)
print([(k, v.shape) for k, v in output.items()])
print("#"*80)
m = torchvision.models.resnet50()
# extract layer1 and layer3, giving as names `feat1` and feat2`
new_m = torchvision.models._utils.IntermediateLayerGetter(m,{'layer1': 'feat1','layer2':'feat2', 'layer3': 'feat3'})
out = new_m(torch.rand(2, 3, 800, 800))
print([(k, v.shape) for k, v in out.items()])

m = torchvision.ops.FeaturePyramidNetwork([256,512,1024], 64)
output = m(out)
print([(k, v.shape) for k, v in output.items()])
print("#"*80)
m = torchvision.models.resnet101()
# extract layer1 and layer3, giving as names `feat1` and feat2`
new_m = torchvision.models._utils.IntermediateLayerGetter(m,{'layer1': 'feat1','layer2':'feat2', 'layer3': 'feat3'})
out = new_m(torch.rand(2, 3, 800, 800))
print([(k, v.shape) for k, v in out.items()])

m = torchvision.ops.FeaturePyramidNetwork([256,512,1024], 64)
output = m(out)
print([(k, v.shape) for k, v in output.items()])