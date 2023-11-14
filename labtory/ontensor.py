import torch
from torch import tensor
# target = torch.Tensor([25, 25])
# a = [0,1]
# # for i,b in zip(target,a):
# #     print(i,b)
# print([ [i,b] for i,b in zip(target,a)])
# print(torch.cat([ [i,b] for i,b in zip(target,a)]))

# torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
from torch.utils.data import default_collate
targets = tuple([{'boxes': tensor([[481.9125, 112.7324, 750.6250, 670.7794],
        [ 66.2625, 669.4647, 231.3000, 773.1080]]), 'labels': tensor([25, 25]), 'image_id': tensor([25]), 'area': tensor([46212.6719,  6539.5483]), 'iscrowd': tensor([0, 0]), 'orig_size': tensor([640, 426]), 'size': tensor([800, 800])} for _ in range(3)])
print(targets)
targets = [dic.to("cuda:0") for dic in targets]
print(targets)
# print(tensor(targets))
# print(torch.cat([v["labels"] for v in [targets]]))
# print(ans)