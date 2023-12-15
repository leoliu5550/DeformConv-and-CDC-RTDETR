label2category = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 6}
category2label = {v: k for k, v in label2category.items()}

import numpy as np
import torch

a = np.array(1)
print(a)
print("="*10)
b = torch.tensor(1)
print(b)
print(int(b))
print(label2category[int(b)])
# print(category2label[b])
from torch import tensor
a=  {'boxes': tensor([[0.3040, 0.1798, 0.7898, 0.8944]], device='cuda:0'), 'labels': tensor([77], device='cuda:0'), 'image_id': tensor([77901], device='cuda:0'), 'area': tensor([155385.5312], device='cuda:0'), 'iscrowd': tensor([0], device='cuda:0'), 'orig_size': tensor([377, 500], device='cuda:0'), 'size': tensor([800, 800], device='cuda:0')}