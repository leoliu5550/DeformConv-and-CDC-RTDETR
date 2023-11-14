import sys
sys.path.append(".")
import torch
import pytest
import yaml
from data import *

# class TestCoCo:
#     with open("model_config.yaml") as file:
#         cfg = yaml.safe_load(file)
#     cfg = cfg['data']['train_dataloader']['dataset']
#     Coco_loader = CocoDetection(
#         img_folder = cfg['img_folder'],
#         ann_file = cfg['ann_file'],
#         # transforms = None,
#         return_masks = False
#     )

#     data = Coco_loader.__getitem__(1)
#     def test_Coco_loader(self):
#         assert self.data[0].shape == torch.Size([3, 426, 640])
#         # print(self.data[1]['orig_size'])
        
#     def test_Coco_Resize(self):
#         # resize input image and target
#         rimg =resize(self.data[0],self.data[1])
#         print(rimg[1])
#         assert rimg[0].shape == torch.Size([3, 800, 800])
        

#     def test_Coco_labelreshape(self):
#         img =resize(self.data[0],self.data[1])
#         # print(img[1]['labels'])
#         labels = torch.tensor([25,25])
#         # print(labels)
#         assert torch.equal(img[1]['labels'],labels) == True


# from torch.utils.data import  DataLoader
cfg_path = "model_config.yaml"
with open(cfg_path,"r") as file:
    cfg = yaml.safe_load(file)



    
# data_cfg = cfg['data'] 
# training_cfg = cfg['train']
# def collate_fn(batch):
#     return tuple(zip(*batch))
# train_dataset = DataLoader(
#         CocoDetection(
#                 img_folder = data_cfg['train_dataloader']['dataset']['img_folder'],
#                 ann_file = data_cfg['train_dataloader']['dataset']['ann_file'],
#                 return_masks = False,
#                 yaml_path=cfg_path
#             ),
#         batch_size=data_cfg['train_dataloader']['batch_size'],
#         shuffle=data_cfg['train_dataloader']['shuffle'],
#         collate_fn=collate_fn
#     )
# # training_cfg['epoch']
# for epoch in range(1):
#     for batch, (data, target) in enumerate(train_dataset, 1):

#         data = torch.stack(data).to(cfg['device'])
#         # for tar in target:
#         #     for key, value in tar.items():
#         #         tar[key] = tar[key].to(cfg['device'])
#         # print(data.size())
#         print(target)
        
#     break

cfg = cfg['data']['train_dataloader']['dataset']
Coco_loader = CocoDetection(
    img_folder = cfg['img_folder'],
    ann_file = cfg['ann_file'],
    # transforms = None,
    return_masks = False,
    yaml_path=cfg_path
)

data = Coco_loader.__getitem__(1)
print(data[1])
# print(data[0].shape)

from torchvision.utils import save_image

save_image(data[0], 'img3.png')