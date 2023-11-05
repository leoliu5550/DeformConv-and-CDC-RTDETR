import sys
sys.path.append(".")
import torch
import pytest
import yaml
from data import *

class TestCoCo:
    with open("model_config.yaml") as file:
        cfg = yaml.safe_load(file)
    cfg = cfg['data']['train_dataloader']['dataset']
    Coco_loader = CocoDetection(
        img_folder = cfg['img_folder'],
        ann_file = cfg['ann_file'],
        # transforms = None,
        return_masks = False
    )
    def test_Coco_loader(self):
        data = self.Coco_loader.__getitem__(1)
        # print(data[0].shape)
        assert data[0].shape == torch.Size([3, 426, 640])
        print(data[1]['orig_size'])
        
    def test_Coco_Resize(self):
        data = self.Coco_loader.__getitem__(1)
        rimg =resize(data[0],data[1])
        print(rimg[1])
        assert rimg[0].shape == torch.Size([3, 800, 800])
        



