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

    data = Coco_loader.__getitem__(1)
    def test_Coco_loader(self):
        assert self.data[0].shape == torch.Size([3, 426, 640])
        # print(self.data[1]['orig_size'])
        
    def test_Coco_Resize(self):
        # resize input image and target
        rimg =resize(self.data[0],self.data[1])
        print(rimg[1])
        assert rimg[0].shape == torch.Size([3, 800, 800])
        

    def test_Coco_labelreshape(self):
        img =resize(self.data[0],self.data[1])
        # print(img[1]['labels'])
        labels = torch.tensor([25,25])
        # print(labels)
        assert torch.equal(img[1]['labels'],labels) == True
