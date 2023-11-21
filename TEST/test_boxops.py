import pytest
import sys
sys.path.append(".")
import torch
import torch.nn
import dynamic_yaml
from model.box_ops import *
from torchvision.ops.boxes import box_area
import logging
import logging.config
logging.config.fileConfig('logging.conf')
logger = logging.getLogger("test.boxops")



class Testbox:
    with open("model_config.yaml") as file:
        cfg = dynamic_yaml.load(file)
    device = cfg['device']
    def test_box_cxcywh_to_xyxy(self):
        x = torch.ones([1, 4]).to(self.device)*0.5*800
        logger.debug(f"origin \n {x}")
        box_trans = box_cxcywh_to_xyxy(x)
        logger.debug(f"origin_box_trans \n {box_trans}"  ) 
        std_ans=torch.tensor([[200., 200., 600., 600.]], device=self.device)
        assert torch.equal(box_trans,std_ans) == True

        logger.debug(f"trans_orgin \n {box_trans/800}" ) 
        logger.debug(f"trans_stdans \n {std_ans/800}" ) 
        assert torch.equal(box_trans/800,std_ans/800) == True

    def test_box_cxcywh_to_xyxy_2(self):
        x = torch.tensor([[0.0017, 0.3910, 0.9573, 0.9865]])
        # [-0.4770, -0.1022,  0.4803,  0.8843]
        logger.debug(f"x_box\n {x}" )
        box_trans = box_cxcywh_to_xyxy(x)
        logger.debug(f"box_trans \n {box_trans}"  ) 
        assert 1==11

    def test_box_xyxy_to_cxcywh(self):
        x = torch.tensor([[0.2500, 0.2500, 0.7500, 0.7500]], device=self.device)

        logger.debug(f"origin \n {x}")
        box_trans = box_xyxy_to_cxcywh(x)
        logger.debug(f"origin \n {box_trans}"  ) 
        std_ans=torch.tensor([[0.5, 0.5,0.5,0.5]], device=self.device)
        assert torch.equal(box_trans,std_ans) == True

    def test_box_area(self):
        x = torch.tensor([[0.2500, 0.2500, 0.7500, 0.7500]], device=self.device)
        area = box_area(x)
        logger.debug(f"box_area : {area}")
        assert area == torch.tensor(0.25)


    def test_box_iou(self):

        box1 = torch.tensor(
            [[0.01, 0.02, 0.0750, 0.0750],
            [0.03, 0.04, 0.0375, 0.0375],
            [0.05, 0.06, 0.0250, 0.0250]], device=self.device)
        logger.debug(f"\nbox1 = \n{box1}")
        box2 = torch.tensor(
            [[0.011, 0.022, 0.0750, 0.0750],
            [0.033, 0.044, 0.0375, 0.0375],
            [0.055, 0.066, 0.0250, 0.0250]], device=self.device)
        logger.debug(f"\nbox2 = \n{box2}")


        iou, union = box_iou(
            box1,
            box2
            )
            
        logger.debug(f"\niou = \n {iou}")
        logger.debug(f"\nunion =  \n {union}")



        assert iou == torch.tensor([[1.]], device=self.device)
        assert union == torch.tensor([[0.2500]], device=self.device)