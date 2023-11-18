import pytest
import sys
sys.path.append(".")
import torch
import torch.nn
import dynamic_yaml
# from model.box_ops import *

import logging
import logging.config
logging.config.fileConfig('logging.conf')
logger = logging.getLogger("test.boxops")



# class Testbox:
#     with open("model_config.yaml") as file:
#         cfg = dynamic_yaml.load(file)
#     device = cfg['device']
#     def test_box_cxcywh_to_xyxy(self):
#         x = torch.ones([2, 2,4]).to(self.device)
        
#         box_trans = box_cxcywh_to_xyxy()

#         assert x.shape == x.shape



logger.debug("dsfgsd")