import sys,os
sys.path.append(".")
print(os.getcwd())
from src import *

from src.core import YAMLConfig 
from src.solver import TASKS
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F 

import logging



import logging.config
logging.config.fileConfig('logging.conf')
logtracker = logging.getLogger(f"test.{__name__}")

logtracker.debug(f"\n {TASKS}")


cfg = YAMLConfig(   
        "configs/rtdetr/rtdetr_r50vd_6x_coco.yml"
)
# solver = TASKS[cfg.yaml_cfg['task']](cfg)
# use yaml call certain task, but yaml only have detection
# logtracker.debug(f"\n {cfg.yaml_cfg['task']}")

# for itm in dir(cfg):
#     print(itm)
    # print(cfg)
with open("model_info2.txt","w") as file:
    file.write(str(cfg.model))
# print(cfg.find_unused_parameters)
# print(cfg.sync_bn)
# class resnet:
#     __inject__ = ['backbone']
#     def __init__(self, backbone: nn.Module):
#         super().__init__()
#         self.backbone = backbone

#     def forward(self, x, targets=None):
#         if self.multi_scale and self.training:
#             sz = np.random.choice(self.multi_scale)
#             x = F.interpolate(x, size=[sz, sz])
            
#         x = self.backbone(x)
#         return x
    
    
# model = resnet()