import sys,pytest
sys.path.append(".")
import torch
import torch.nn as nn
from model.rtdetr import rtdetr
import dynamic_yaml


class Testrtdetr:

    with open('model_config.yaml') as file:
        cfg = dynamic_yaml.load(file)
    
    model = rtdetr('model_config.yaml').to(cfg.device)
    @pytest. mark. skip(reason="dont test it now")
    def test_rtdetr(self):
        x =torch.ones([4,3,800,800]).to(self.cfg.device)
        output = self.model(x)
        wh = [100,50,25]

        for i,feat in enumerate(output):
            # print(i,feat.shape)
            assert feat.shape == torch.Size([4,self.cfg.model.hybirdencoder.hidden_dim,wh[i],wh[i]])