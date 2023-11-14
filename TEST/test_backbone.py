import sys,pytest
sys.path.append(".")
import torch
import torch.nn as nn
from model.backbone import Backbone
from model.comm.common import FrozenBatchNorm2d
import dynamic_yaml

class Testbackbone:
    with open('model_config.yaml') as fileobj:
        cfg = dynamic_yaml.load(fileobj)
    device= cfg.device
    backbone= cfg.model.backbone.backbone
    norm_layer= cfg.model.backbone.norm_layer



    x = torch.ones([2,3,800,800]).to(device)
    # @pytest. mark. skip(reason="no way of currently testing this")
    def test_FRbackbone(self):
        Backbone_model = Backbone(
            backbone='resnet50',
            norm_layer=False
        ).to(self.device)
        # output = list(Backbone_model(self.x).items())
        output = Backbone_model(self.x)
        assert output['feat1'].shape == torch.Size([2, 512, 100, 100])#512
        assert output['feat2'].shape == torch.Size([2, 1024, 50, 50]) #1024
        assert output['feat3'].shape == torch.Size([2, 2048, 25, 25])

    # @pytest. mark. skip(reason="change to with out FPN")
    def test_BNbackbone2(self):
        Backbone_model = Backbone(
            backbone='resnet50',
            norm_layer=None
        ).to(self.device)
        output = Backbone_model(self.x)
        assert output['feat1'].shape == torch.Size([2, 512, 100, 100])#512
        assert output['feat2'].shape == torch.Size([2, 1024, 50, 50]) #1024
        assert output['feat3'].shape == torch.Size([2, 2048, 25, 25])

    @pytest. mark. skip(reason="change to orderdict type")
    def test_BNbackbone3(self):
        Backbone_model = Backbone(
            backbone='resnet50',
            out_channel = 1024,
            norm_layer=None
        ).to(self.device)
        output = Backbone_model(self.x)
        assert output[0][1].shape == torch.Size([2, 2048, 100, 100])#512
        assert output[1][1].shape == torch.Size([2, 2048, 50, 50]) #1024
        assert output[2][1].shape == torch.Size([2, 2048, 25, 25])



