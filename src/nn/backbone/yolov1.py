import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from src.core import register

# region
import logging
import logging.config
logging.config.fileConfig('logging.conf')
logtracker = logging.getLogger(f"model.HybridEncoder.{__name__}")
__all__ = ['Yolov1']
# endregion

@register
class Yolov1(nn.Module):
    def __init__(self,in_channels = 3):
        architecture_config = [
            #Conv (kernl_size,out_put,stride,padding)
            (7, 64, 2, 3),
            "M",#MaxPooling (kernl_size =2 ,stride = 2)
            (15, 64, 1, 7),
            "M",
            # (1, 128, 1, 0),
            # (3, 256, 1, 1),
            # (1, 128, 1, 0),
            (3, 256, 1, 1),
            "feat1",
            "M",
            #[conv,Conv,repeat_times]
            # [(1, 256, 1, 0), (3, 512, 1, 1), 1],
            # (3, 256, 1, 1),
            # (1, 512, 1, 0),
            (3, 256, 1, 1),
            "feat2",
            "M",
            # [(1, 256, 1, 0), (3, 512, 1, 1), 1],
            # (3, 256, 1, 1),
            # (1, 512, 1, 0),
            (3, 256, 1, 1),
            "feat3"
        ]
        super(Yolov1,self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknetdict = self._create_conv_layers(self.architecture)

    def forward(self,x):
        
        feat1 = self.darknetdict['feat1'](x)
        feat2 = self.darknetdict['feat2'](feat1)
        feat3 = self.darknetdict['feat3'](feat2)
        # x = torch.flatten(x,start_dim=1)
        # [feat1,feat2,feat3]
        return [feat1,feat2,feat3]
    # {
    #         'feat1':feat1,
    #         'feat2':feat2,
    #         'feat3':feat3
    #     }
    def _create_conv_layers(self,architecture):
        layers = []
        subdict = {}
        in_channels = self.in_channels
        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels=in_channels,
                        out_channels=x[1],
                        kernel_size =x[0],
                        stride = x[2],
                        padding = x[3])]
                in_channels = x[1]
            elif x in ["feat1","feat2","feat3"]:
                subdict[x] = nn.Sequential(*layers)
                layers = []
            elif x == 'M':
                layers+=[
                    nn.MaxPool2d(kernel_size=(2,2),stride=2)
                    ]
            # elif x == 'M2':
            #     layers+=[
            #         nn.MaxPool2d(kernel_size=(2,2),stride=2)
            #         ]
            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers +=[
                        CNNBlock(
                            in_channels=in_channels,
                            out_channels=conv1[1],
                            kernel_size = conv1[0],
                            stride=conv1[2],
                            padding = conv1[3]
                        )]
                    layers +=[
                        CNNBlock(
                            in_channels=conv1[1],
                            out_channels=conv2[1],
                            kernel_size = conv2[0],
                            stride = conv2[2],
                            padding = conv2[3]
                        )]
                    in_channels = conv2[1]

        return nn.ModuleDict(subdict)
    
    
class CNNBlock(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super(CNNBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,bias=False,**kwargs) #
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self,x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)
        return x