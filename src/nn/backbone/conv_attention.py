import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from .utils import get_activation
from src.core import register

# region
import logging
import logging.config
logging.config.fileConfig('logging.conf')
logtracker = logging.getLogger(f"model.{__name__}")
# endregion

__all__ = ['cbam']

class spatial_attention(nn.Module):
    def __init__(self,kernel_size=3,stride =1,padding =None,act = 'silu'):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels = 2,
            out_channels = 1,
            kernel_size = kernel_size,
            stride = stride,
            padding = (kernel_size-1)//2 if padding is None else padding
        )
        self.sig = nn.Sigmoid()
        self.act = nn.Identity() if act is None else get_activation(act) 
        
    def forward(self,x):
        max_x,_ = torch.max( x , dim=1, keepdim=True)
        avg_x = torch.mean( x , dim=1, keepdim=True)
        attention = self.sig(self.conv(torch.cat((max_x,avg_x),1)) )
        out = x * attention
        return self.act(out)

# channel attention
# channel-based attention in convolutional neural networks. We produce a channel attention map by exploiting the inter-channel relationship of features.
class channel_attention(nn.Module):
    # 初始化, in_channel代表输入特征图的通道数, ratio代表第一个全连接的通道下降倍数
    def __init__(self, ch_in, ratio=4):
        # 继承父类初始化方法
        super().__init__()
        
        # 全局最大池化 [b,c,h,w]==>[b,c,1,1]
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        
        # 第一个全连接层, 通道数下降4倍
        self.fc1 = nn.Linear(in_features=ch_in, out_features=ch_in//ratio, bias=False)
        # 第二个全连接层, 恢复通道数
        self.fc2 = nn.Linear(in_features=ch_in//ratio, out_features=ch_in, bias=False)
        
        # relu激活函数
        self.relu = nn.ReLU()
        # sigmoid激活函数
        self.sigmoid = nn.Sigmoid()
    
    # 前向传播
    def forward(self, inputs):
        # 获取输入特征图的shape
        b, c, _, _ = inputs.shape
        
        # 输入图像做全局最大池化 [b,c,h,w]==>[b,c,1,1]
        max_pool = self.max_pool(inputs)
        # 输入图像的全局平均池化 [b,c,h,w]==>[b,c,1,1]
        avg_pool = self.avg_pool(inputs)

        # 调整池化结果的维度 [b,c,1,1]==>[b,c]
        max_pool = max_pool.view([b,c])
        avg_pool = avg_pool.view([b,c])

        # 第一个全连接层下降通道数 [b,c]==>[b,c//4]
        x_maxpool = self.fc1(max_pool)
        x_avgpool = self.fc1(avg_pool)

        # 激活函数
        x_maxpool = self.relu(x_maxpool)
        x_avgpool = self.relu(x_avgpool)
        
        # 第二个全连接层恢复通道数 [b,c//4]==>[b,c]
        x_maxpool = self.fc2(x_maxpool)
        x_avgpool = self.fc2(x_avgpool)
        
        # 将这两种池化结果相加 [b,c]==>[b,c]
        x = x_maxpool + x_avgpool
        # sigmoid函数权值归一化
        x = self.sigmoid(x)
        # 调整维度 [b,c]==>[b,c,1,1]
        x = x.view([b,c,1,1])
        # 输入特征图和通道权重相乘 [b,c,h,w]
        outputs = inputs * x
        
        return outputs

# Convolutional Block Attention Module
class cbams(nn.Module):
    def __init__(self, ch_in,ratio=4, kernel_size=3,stride =1,padding =None,act = 'silu'):
        super().__init__()
        self.sp_att = spatial_attention(
            kernel_size=kernel_size,stride =stride,padding =padding,act = act
            )
        self.chl_att = channel_attention(ch_in = ch_in,ratio = ratio)
    
    def forward(self, x):
        x = self.chl_att(x)
        x = self.sp_att(x)
        
        return x
    
# Convolutional Block Attention Module 3

@register
class cbam(nn.Module):
    def __init__(self):
        super().__init__()
        self.sp_att = nn.Sequential()
        for _ in range(3):
            self.sp_att.append(cbams(ch_in=256))
    def forward(self, b_x,sub_x):
        # for i in range(3):
        #     logtracker.debug(f"sub_x[{i}] = {self.sp_att[i](sub_x[i]).shape}")
        #     logtracker.debug(f"b_x[{i}] = {b_x[i].shape}")     
        for i in range(3):
            b_x[i] = self.sp_att[i](sub_x[i]) * b_x[i]
        return b_x
    