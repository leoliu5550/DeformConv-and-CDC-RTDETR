'''by lyuwenyu
'''

import copy
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision.ops import deform_conv2d

from .utils import get_activation

from src.core import register
import math
import logging
import logging.config
logging.config.fileConfig('logging.conf')
logtracker = logging.getLogger(f"model.HybridEncoder.{__name__}")
__all__ = ['HybridEncoder']

class DeformConvBlock(nn.Module):
    def __init__(self,ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act='relu'):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias
            )
        self.conv_offset = nn.Conv2d(
            ch_in, 
            (kernel_size**2)*2, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias
            )
        # nn.init.constant_(self.conv_offset.weight,0)
        nn.init.trunc_normal_(self.conv_offset.weight)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        offset = self.conv_offset(x)
        # print(f"offset shape = {offset.shape}")
        # mask = self.sig(self.conv_mask(x)) 

        out = deform_conv2d(
            input=x, offset=offset, 
            weight=self.conv.weight, 
            mask=None, padding=(1, 1))

        
        return out



class Conv2d_cdiffBlock(nn.Module):
    def __init__(self,
                ch_in, 
                ch_out, 
                kernel_size, 
                stride, 
                padding=None, 
                bias=False, 
                act='relu',
                theta=0.7):

        super().__init__() 
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        nn.init.normal_(self.conv.weight)
        self.theta = theta
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        out_normal = self.conv(x)
        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            # [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)

            kernel_diff = kernel_diff[:, :, None, None]

            out_diff = F.conv2d(
                input=x, 
                weight=kernel_diff, 
                bias=self.conv.bias, 
                stride=self.conv.stride, 
                padding=0
                )
            out = out_normal - self.theta * out_diff
            out = self.act(out)
            return out


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# Spatail Attention Module
class SPattenBlock(nn.Module):
    def __init__(self,ch_in,ch_out,act = 'relu'):
        super().__init__()
        self.ch_in = ch_in//2
        self.ch_out = ch_out//2
        self.conv1 = ConvNormLayer(self.ch_in, self.ch_out , 3, 1, padding=1, act=act)
        self.subconv2 =  Conv2d_cdiffBlock(self.ch_in, self.ch_out , 3, 1, padding=1, act=act)
        self.subconv3 =  Conv2d_cdiffBlock(self.ch_in, self.ch_out , 3, 1, padding=1, act=act)
        
        self.act = nn.Identity() if act is None else get_activation(act) 
        self.pooling = nn.AvgPool2d(3, 1 ,1)
    def forward(self,x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            upper_x = x[...,:self.ch_in,:,:]
            lower_x = x[...,self.ch_in:,:,:]
            upper_x = self.subconv2(self.conv1(upper_x))
            lower_x = self.subconv3(self.conv1(lower_x))
            x = torch.cat((upper_x,lower_x ),1)
            y = self.pooling(x)
            y = self.act(y)
        return y

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 
        # self.__delattr__('conv1')
        # self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1


    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 
        # self.__delattr__('conv1')
        # self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                num_blocks=3,
                expansion=1.0,
                bias=None,
                act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                d_model,
                nhead,
                dim_feedforward=2048,
                dropout=0.1,
                activation="relu",
                normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation) 

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


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


@register
class HybridEncoder(nn.Module):
    def __init__(self,
                in_channels=[512, 1024, 2048],
                feat_strides=[8, 16, 32],
                hidden_dim=256,
                nhead=8,
                dim_feedforward = 1024,
                dropout=0.0,
                enc_act='gelu',
                use_encoder_idx=[2],
                num_encoder_layers=1,
                pe_temperature=10000,
                expansion=1.0,
                depth_mult=1.0,
                act='silu',
                eval_spatial_size=None):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size

        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        
        # channel projection
        self.input_proj = nn.ModuleList()
        # for layer_idx,in_channel in enumerate(in_channels) :
        #     # add deform convlayer to all
        #     # self.input_proj.append(
        #     #     nn.Sequential(
        #     #         # let deform conv remain same size after deformconv
        #     #         DeformConvBlock(in_channel, hidden_dim, kernel_size=3, stride=1, padding=1,bias=False),
        #     #         nn.BatchNorm2d(hidden_dim)
        #     #     )
        #     # )
        
        #     if layer_idx == len(in_channels)-1:
        #         self.input_proj.append(
        #             nn.Sequential(
        #                 # let deform conv remain same size after deformconv
        #                 Conv2d_cdiffBlock(in_channel, hidden_dim, kernel_size=3, stride=1, padding=1,bias=False),
        #                 nn.BatchNorm2d(hidden_dim)
        #             )
        #         )
        #     else:
        #         self.input_proj.append(
        #             nn.Sequential(
        #                 nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
        #                 nn.BatchNorm2d(hidden_dim)
        #             )
        #         )
                
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim)
                )
            )

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, 
            nhead=nhead,
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation=enc_act)

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        self._reset_parameters()
        # self.yolov1 = Yolov1()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        '''
        '''
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats,ori_x):
        # yolo_feats = self.yolov1(ori_x)
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # encoder
        # for rwo in proj_feats:
        #     logtracker.debug(f" shape is {rwo.shape}")
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # logtracker.debug(f"at {enc_ind} shape is {proj_feats[enc_ind].shape}")
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
                # print([x.is_contiguous() for x in proj_feats ])

        # add yolov1 backbone after encoder layer
        # for i in range(len(yolo_feats)):
        #     proj_feats[i] =  yolo_feats[i]+proj_feats[i]



        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            outs.append(out)
        # for rw in outs:
        #     logtracker.debug(f" shape is {rw.shape}")
        return outs
