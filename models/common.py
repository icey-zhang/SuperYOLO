# YOLOv5 common modules

import math
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.cuda import amp

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, xyxy2xywh
from utils.plots import color_list, plot_one_box
from utils.torch_utils import time_synchronized
from utils.activations import Mish
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
# from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
# from models.ir_1w32a import IRConv2d
# from models.ir_1w32a import IRConv2d_test
# from models.sr_decoder import Decoder
# from models.ir_1w1a import IRConv2d as IRConv2d_1w1a
# from models.ir_1w1a import IRConv2d_test as IRConv2d_1w1a_test
# from models.quantization import Qout_Activation_Quantize
# from utils.datasets import get_edge
class SiLU(torch.nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)#.cuda()
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity()) #nn.ReLU() #nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = nn.ReLU() #if act is True else (act if isinstance(act, nn.Module) else nn.Identity()) #nn.ReLU() #nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        y = self.act(self.bn(self.conv(x)))
        return y


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))

class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))

class BottleneckCSP2(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP2, self).__init__()
        c_ = int(c2)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_) 
        self.act = nn.LeakyReLU(0.1, inplace=True) #Mish()
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(self.act(self.bn(torch.cat((y1, y2), dim=1))))

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])
        

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class SPPCSP(nn.Module):
    # CSP SPP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSP, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.bn = nn.BatchNorm2d(2 * c_) 
        self.act = Mish()
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(self.act(self.bn(torch.cat((y1, y2), dim=1))))

########zjq############
class RB(nn.Module): #resnetblock
    def __init__(self, channels):
        super(RB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.SiLU(),
            # nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
        )
    def forward(self, x):
        out = self.body(x)
        return out + x

# import numpy as np
from skimage import morphology
def morphologic_process(mask):
    device = mask.device
    b,_,_,_ = mask.shape
    mask = ~mask
    mask_np = mask.cpu().numpy().astype(bool)
    mask_np = morphology.remove_small_objects(mask_np, 20, 2)
    mask_np = morphology.remove_small_holes(mask_np, 10, 2)
    for idx in range(b):
        buffer = np.pad(mask_np[idx,0,:,:],((3,3),(3,3)),'constant')
        buffer = morphology.binary_closing(buffer, morphology.disk(3))
        mask_np[idx,0,:,:] = buffer[3:-3,3:-3]
    mask_np = 1-mask_np
    mask_np = mask_np.astype(float)
    return torch.from_numpy(mask_np).float().to(device)

class SAM(nn.Module):# stereo attention block
    def __init__(self, channels):
        super(SAM, self).__init__()
        self.b1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.rb = RB(channels)
        self.softmax = nn.Softmax(-1)
        self.bottleneck = nn.Conv2d(channels * 2+1, channels, 1, 1, 0, bias=True)
        # self.bottleneck = nn.Conv2d(channels * 2, channels, 1, 1, 0, bias=True)

    def forward(self, x):# B * C * H * W #x_left, x_right
        x_left, x_right = x[0],x[1]
        b, c, h, w = x_left.shape
        buffer_left = self.rb(x_left)
        buffer_right = self.rb(x_right)
        ### M_{right_to_left}
        Q = self.b1(buffer_left).permute(0, 2, 3, 1)  # B * H * W * C
        S = self.b2(buffer_right).permute(0, 2, 1, 3)  # B * H * C * W
        score = torch.bmm(Q.contiguous().view(-1, w, c),
                          S.contiguous().view(-1, c, w))  # (B*H) * W * W
        M_right_to_left = self.softmax(score)

        score_T = score.permute(0,2,1)
        M_left_to_right = self.softmax(score_T)

        #valid mask
        V_left_to_right = torch.sum(M_left_to_right.detach(), 1) > 0.1
        V_left_to_right = V_left_to_right.view(b, 1, h, w)  # B * 1 * H * W
        V_left_to_right = morphologic_process(V_left_to_right)
        V_right_to_left = torch.sum(M_right_to_left.detach(), 1) > 0.1
        V_right_to_left = V_right_to_left.view(b, 1, h, w)  # B * 1 * H * W
        V_right_to_left = morphologic_process(V_right_to_left)

        buffer_R = x_right.permute(0, 2, 3, 1).contiguous().view(-1, w, c)  # (B*H) * W * C
        buffer_l = torch.bmm(M_right_to_left, buffer_R).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)  # B * C * H * W

        buffer_L = x_left.permute(0, 2, 3, 1).contiguous().view(-1, w, c)  # (B*H) * W * C
        buffer_r = torch.bmm(M_left_to_right, buffer_L).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)  # B * C * H * W

        out_L = self.bottleneck(torch.cat((buffer_l, x_left, V_left_to_right), 1))
        out_R = self.bottleneck(torch.cat((buffer_r, x_right, V_right_to_left), 1))
        # out_L = self.bottleneck(torch.cat((buffer_l, x_left), 1))
        # out_R = self.bottleneck(torch.cat((buffer_r, x_right), 1))

        # return out_L, out_R,\
        #        (M_right_to_left.contiguous().view(b, h, w, w),M_left_to_right.contiguous().view(b, h, w, w)),\
        #         (V_right_to_left, V_left_to_right)
        return torch.cat([out_L,out_R],1)


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # squeeze操作
        y = self.fc(y).view(b, c, 1, 1) # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x) # 注意力作用每一个通道上


class MF(nn.Module):# stereo attention block
    def __init__(self, channels):
        super(MF, self).__init__()
        # self.b1 = Conv(channels, channels, 3, 1, 1)
        # self.b2 = Conv(channels, channels, 3, 1, 1)
        # self.b = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        # self.b_ = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.mask_map_r = nn.Conv2d(channels, 1, 1, 1, 0, bias=True)
        self.mask_map_i = nn.Conv2d(1, 1, 1, 1, 0, bias=True)
        # self.rb = RB(channels)
        self.softmax = nn.Softmax(-1)
        self.bottleneck1 = nn.Conv2d(1, 16, 3, 1, 1, bias=False)
        self.bottleneck2 = nn.Conv2d(channels, 48, 3, 1, 1, bias=False)
        # self.bottleneck1 = nn.Conv2d(channels, 8, 3, 1, 1, bias=False)
        # self.bottleneck2 = nn.Conv2d(channels, 24, 3, 1, 1, bias=False)
        # self.bottleneck1 = nn.Conv2d(channels, 16, 3, 1, 1, bias=True)
        # self.bottleneck2 = nn.Conv2d(channels, 48, 3, 1, 1, bias=True)
        self.se = SE_Block(64)
        self.se_r = SE_Block(3)
        self.se_i = SE_Block(1)
        # self.se_i_r = SE_Block(4)
        # self.bottleneck1 = Conv(channels, 16, 3, 1, 1)
        # self.bottleneck2 = Conv(channels, 48, 3, 1, 1)

    def forward(self, x):# B * C * H * W #x_left, x_right
        x_left_ori, x_right_ori = x[0],x[1]
        b, c, h, w = x_left_ori.shape
        ### M_{right_to_left}
        x_left = self.se_r(x_left_ori)
        x_right = self.se_i(x_right_ori)
        # x = self.se_i_r(torch.cat([x_left,x_right],1))
        # x_left = x[:,0:3,:,:,]
        # x_right = x[:,3:,:,:,]

        x_mask_left = torch.mul(self.mask_map_r(x_left).repeat(1,3,1,1),x_left)
        x_mask_right = torch.mul(self.mask_map_i(x_right),x_right)
        # x_mask_left = torch.mul(self.mask_map_i(x_right).repeat(1,3,1,1),x_left)+x_mask_left
        # x_mask_right = torch.mul(self.mask_map_r(x_left),x_right)+x_mask_right
        # Q = self.b(x_left).permute(0, 2, 3, 1)  # B * H * W * C
        # S = self.b_(x_right).permute(0, 2, 1, 3)  # B * H * C * W

        # score = torch.bmm(Q.contiguous().view(-1, w, c),
        #                   S.contiguous().view(-1, c, w))  # (B*H) * W * W
        # M_right_to_left = self.softmax(score)

        # score_T = score.permute(0,2,1)
        # M_left_to_right = self.softmax(score_T)

        # buffer_R = x_right.permute(0, 2, 3, 1).contiguous().view(-1, w, c)  # (B*H) * W * C
        # buffer_l = torch.bmm(M_right_to_left, buffer_R).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)  # B * C * H * W #IR

        # buffer_L = x_left.permute(0, 2, 3, 1).contiguous().view(-1, w, c)  # (B*H) * W * C
        # buffer_r = torch.bmm(M_left_to_right, buffer_L).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)  # B * C * H * W #RGB 

        
        # out = torch.cat([buffer_r+x_left,buffer_l+x_right],1)
        # out_IR = self.bottleneck1(buffer_l+x_right)
        # out_RGB = self.bottleneck2(buffer_r+x_left) #RGB
        out_IR = self.bottleneck1(x_mask_right+x_right_ori)
        out_RGB = self.bottleneck2(x_mask_left+x_left_ori) #RGB
        # out_IR = self.bottleneck1(x_mask_right+x_right_ori)
        # out_RGB = self.bottleneck2(x_mask_left+x_left_ori) #RGB
        out = self.se(torch.cat([out_RGB,out_IR],1))
        # import scipy.io as sio
        # sio.savemat('features/output.mat', mdict={'data':out.cpu().numpy()})

        return out

class ScaledDotProductAttentionOnly(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, v, k, q, mask=None):
        b, c, h, w      = q.size(0), q.size(1), q.size(2), q.size(3)

        # Reshaping K,Q, and Vs...
        q = q.view(b, c, h*w)
        k = k.view(b, c, h*w)
        v = v.view(b, c, h*w)


        # Compute attention
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        # Normalization (SoftMax)
        attn    = F.softmax(attn, dim=-1)

        # Attention output
        output  = torch.matmul(attn, v)

        #Reshape output to original format
        output  = output.view(b, c, h, w)
        return output

# class MF(nn.Module):# stereo attention block
#     def __init__(self, channels):
#         super(MF, self).__init__()
#         self.b1 = nn.Conv2d(channels, channels, 1, 1, 0,bias=True)
#         self.b2 = nn.Conv2d(channels, channels, 1, 1, 0,bias=True)
#         self.attention1 = ScaledDotProductAttentionOnly(1)
#         self.attention2 = ScaledDotProductAttentionOnly(1)
#         self.bottleneck1 = nn.Conv2d(channels, channels, 1, 1, 0,bias=True)
#         self.bottleneck2 = nn.Conv2d(channels, channels, 1, 1, 0,bias=True)
#         self.se= SE_Block(6)


#     def forward(self, x):# B * C * H * W #x_left, x_right
#         x_left, x_right = x[0],x[1]
#         rgb = self.b1(x_left)
#         ir = self.b2(x_right)
#         feature1 = self.attention1(rgb,rgb,ir)
#         feature2 = self.attention2(ir,ir,rgb)

#         out_RGB = self.bottleneck1(feature1)+x_left
#         out_IR = self.bottleneck2(feature2)+x_right #RGB
#         out = self.se(torch.cat([out_RGB,out_IR],1))
#         return out

# class MF(nn.Module):# stereo attention block
#     def __init__(self, channels):
#         super(MF, self).__init__()
#         # self.b1 = Conv(channels, channels, 3, 1, 1)
#         # self.b2 = Conv(channels, channels, 3, 1, 1)
#         self.b = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
#         self.b_ = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
#         # self.rb = RB(channels)
#         self.softmax = nn.Softmax(-1)
#         self.bottleneck1 = nn.Conv2d(channels, 16, 3, 1, 1, bias=False)
#         self.bottleneck2 = nn.Conv2d(channels, 48, 3, 1, 1, bias=False)
#         # self.bottleneck1 = nn.Conv2d(channels, 8, 3, 1, 1, bias=False)
#         # self.bottleneck2 = nn.Conv2d(channels, 24, 3, 1, 1, bias=False)
#         # self.bottleneck1 = nn.Conv2d(channels, 16, 3, 1, 1, bias=True)
#         # self.bottleneck2 = nn.Conv2d(channels, 48, 3, 1, 1, bias=True)
#         self.se = SE_Block(64)

#         # self.bottleneck1 = Conv(channels, 16, 3, 1, 1)
#         # self.bottleneck2 = Conv(channels, 48, 3, 1, 1)

#     def forward(self, x):# B * C * H * W #x_left, x_right
#         x_left, x_right = x[0],x[1]
#         b, c, h, w = x_left.shape
#         ### M_{right_to_left}
#         Q = self.b(x_left).permute(0, 2, 3, 1)  # B * H * W * C
#         S = self.b_(x_right).permute(0, 1, 2, 3)  # B * C * H * W
#         score = torch.bmm(Q.contiguous().view(-1, h*w, c),
#                           S.contiguous().view(-1, c, h*w))  # B*(H * W) * (H*W)
#         M_right_to_left = self.softmax(score)

#         score_T = score.permute(0,2,1)
#         M_left_to_right = self.softmax(score_T)

#         buffer_R = x_right.permute(0, 2, 3, 1).contiguous().view(-1, h*w, c)  # (B*H) * W * C
#         buffer_l = torch.bmm(M_right_to_left, buffer_R).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)  # B * C * H * W #IR

#         buffer_L = x_left.permute(0, 2, 3, 1).contiguous().view(-1, h*w, c)  # (B*H) * W * C
#         buffer_r = torch.bmm(M_left_to_right, buffer_L).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)  # B * C * H * W #RGB 

        
#         # out = torch.cat([buffer_r+x_left,buffer_l+x_right],1)
#         out_IR = self.bottleneck1(buffer_l+x_right)
#         out_RGB = self.bottleneck2(buffer_r+x_left) #RGB
#         out = self.se(torch.cat([out_RGB,out_IR],1))
#         return out

class SSTN(nn.Module):
    def __init__(self, channels):
        super(SSTN, self).__init__()
        self.b1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.b2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.b3 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.b4 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.b5 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)

        

    def forward(self, x):# B * C * H * W #x_left, x_right
        img, ir = x[0],x[1]
        img_hr = get_edge(img)
        ir_hr = get_edge(ir)
        img_lr = img - img_hr
        ir_lr = ir - ir_hr
        buffer_img = self.b1(img)
        buffer_ir = self.b2(ir)
        feature_specific_img = self.b3(img_lr)
        feature_specific_ir = self.b4(ir_lr)

        feature_shared_img = self.b5(img_hr)
        feature_shared_ir = self.b5(ir_hr)
 
        return torch.cat([buffer_img,buffer_ir,feature_specific_img,feature_specific_ir,feature_shared_img,feature_shared_ir],1)

class SSTN3(nn.Module):
    def __init__(self, channels):
        super(SSTN3, self).__init__()
        self.b1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b3 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b4 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b5 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)

    def forward(self, x):# B * C * H * W #x_left, x_right
        img, ir = x[0],x[1]
        img_hr = get_edge(img)
        ir_hr = get_edge(ir)
        img_lr = img - img_hr
        ir_lr = ir - ir_hr
        buffer_img = self.b1(img)
        buffer_ir = self.b2(ir)
        feature_specific_img = self.b3(img_hr)
        feature_specific_ir = self.b4(ir_hr)

        feature_shared_img = self.b5(img_lr)
        feature_shared_ir = self.b5(ir_lr)
 
        return torch.cat([buffer_img,buffer_ir,feature_specific_img,feature_specific_ir,feature_shared_img,feature_shared_ir],1)

class SSFF1(nn.Module):
    def __init__(self, channels):
        super(SSFF1, self).__init__()
        self.b1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b3 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b4 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b5 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b6 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b7 = nn.Conv2d(channels*3, channels, 1, 1, 0, bias=True)

    def forward(self, x):# B * C * H * W #x_left, x_right
        img, ir = x[0],x[1]
        img_hr = get_edge(img)
        ir_hr = get_edge(ir)
        img_lr = img - img_hr
        ir_lr = ir - ir_hr
        buffer_img_specific = self.b1(img)
        buffer_ir_specific = self.b2(ir)
        feature_specific_img_hr = self.b3(img_hr)
        feature_specific_ir_hr = self.b4(ir_hr)

        feature_specific_img_lr = self.b5(img_lr)
        feature_specific_ir_lr = self.b6(ir_lr)

        buffer_img_shared = self.b7(torch.cat([buffer_img_specific,feature_specific_img_hr,feature_specific_img_lr],1))
        buffer_ir_shared = self.b7(torch.cat([buffer_ir_specific,feature_specific_ir_hr,feature_specific_ir_lr],1))
 
        return torch.cat([buffer_img_specific,buffer_ir_specific,buffer_img_shared,buffer_ir_shared,feature_specific_img_hr,feature_specific_ir_hr,feature_specific_img_lr,feature_specific_ir_lr],1)

class SSFF2(nn.Module):
    def __init__(self, channels):
        super(SSFF2, self).__init__()
        self.b1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b3 = nn.Conv2d(channels*3, channels, 1, 1, 0, bias=True)
        self.b4 = nn.Conv2d(channels*3, channels, 1, 1, 0, bias=True)
        self.b5 = nn.Conv2d(channels*3, channels, 1, 1, 0, bias=True)

    def forward(self, x):# B * C * H * W #x_left, x_right
        img, ir = x[0],x[1]
        img_hr = get_edge(img)
        ir_hr = get_edge(ir)
        img_lr = img - img_hr
        ir_lr = ir - ir_hr
        buffer_img_specific = self.b1(img)
        buffer_ir_specific = self.b2(ir)

        feature_specific_img = self.b3(torch.cat([img_hr,img_lr,buffer_img_specific],1))
        feature_specific_ir = self.b4(torch.cat([ir_hr,ir_lr,buffer_ir_specific],1))

        buffer_img_shared = self.b5(torch.cat([img_hr,img_lr,buffer_img_specific],1))
        buffer_ir_shared = self.b5(torch.cat([ir_hr,ir_lr,buffer_ir_specific],1))
 
        return torch.cat([feature_specific_img,feature_specific_ir,buffer_img_shared,buffer_ir_shared],1)

class SSFF(nn.Module):
    def __init__(self, channels):
        super(SSFF, self).__init__()
        self.b1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b3 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b4 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b5 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b6 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b7 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)

    def forward(self, x):# B * C * H * W #x_left, x_right
        img, ir = x[0],x[1]
        img_hr = get_edge(img)
        ir_hr = get_edge(ir)
        img_lr = img - img_hr
        ir_lr = ir - ir_hr
        buffer_img_specific = self.b1(img)
        buffer_ir_specific = self.b2(ir)
        feature_specific_img_hr = self.b3(img_hr)
        feature_specific_ir_hr = self.b4(ir_hr)

        feature_specific_img_lr = self.b5(img_lr)
        feature_specific_ir_lr = self.b6(ir_lr)

        buffer_img_shared = self.b7(img)
        buffer_ir_shared = self.b7(ir)
 
        return torch.cat([buffer_img_specific,buffer_ir_specific,buffer_img_shared,buffer_ir_shared,feature_specific_img_hr,feature_specific_ir_hr,feature_specific_img_lr,feature_specific_ir_lr],1)

class SSTN1(nn.Module):
    def __init__(self, channels):
        super(SSTN1, self).__init__()
        self.b1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.b2 = nn.Conv2d(1, 1, 3, 1, 1, bias=True)
        self.b3 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.b4 = nn.Conv2d(1, 1, 3, 1, 1, bias=True)
        self.b5 = nn.Conv2d(channels, 8, 3, 1, 1, bias=True)
        self.bottleneck1 = nn.Conv2d(channels, 32, 3, 1, 1, bias=True)
        self.bottleneck2 = nn.Conv2d(1, 16, 3, 1, 1, bias=True)
        # self.bottleneck3 = nn.Conv2d(channels, 16, 3, 1, 1, bias=True)

    def forward(self, x):# B * C * H * W #x_left, x_right
        img, ir = x[0],x[1]
        # img_hr = get_edge(img)
        # ir_hr = get_edge(ir)
        buffer_img = self.b1(img)
        buffer_ir = self.b2(ir)
        feature_specific_img = self.b3(buffer_img)
        feature_specific_ir = self.b4(buffer_ir)
        x_mask_img = torch.mul(feature_specific_img,buffer_img)
        x_mask_ir = torch.mul(feature_specific_ir,buffer_ir)
        out_RGB = self.bottleneck1(x_mask_img+buffer_img)
        out_IR = self.bottleneck2(x_mask_ir+buffer_ir)

        feature_shared_img = self.b5(buffer_img)
        feature_shared_ir = self.b5(buffer_ir.repeat(1,3,1,1))
        # self.bottleneck3
 
        return torch.cat([out_RGB,out_IR,feature_shared_img,feature_shared_ir],1)

class SSTN12(nn.Module):
    def __init__(self, channels):
        super(SSTN12, self).__init__()
        self.b1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b3 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b4 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b5 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)

    def forward(self, x):# B * C * H * W #x_left, x_right
        img, ir = x[0],x[1]
        img_hr = get_edge(img)
        ir_hr = get_edge(ir)
        buffer_img = self.b1(img)
        buffer_ir = self.b2(ir)
        feature_specific_img = self.b3(buffer_img)
        feature_specific_ir = self.b4(buffer_ir)

        feature_shared_img = self.b5(img_hr)
        feature_shared_ir = self.b5(ir_hr)
 
        return torch.cat([feature_specific_img,feature_specific_ir,feature_shared_img,feature_shared_ir],1)

class SSTN11(nn.Module):
    def __init__(self, channels):
        super(SSTN11, self).__init__()
        self.b1 = Conv(channels, channels, 1, 1, 0)
        self.b2 = Conv(channels, channels, 1, 1, 0)
        self.b3 = Conv(channels, channels, 1, 1, 0)
        self.b4 = Conv(channels, channels, 1, 1, 0)
        self.b5 = Conv(channels, channels, 1, 1, 0)

    def forward(self, x):# B * C * H * W #x_left, x_right
        img, ir = x[0],x[1]
        # img_hr = get_edge(img)
        # ir_hr = get_edge(ir)
        buffer_img = self.b1(img)
        buffer_ir = self.b2(ir)
        feature_specific_img = self.b3(buffer_img)
        feature_specific_ir = self.b4(buffer_ir)

        feature_shared_img = self.b5(buffer_img)
        feature_shared_ir = self.b5(buffer_ir)
 
        return torch.cat([feature_specific_img,feature_specific_ir,feature_shared_img,feature_shared_ir],1)

class SSTN2(nn.Module):
    def __init__(self, channels):
        super(SSTN2, self).__init__()
        self.b1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b3 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        # self.b4 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        # self.b5 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)

    def forward(self, x):# B * C * H * W #x_left, x_right
        img, ir = x[0],x[1]
        img_hr = get_edge(img)
        ir_hr = get_edge(ir)
        img_lr = img - img_hr
        ir_lr = ir - ir_hr
        feature_specific_img = self.b1(img_lr)
        feature_specific_ir = self.b2(ir_lr)

        feature_shared_img = self.b3(img_hr)
        feature_shared_ir = self.b3(ir_hr)
 
        return torch.cat([feature_specific_img,feature_specific_ir,feature_shared_img,feature_shared_ir],1)

class SSTN4(nn.Module):
    def __init__(self, channels):
        super(SSTN4, self).__init__()
        self.b1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b3 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        # self.b4 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        # self.b5 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)

    def forward(self, x):# B * C * H * W #x_left, x_right
        img, ir = x[0],x[1]
        img_hr = get_edge(img)
        ir_hr = get_edge(ir)
        img_lr = img - img_hr
        ir_lr = ir - ir_hr
        feature_specific_img = self.b1(img_hr)
        feature_specific_ir = self.b2(ir_hr)

        feature_shared_img = self.b3(img_lr)
        feature_shared_ir = self.b3(ir_lr)
 
        return torch.cat([feature_specific_img,feature_specific_ir,feature_shared_img,feature_shared_ir],1)

class SSTN5(nn.Module):
    def __init__(self, channels):
        super(SSTN5, self).__init__()
        self.b1 = Conv(channels, channels, 1, 1, 0)
        self.b2 = Conv(channels, channels, 1, 1, 0)
        self.b3 = Conv(channels, channels, 1, 1, 0)
        # self.b4 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        # self.b5 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)

    def forward(self, x):# B * C * H * W #x_left, x_right
        img, ir = x[0],x[1]
        img_hr = get_edge(img)
        ir_hr = get_edge(ir)
        img_lr = img - img_hr
        ir_lr = ir - ir_hr
        feature_specific_img = self.b1(img_lr)
        feature_specific_ir = self.b2(ir_lr)

        feature_shared_img = self.b3(img_hr)
        feature_shared_ir = self.b3(ir_hr)
 
        return torch.cat([feature_specific_img,feature_specific_ir,feature_shared_img,feature_shared_ir],1)

class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1/16.)

            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, padding=0, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)

            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)

class SSWT(nn.Module):
    def __init__(self, channels):
        super(SSWT, self).__init__()
        self.b1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b3 = nn.Conv2d(channels*3, channels*3, 1, 1, 0, bias=True)
        self.HaarDownsampling = HaarDownsampling(channels)
        # self.b4 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        # self.b5 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)

    def forward(self, x):# B * C * H * W #x_left, x_right
        img, ir = x[0],x[1]
        img_fre = self.HaarDownsampling(img)
        ir_fre = self.HaarDownsampling(ir)
        img_hr = img_fre[:,3:,:,:]
        ir_hr = ir_fre[:,3:,:,:]
        img_lr = img_fre[:,:3,:,:]
        ir_lr = ir_fre[:,:3,:,:]
        feature_specific_img = self.b1(img_lr)
        feature_specific_ir = self.b2(ir_lr)

        feature_shared_img = self.b3(img_hr)
        feature_shared_ir = self.b3(ir_hr)
 
        return torch.cat([feature_specific_img,feature_specific_ir,feature_shared_img,feature_shared_ir],1)

class WT(nn.Module):
    def __init__(self, channels):
        super(WT, self).__init__()
        self.b1 = nn.Conv2d(3, 3, 1, 1, 0, bias=True)
        self.b2 = nn.Conv2d(1, 1, 1, 1, 0, bias=True)
        self.b3 = nn.Conv2d(12, 12, 1, 1, 0, bias=True)
        self.HaarDownsampling = HaarDownsampling(4)
        # self.b4 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        # self.b5 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)

    def forward(self, x):# B * C * H * W #x_left, x_right
        # img, ir = x[0],x[1]
        img_fre = self.HaarDownsampling(x)
        img_hr = img_fre[:,4:,:,:]
        img_lr = img_fre[:,:3,:,:]
        ir_lr = img_fre[:,3:4,:,:]
        feature_specific_img = self.b1(img_lr)
        feature_specific_ir = self.b2(ir_lr)
        feature_shared = self.b3(img_hr)
 
        return torch.cat([feature_specific_img,feature_specific_ir,feature_shared],1)

class WT1(nn.Module):
    def __init__(self, channels):
        super(WT1, self).__init__()
        self.b1 = nn.Conv2d(3, 3, 1, 1, 0, bias=True)
        self.b2 = nn.Conv2d(1, 1, 1, 1, 0, bias=True)
        self.b3 = nn.Conv2d(3, 3, 1, 1, 0, bias=True)
        self.b4 = nn.Conv2d(1, 1, 1, 1, 0, bias=True)
        self.b5 = nn.Conv2d(12, 12, 1, 1, 0, bias=True)
        self.HaarDownsampling = HaarDownsampling(4)
        # self.b4 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        # self.b5 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)

    def forward(self, x):# B * C * H * W #x_left, x_right
        img, ir = x[:,0:3,:,:],x[:,3:,:,:]
        buffer_img = self.b1(img)
        buffer_ir = self.b2(ir)
        img_fre = self.HaarDownsampling(torch.cat([buffer_img,buffer_ir],1))
        img_hr = img_fre[:,4:,:,:]
        img_lr = img_fre[:,:3,:,:]
        ir_lr = img_fre[:,3:4,:,:]
        feature_specific_img = self.b3(img_lr)
        feature_specific_ir = self.b4(ir_lr)
        feature_shared = self.b5(img_hr)
 
        return torch.cat([feature_specific_img,feature_specific_ir,feature_shared],1)

class SSWT1(nn.Module):
    def __init__(self, channels):
        super(SSWT1, self).__init__()
        self.b1 = nn.Conv2d(channels, channels, 1, 2, 0, bias=True)
        self.b2 = nn.Conv2d(channels, channels, 1, 2, 0, bias=True)
        self.b3 = nn.Conv2d(channels*3, channels*3, 1, 1, 0, bias=True)
        self.HaarDownsampling = HaarDownsampling(channels)

    def forward(self, x):# B * C * H * W #x_left, x_right
        img, ir = x[0],x[1]
        img_fre = self.HaarDownsampling(img)
        ir_fre = self.HaarDownsampling(ir)
        img_hr = img_fre[:,3:,:,:]
        ir_hr = ir_fre[:,3:,:,:]
        feature_specific_img = self.b1(img)
        feature_specific_ir = self.b2(ir)

        feature_shared_img = self.b3(img_hr)
        feature_shared_ir = self.b3(ir_hr)
 
        return torch.cat([feature_specific_img,feature_specific_ir,feature_shared_img,feature_shared_ir],1)


class SSWT2(nn.Module):
    def __init__(self, channels):
        super(SSWT2, self).__init__()
        self.b1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b3 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.HaarDownsampling = HaarDownsampling(channels)
        self.b4 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b5 = nn.Conv2d(channels*3, channels*3, 1, 1, 0, bias=True)

    def forward(self, x):# B * C * H * W #x_left, x_right
        img, ir = x[0],x[1]
        buffer_img = self.b1(img)
        buffer_ir = self.b2(ir)
        img_fre = self.HaarDownsampling(buffer_img)
        ir_fre = self.HaarDownsampling(buffer_ir)
        img_hr = img_fre[:,3:,:,:]
        ir_hr = ir_fre[:,3:,:,:]
        img_lr = img_fre[:,:3,:,:]
        ir_lr = ir_fre[:,:3,:,:]
        feature_specific_img = self.b3(img_lr)
        feature_specific_ir = self.b4(ir_lr)

        feature_shared_img = self.b5(img_hr)
        feature_shared_ir = self.b5(ir_hr)
 
        return torch.cat([feature_specific_img,feature_specific_ir,feature_shared_img,feature_shared_ir],1)


class SSWT3(nn.Module):
    def __init__(self, channels):
        super(SSWT3, self).__init__()
        self.b1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b3 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.HaarDownsampling = HaarDownsampling(channels)
        self.b4 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b5 = nn.Conv2d(channels*3, channels*3, 1, 1, 0, bias=True)

    def forward(self, x):# B * C * H * W #x_left, x_right
        img, ir = x[0],x[1]
        img_fre = self.HaarDownsampling(img)
        ir_fre = self.HaarDownsampling(ir)
        img_hr = img_fre[:,3:,:,:]
        ir_hr = ir_fre[:,3:,:,:]
        img_lr = img_fre[:,:3,:,:]
        ir_lr = ir_fre[:,:3,:,:]
        buffer_img = self.b1(img_lr)
        buffer_ir = self.b2(ir_lr)
        feature_specific_img = self.b3(buffer_img)
        feature_specific_ir = self.b4(buffer_ir)

        feature_shared_img = self.b5(img_hr)
        feature_shared_ir = self.b5(ir_hr)
 
        return torch.cat([feature_specific_img,feature_specific_ir,feature_shared_img,feature_shared_ir],1)

class SSWT4(nn.Module):
    def __init__(self, channels):
        super(SSWT4, self).__init__()
        self.b1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b3 = nn.Conv2d(channels, channels, 1, 2, 0, bias=True)
        self.HaarDownsampling = HaarDownsampling(channels)
        self.b4 = nn.Conv2d(channels, channels, 1, 2, 0, bias=True)
        self.b5 = nn.Conv2d(channels*3, channels*3, 1, 1, 0, bias=True)

    def forward(self, x):# B * C * H * W #x_left, x_right
        img, ir = x[0],x[1]
        buffer_img = self.b1(img)
        buffer_ir = self.b2(ir)
        img_fre = self.HaarDownsampling(buffer_img)
        ir_fre = self.HaarDownsampling(buffer_ir)
        img_hr = img_fre[:,3:,:,:]
        ir_hr = ir_fre[:,3:,:,:]
        feature_specific_img = self.b3(buffer_img)
        feature_specific_ir = self.b4(buffer_ir)

        feature_shared_img = self.b5(img_hr)
        feature_shared_ir = self.b5(ir_hr)
 
        return torch.cat([feature_specific_img,feature_specific_ir,feature_shared_img,feature_shared_ir],1)

############################
class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=720, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/samples/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(720,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(720,1280,3)
        #   numpy:           = np.zeros((720,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,720,1280)  # BCHW
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_synchronized()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            if isinstance(im, str):  # filename or uri
                im, f = Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im), im  # open
                im.filename = f  # for uri
            files.append(Path(im.filename).with_suffix('.jpg').name if isinstance(im, Image.Image) else f'image{i}.jpg')
            im = np.array(im)  # to numpy
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_synchronized())

        with torch.no_grad(), amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_synchronized())

            # Post-process
            y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

        t.append(time_synchronized())
        return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, render=False, save_dir=''):
        colors = color_list()
        for i, (img, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {img.shape[0]}x{img.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render:
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(box, img, label=label, color=colors[int(cls) % 10])
            img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
            if pprint:
                print(str.rstrip(', '))
            if show:
                img.show(self.files[i])  # show
            if save:
                f = Path(save_dir) / self.files[i]
                img.save(f)  # save
                print(f"{'Saving' * (i == 0)} {f},", end='' if i < self.n - 1 else ' done.\n')
            if render:
                self.imgs[i] = np.asarray(img)

    def print(self):
        self.display(pprint=True)  # print results
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='results/'):
        Path(save_dir).mkdir(exist_ok=True)
        self.display(save=True, save_dir=save_dir)  # save results

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def __len__(self):
        return self.n

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)

###############

class AttentionModel(nn.Module):
    def __init__(self,c1, k=3, s=1):
        super(AttentionModel, self).__init__()
        self.conv = nn.Conv2d(c1, 1, k, s, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self,x):
        x1 = self.conv(x)
        attention_map = self.output_act(x1)
        output = x + x * torch.exp(attention_map)
        return attention_map,output


def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda(0).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda(0).unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]

def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)

def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


class ACmix(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(ACmix, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)

        self.fc = nn.Conv2d(3*self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes, kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1, stride=stride)

        self.reset_parameters()
    
    def reset_parameters(self):
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i//self.kernel_conv, i%self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x) #[2, 16, 128, 128] 
        scaling = float(self.head_dim) ** -0.5
        b, c, h, w = q.shape
        h_out, w_out = h//self.stride, w//self.stride


# ### att
        # ## positional encoding
        pe = self.conv_p(position(h, w, x.is_cuda))

        q_att = q.view(b*self.head, self.head_dim, h, w) * scaling #[4, 8, 128, 128]
        k_att = k.view(b*self.head, self.head_dim, h, w) #[4, 8, 128, 128]
        v_att = v.view(b*self.head, self.head_dim, h, w) #[4, 8, 128, 128]

        if self.stride > 1:
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe


        unfold_k = self.unfold(self.pad_att(k_att)).view(b*self.head, self.head_dim, self.kernel_att*self.kernel_att, h_out, w_out) # b*head, head_dim, k_att^2, h_out, w_out #[4, 8, 49, 128, 128]
        unfold_rpe = self.unfold(self.pad_att(pe)).view(1, self.head_dim, self.kernel_att*self.kernel_att, h_out, w_out) # 1, head_dim, k_att^2, h_out, w_out #[1, 8, 49, 128, 128]
        
        att = (q_att.unsqueeze(2)*(unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(1) # (b*head, head_dim, 1, h_out, w_out) * (b*head, head_dim, k_att^2, h_out, w_out) -> (b*head, k_att^2, h_out, w_out)
        att = self.softmax(att) #[4, 49, 128, 128]

        out_att = self.unfold(self.pad_att(v_att)).view(b*self.head, self.head_dim, self.kernel_att*self.kernel_att, h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out) #([2, 16, 128, 128])
        
## conv
        f_all = self.fc(torch.cat([q.view(b, self.head, self.head_dim, h*w), k.view(b, self.head, self.head_dim, h*w), v.view(b, self.head, self.head_dim, h*w)], 1)) #torch.Size([2, 6, 8, 16384]) [2, 9, 8, 16384]
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1]) #[2, 72, 128, 128] (2, -1, 128, 128)
        
        out_conv = self.dep_conv(f_conv)

        return self.rate1 * out_att + self.rate2 * out_conv
