import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.sr_decoder_noBN_noD import Decoder
from models.edsr import EDSR

# class AttentionModel(nn.module):
#     def __init__(self,feature_in):
#         self.conv = nn.conv2d(feature_in,1,kernel_size=3, padding=1)
#         self.output_act = nn.Sigmoid()
#     def foward(self,x):
#         x = self.conv(x)
#         attention_map = self.output_act(x)
#         output = x + x * torch.exp(attention_map)
#         return attention_map,output

class EDSRConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EDSRConv, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            )

        self.residual_upsampler = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            )

    def forward(self, input):
        return self.conv(input)+self.residual_upsampler(input)


class DeepLab(nn.Module):
    def __init__(self, ch, c1=128, c2=512,factor=2, sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        #self.attention = AttentionModel(128)
        self.sr_decoder = Decoder(c1,c2)
        self.edsr = EDSR(num_channels=ch,input_channel=64, factor=8)
        # self.up_sr_1 = nn.ConvTranspose2d(64, 64, 2, stride=2) 
        # self.up_edsr_1 = EDSRConv(64,64)
        # self.up_sr_2 = nn.ConvTranspose2d(64, 32, 2, stride=2) 
        # self.up_edsr_2 = EDSRConv(32,32)
        # self.up_sr_3 = nn.ConvTranspose2d(32, 16, 2, stride=2) 
        # self.up_edsr_3 = EDSRConv(16,16)
        # self.up_conv_last = nn.Conv2d(16,ch,1)
        self.factor = factor


        # self.freeze_bn = freeze_bn

    def forward(self, low_level_feat,x):
        x_sr= self.sr_decoder(x, low_level_feat,self.factor)
        x_sr_up = self.edsr(x_sr)
        #attention_map,x_sr = self.attention(x_sr)
        # x_sr_up = self.up_sr_1(x_sr)
        # x_sr_up=self.up_edsr_1(x_sr_up)

        # x_sr_up = self.up_sr_2(x_sr_up)
        # x_sr_up=self.up_edsr_2(x_sr_up)

        # x_sr_up = self.up_sr_3(x_sr_up)
        # x_sr_up=self.up_edsr_3(x_sr_up)
        # x_sr_up=self.up_conv_last(x_sr_up)

        return x_sr_up

    # def freeze_bn(self):
    #     for m in self.modules():
    #         if isinstance(m, SynchronizedBatchNorm2d):
    #             m.eval()
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.eval()



