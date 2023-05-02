import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import argparse
import numpy as np
from torch import nn
from torch.nn import init
from os.path import join
from lib import resnet
# from lib.pvtv2 import pvt_v2_b2
#from lib.Res2Net_v1b import res2net50_v1b_26w_4s


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class PA(nn.Module):
    def __init__(self, in_dim):
        super(PA, self).__init__()
        # Posting-H
        self.query_conv_h = nn.Conv2d(in_dim, in_dim, kernel_size=(1, 1))
        self.key_conv_h = nn.Conv2d(in_dim, in_dim, kernel_size=(1, 1))
        self.value_conv_h = nn.Conv2d(in_dim, in_dim, kernel_size=(1, 1))

        # Posting-W
        self.query_conv_w = nn.Conv2d(in_dim, in_dim, kernel_size=(1, 1))
        self.key_conv_w = nn.Conv2d(in_dim, in_dim, kernel_size=(1, 1))
        self.value_conv_w = nn.Conv2d(in_dim, in_dim, kernel_size=(1, 1))

        self.la = nn.Parameter(torch.zeros(1))
        self.lb = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        # finally refine
        self.conv_final = nn.Conv2d(2 * in_dim, in_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.size()
        axis_h = 1
        axis_h *= H
        view = (B, -1, axis_h)
        projected_query_h = self.query_conv_h(x).view(*view).permute(0, 2, 1)  # [B,H,CW]
        projected_key_h = self.key_conv_h(x).view(*view)  # [B,CW,H]
        attention_map_h = torch.bmm(projected_query_h, projected_key_h)  # [B,H,H]
        attention_h = self.softmax(attention_map_h)  # [B,H,H]
        projected_value_h = self.value_conv_h(x).view(*view)  # [B,WC,H]
        out_h = torch.bmm(projected_value_h, attention_h.permute(0, 2, 1))  # [B,wc,h]
        out_h = out_h.view(B, C, H, W)  # [b, c, h, w]
        out_h = self.la * out_h
        # Position-W
        axis_w = 1
        axis_w *= W
        view = (B, -1, axis_w)
        projected_query_w = self.query_conv_w(x).view(*view).permute(0, 2, 1)  # [B,H,CW]
        projected_key_w = self.key_conv_w(x).view(*view)  # [B,CW,H]
        attention_map_w = torch.bmm(projected_query_w, projected_key_w)  # [B,H,H]
        attention_w = self.softmax(attention_map_w)  # [B,H,H]
        projected_value_w = self.value_conv_w(x).view(*view)  # [B,WC,H]
        out_w = torch.bmm(projected_value_w, attention_w.permute(0, 2, 1))  # [B,wc,h]
        out_w = out_w.view(B, C, H, W)  # [b, c, h, w]
        out_w = self.lb * out_w
        out_fianl = torch.cat([out_h, out_w], 1)
        out_final = self.conv_final(out_fianl) + x
        return out_final


class CPM(nn.Module):
    def __init__(self, in_dim):
        super(CPM, self).__init__()
        self.pa = PA(in_dim)
    def forward(self, xg, gt):
        _, _, H, W = xg.size()
        if self.training:
            gt = F.interpolate(gt, size=(H, W), mode='bilinear', align_corners=True)
            pos = self.pa(xg * gt)
            nes = self.pa(xg * (1 - gt))
            fg = self.pa(xg)

            v1 = torch.mean(fg, (0, 2, 3), True)
            v2 = torch.mean(pos, (0, 2, 3), True)
            v3 = torch.mean(nes, (0, 2, 3), True)
        else:
            fg = self.pa(xg)
            v1 = None
            v2 = None
            v3 = None
        return fg, v1, v2, v3


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class BEM(nn.Module):
    def __init__(self):
        super(BEM, self).__init__()

        self.conv_1 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.conv_2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.conv_3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.conv_4 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.conv_out = BasicConv2d(64, 1, kernel_size=1)

        self.sa = SpatialAttention()

    def forward(self, x1, x2, fg):
        if x2.size()[2:] != x1.size()[2:]:
            x2 = F.interpolate(x2, size=x1.size()[2:], mode='bilinear')
        if fg.size()[2:] != x1.size()[2:]:
            fg = F.interpolate(fg, size=x1.size()[2:], mode='bilinear')
        x12 = x1 + x2
        x12_g = x12 * fg
        x12_conv = self.conv_2(self.conv_1(x12_g))
        xe = x12_conv + x12_g
        xe = self.conv_4(xe)

        xe_sa = self.sa(xe) * xe

        xe_conv = self.conv_3(xe_sa)
        out = xe_conv + xe

        fe_out = self.conv_out(out)
        return out, fe_out


class ASPP(nn.Module):
    def __init__(self, outchannel):
        super(ASPP, self).__init__()
        self.conv1 = BasicConv2d(outchannel, outchannel, kernel_size=3, padding=1, dilation=1)
        self.conv2 = BasicConv2d(outchannel, outchannel, kernel_size=3, padding=2, dilation=2)
        self.conv3 = BasicConv2d(outchannel, outchannel, kernel_size=3, padding=3, dilation=3)
        self.conv4 = BasicConv2d(outchannel, outchannel, kernel_size=3, padding=4, dilation=4)
        self.conv0 = BasicConv2d(outchannel, outchannel, kernel_size=1)

        self.conv = BasicConv2d(5 * outchannel, outchannel, kernel_size=1)

        self.rconv = nn.Sequential(
            nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU()
        )

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        out = torch.cat((x0, x1, x2, x3, x4), dim=1)
        out = self.conv(out)

        out = out + x
        out = self.rconv(out)

        return out


class AIM(nn.Module):
    def __init__(self, channel):
        super(AIM, self).__init__()
        self.conv_1 = BasicConv2d(channel, 64, kernel_size=3, stride=1, padding=1)
        self.conv_2 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_3 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv_12 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_13 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_4 = BasicConv2d(2 * 64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_5 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_6 = BasicConv2d(2 * 64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_7 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.local_att = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.Sigmoid()
        )
        # global_att
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.Sigmoid()
        )
        self.aspp = ASPP(64)

    def forward(self, x, xe, xg):
        # x: 代表当前层特征; xe: 代表边界特征; xg: 代表全局特征
        if xg.size()[2:] != x.size()[2:]:
            xg = F.interpolate(xg, size=x.size()[2:], mode='bilinear')
        if xe.size()[2:] != x.size()[2:]:
            xe = F.interpolate(xe, size=x.size()[2:], mode='bilinear')

        x1 = xg
        x2 = xe

        x = self.conv_1(x)
        x1 = self.conv_2(x1)
        x2 = self.conv_3(x2)

        x_1 = self.conv_12(x * x1)
        x_2 = self.conv_13(x * x2)

        x_1_att = self.global_att(x_1) * x_1
        x_1_conv = self.conv_5(x_1)

        x_2_cat = torch.cat((x_1_att, x_2), 1)
        x_2_cat_conv = self.conv_4(x_2_cat)

        x_2_att = self.local_att(x_2) * x_2
        x_1_cat = torch.cat((x_2_att, x_1_conv), 1)
        x_1_cat_conv = self.conv_6(x_1_cat)

        xe = self.conv_7(x_2_cat_conv)
        xg = x_1_cat_conv
        final = xe + xg
        final = self.aspp(final)
        return final


class Network(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=64, imagenet_pretrained=True):
        super(Network, self).__init__()
        self.resnet = resnet.resnet50(backbone_path='/hy-tmp/SINet-V2-main/lib/resnet50-19c8e357.pth')

        self.reduce4 = BasicConv2d(2048, 64, kernel_size=1)
        self.reduce3 = BasicConv2d(1024, 64, kernel_size=1)
        self.reduce2 = BasicConv2d(512, 64, kernel_size=1)
        self.reduce1 = BasicConv2d(256, 64, kernel_size=1)

        self.aim4 = AIM(64)
        self.aim3 = AIM(64)
        self.aim2 = AIM(64)
        self.aim1 = AIM(64)

        self.e = BEM()

        self.g = CPM(64)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv64_1 = nn.Conv2d(64, 1, 1)

        self.decoder_conv1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64),
                                           nn.ReLU(inplace=True))
        self.decoder_conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64),
                                           nn.ReLU(inplace=True))
        self.decoder_conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64),
                                           nn.ReLU(inplace=True))

    def forward(self, x, gt):
        # backbone
        r0, r1, r2, r3, r4 = self.resnet(x)
        r1 = self.reduce1(r1)
        r2 = self.reduce2(r2)
        r3 = self.reduce3(r3)
        r4 = self.reduce4(r4)

        # ----CRM---#
        f_g, v1, v2, v3 = self.g(r4, gt)
        # ---BEM---#
        f_e, f_e_out = self.e(r1, r2, f_g)

        S_e_pred = torch.sigmoid(f_e_out)
        S_e_pred = F.interpolate(S_e_pred, scale_factor=4,
                                 mode='bilinear')
        # Decoder
        f4 = self.aim4(r4, f_e, f_g)
        f3 = self.aim3(r3, f_e, f_g)
        f2 = self.aim2(r2, f_e, f_g)
        f1 = self.aim1(r1, f_e, f_g)

        S_3 = self.decoder_conv1(f3 + self.upsample(f4))
        S_2 = self.decoder_conv2(f2 + self.upsample(S_3))
        S_1 = self.decoder_conv3(f1 + self.upsample(S_2))

        S_4_pred = F.interpolate(self.conv64_1(f4), scale_factor=32,
                                 mode='bilinear')
        S_3_pred = F.interpolate(self.conv64_1(S_3), scale_factor=16,
                                 mode='bilinear')
        S_2_pred = F.interpolate(self.conv64_1(S_2), scale_factor=8,
                                 mode='bilinear')
        S_1_pred = F.interpolate(self.conv64_1(S_1), scale_factor=4,
                                 mode='bilinear')
        return v1, v2, v3, S_e_pred, S_4_pred, S_3_pred, S_2_pred, S_1_pred


if __name__ == '__main__':
    import numpy as np
    from time import time

    net = Network(imagenet_pretrained=False)
    net.eval()

    dump_x = torch.randn(1, 3, 384, 384)
    frame_rate = np.zeros((1000, 1))
    for i in range(1000):
        start = time()
        y = net(dump_x)
        end = time()
        running_frame_rate = 1 * float(1 / (end - start))
        print(i, '->', running_frame_rate)
        frame_rate[i] = running_frame_rate
    print(np.mean(frame_rate))
    print(y.shape)