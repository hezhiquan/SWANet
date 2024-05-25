import torch.nn as nn
import torch

from utils.conv_util import conv


class CBAMBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        print("[attention: init {}]".format(self.__class__.__name__))
        super(CBAMBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel // reduction), bias=False),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel // reduction), channel, bias=False),
                                                )
        self.sigmoid = nn.Sigmoid()

        self.spatial_excitation = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7,
                                                          stride=1, padding=3, bias=False),
                                                )

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        # 通道注意力模块
        # 通道平均池化
        chn_avg = self.avg_pool(x).view(bahs, chs)
        chn_avg = self.channel_excitation(chn_avg).view(bahs, chs, 1, 1)
        # 通道最大池化
        chn_max = self.max_pool(x).view(bahs, chs)
        chn_max = self.channel_excitation(chn_max).view(bahs, chs, 1, 1)
        chn_add = chn_avg + chn_max
        chn_add = self.sigmoid(chn_add)

        chn_cbam = torch.mul(x, chn_add)

        # 空间注意力模块
        avg_out = torch.mean(chn_cbam, dim=1, keepdim=True)
        max_out, _ = torch.max(chn_cbam, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        spa_add = self.spatial_excitation(cat)
        spa_add = self.sigmoid(spa_add)

        spa_cbam = torch.mul(chn_cbam, spa_add)
        return spa_cbam


class SENet(nn.Module):
    def __init__(self, channel, reduction=16):
        print("[attention: 初始化 {}]".format(self.__class__.__name__))
        super(SENet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # b为batch
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # return x * y.expand_as(x)
        return torch.mul(x, y)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        print("[attention: 初始化 {}]".format(self.__class__.__name__))
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        sig = self.sigmoid(avgout + maxout)
        return sig * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        print("[attention: 初始化 {}]".format(self.__class__.__name__))
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        combine = torch.cat([avgout, maxout], dim=1)
        combine = self.conv(combine)
        result = self.sigmoid(combine)
        # return result.repeat(1, 64, 1, 1)
        return result * x


class DAU(nn.Module):
    """
    DAU模块
    ref:MIRNet
    """

    def __init__(self, n_feat, kernel_size=3, reduction=16, bias=False):
        print("[attention: 初始化 {}]".format(self.__class__.__name__))
        super(DAU, self).__init__()
        modules_body = \
            [
                conv(n_feat * 2, n_feat, kernel_size, bias=bias),
                nn.ReLU(),
                conv(n_feat, n_feat * 2, kernel_size, bias=bias)
            ]
        # 一系列的卷积、relu操作,是DAU模块的前三个小模块(conv、prelu、conv)
        self.body = nn.Sequential(*modules_body)
        self.WSA = SALayer()
        self.WCA = CALayer(n_feat * 2, reduction, bias=bias)

        self.conv1x1 = nn.Conv2d(n_feat * 4, n_feat * 2, kernel_size=1, bias=bias)

    def forward(self, x):
        res = self.body(x)
        # 空间、通道注意力分支  ( DAU )
        branch_sa = self.WSA(res)
        branch_ca = self.WCA(res)
        res = torch.cat([branch_sa, branch_ca], dim=1)
        res = self.conv1x1(res) + x
        return res


class DAU_V2(nn.Module):
    """
    DAU模块
    ref:MIRNet
    """

    def __init__(self, n_feat, kernel_size=3, reduction=16, bias=False):
        print("[attention: 初始化 {}]".format(self.__class__.__name__))
        super(DAU_V2, self).__init__()

        # 一系列的卷积、relu操作,是DAU模块的前三个小模块(conv、prelu、conv)
        self.body = conv(n_feat, n_feat, 1, True, 1)
        self.WSA = SALayer()
        self.WCA = CALayer(n_feat, reduction, bias=bias)

        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        res = self.body(x)
        # 空间、通道注意力分支  ( DAU )
        branch_sa = self.WSA(res)
        branch_ca = self.WCA(res)
        res = torch.cat([branch_sa, branch_ca], dim=1)
        res = self.conv1x1(res) + x
        return res


class CBAM_ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CBAM_ChannelAttention, self).__init__()
        # 平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 最大池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # MLP  除以16是降维系数
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  # kernel_size=1
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # 结果相加
        out = avg_out + max_out
        return self.sigmoid(out) * x


# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        print("[Module:{}; reduction:{}]".format(self.__class__.__name__, reduction))
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# Spatial Attention Layer
class SALayer(nn.Module):
    def __init__(self, kernel_size=5, bias=False):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        # torch.max will output 2 things, and we want the 1st one
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, 1, keepdim=True)
        channel_pool = torch.cat([max_pool, avg_pool], dim=1)  # [N,2,H,W]  could add 1x1 conv -> [N,3,H,W]
        y = self.conv_du(channel_pool)

        return x * y


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
        # self.gate_activation = gate_activation
        self.gate_c = nn.Sequential()
        self.gate_c.add_module('flatten', Flatten())
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range(len(gate_channels) - 2):
            self.gate_c.add_module('gate_c_fc_%d' % i, nn.Linear(gate_channels[i], gate_channels[i + 1]))
            self.gate_c.add_module('gate_c_bn_%d' % (i + 1), nn.BatchNorm1d(gate_channels[i + 1]))
            self.gate_c.add_module('gate_c_relu_%d' % (i + 1), nn.ReLU())
        self.gate_c.add_module('gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]))

    def forward(self, in_tensor):
        avg_pool = F.avg_pool2d(in_tensor, in_tensor.size(2), stride=in_tensor.size(2))
        return self.gate_c(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)


class SpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module('gate_s_conv_reduce0',
                               nn.Conv2d(gate_channel, gate_channel // reduction_ratio, kernel_size=1))
        self.gate_s.add_module('gate_s_bn_reduce0', nn.BatchNorm2d(gate_channel // reduction_ratio))
        self.gate_s.add_module('gate_s_relu_reduce0', nn.ReLU())
        for i in range(dilation_conv_num):
            self.gate_s.add_module('gate_s_conv_di_%d' % i,
                                   nn.Conv2d(gate_channel // reduction_ratio, gate_channel // reduction_ratio,
                                             kernel_size=3, padding=dilation_val, dilation=dilation_val))
            self.gate_s.add_module('gate_s_bn_di_%d' % i, nn.BatchNorm2d(gate_channel // reduction_ratio))
            self.gate_s.add_module('gate_s_relu_di_%d' % i, nn.ReLU())
        self.gate_s.add_module('gate_s_conv_final', nn.Conv2d(gate_channel // reduction_ratio, 1, kernel_size=1))

    def forward(self, in_tensor):
        return self.gate_s(in_tensor).expand_as(in_tensor)


class BAM(nn.Module):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)

    def forward(self, in_tensor):
        # att = 1 + F.sigmoid( self.channel_att(in_tensor) * self.spatial_att(in_tensor) )
        # att = 1 + torch.nn.Sigmoid()( self.channel_att(in_tensor) * self.spatial_att(in_tensor) )
        att = 1 + torch.sigmoid(self.channel_att(in_tensor) + self.spatial_att(in_tensor))
        return att * in_tensor
