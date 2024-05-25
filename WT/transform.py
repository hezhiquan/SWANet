import torch
import torch.nn as nn


def dwt_init(x):
    """
    离散小波变换
    :param x:[B,C,H,W]
    :return: [B,4C,H/2,W/2]
    """
    # (先对行进行奇数、偶数行采样，再对列进行奇数、偶数列采样，得到4个结果)
    # 将输入x进行二倍降采样，得到x01，x02
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    # 将x01，x02分别进行二倍降采样
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    # 
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    # print(x_HH[:, 0, :, :])
    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    """
    逆离散小波变换
    :param x: [B,C,H,W]
    :return: [B,C/4,2H,2W]
    """
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).to(x.device)  #

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        """
        离散小波变换
        :param x:[B,C,H,W]
        :return: [B,4C,H/2,W/2]
        """
        super(DWT, self).__init__()
        self.requires_grad = True

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    """
    逆离散小波变换
    :param x: [B,C,H,W]
    :return: [B,C/4,2H,2W]
    """
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = True

    def forward(self, x):
        return iwt_init(x)
