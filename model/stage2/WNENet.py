import torch
from torch import nn
from WT import DWT, IWT
from utils import conv, conv1x1, conv3x3


class CALayerV2(nn.Module):
    def __init__(self, channel, factor=128, bias=False):
        super(CALayerV2, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # feature channel downscale and upscale --> channel weight
        print("[Module:{}; reduction:{}]".format(self.__class__.__name__, factor))
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel * factor, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel * factor, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class CAB(nn.Module):
    def __init__(self, in_chn, out_chn, factor=128):
        print("[info: factor:{}]".format(factor))
        super(CAB, self).__init__()
        self.head = nn.Sequential(conv(in_chn, in_chn, 3, True),
                                  nn.ReLU(),
                                  conv(in_chn, in_chn, 3, True)
                                  )
        self.CA = CALayerV2(in_chn, factor)
        self.tail = conv(in_chn, out_chn, 1, True)

    def forward(self, x):
        res = self.head(x)
        res = self.CA(res)
        res += x
        res = self.tail(res)
        return res


class DWTBranchV1(nn.Module):
    """
    v1:用4个卷积来自适应调整权重,concat
    """

    def __init__(self, in_chn, out_chn, width):
        super().__init__()
        self.dwt = DWT()
        self.iwt = IWT()
        width4 = width // 4
        self.conv_LL = conv3x3(in_chn, width4)
        self.conv_HL = conv3x3(in_chn, width4)
        self.conv_LH = conv3x3(in_chn, width4)
        self.conv_HH = conv3x3(in_chn, width4)
        self.conv = nn.Sequential(conv3x3(width, width), nn.ReLU())

        self.cab = CAB(width, width, factor=128)
        # 12/4=3
        self.conv_tail = conv1x1(width, in_chn * 4)

    def forward(self, x):
        LL, HL, LH, HH = torch.chunk(self.dwt(x), 4, dim=1)
        adjust_x = torch.cat([self.conv_LL(LL), self.conv_HL(HL), self.conv_LH(LH), self.conv_HH(HH)], dim=1)
        adjust_x = self.conv(adjust_x)
        x1 = self.cab(adjust_x)
        x1 = self.conv_tail(x1)
        x1 = self.iwt(x1)
        return x1


class WNENet(nn.Module):

    def __init__(self, in_chn=3, out_chn=3, width=64):
        super().__init__()
        print("[Module {} ; in_chn: {}; out_chn: {}; width: {}; ]".format(self.__class__.__name__, in_chn,
                                                                          out_chn,
                                                                          width))
        self.dwt_branch1 = DWTBranchV1(in_chn, out_chn, width)

    def forward(self, x):
        y1 = self.dwt_branch1(x)
        return y1, []


if __name__ == "__main__":
    from thop import profile, clever_format

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_data = torch.ones(4, 3, 400, 592, dtype=torch.float, requires_grad=False).to(device)

    model = WNENet(3, 3, 64).to(device)
    out = model(input_data)
    flops, params = profile(model, inputs=(input_data,))
    flops_format, params_format = clever_format([flops, params], "%.3f")
    print('input shape:', input_data.shape)
    print('parameters:', params / 1e6)
    print('flops', flops / 1e9)
    print("flops_format: ", flops_format, "params_format: ", params_format)
    print('output shape', out.shape)
