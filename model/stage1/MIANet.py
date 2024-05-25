import torch
import torch.nn as nn
from utils import CBAMBlock, bili_resize, LayerNorm2d


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.body = [NAFBlock(in_size, in_size)]
        self.body = nn.Sequential(*self.body)

        if downsample:
            self.downsample = PS_down(out_size, out_size, downscale=2)

        self.tail = nn.Conv2d(in_size, out_size, kernel_size=1)

    def forward(self, x):
        out = self.body(x)
        out = self.tail(out)
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out


class EnhanceNetwork(nn.Module):
    def __init__(self, layers, in_channel, out_channel):
        super(EnhanceNetwork, self).__init__()
        print("[Init {}]".format(self.__class__.__name__))
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=1,
                      padding=padding),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_size, stride=1,
                      padding=padding),
            # nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=3, kernel_size=3, stride=1, padding=1),
            # nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)

        illu = fea + input
        return illu


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUpBlock, self).__init__()
        self.up = PS_up(in_size, out_size, upscale=2)
        self.conv_block = UNetConvBlock(in_size, out_size, downsample=False)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], dim=1)
        out = self.conv_block(out)
        return out


class PS_down(nn.Module):
    """
    PixelUnshuffle down sample
    input: x=[N,in_size,H,W]
    output: y=[N,out_size,H/downscale,W/downscale]
    """

    def __init__(self, in_size, out_size, downscale):
        super(PS_down, self).__init__()
        # x=[N,C,W,H], PixelUnshuffle(x)=[N, C x downscale x downscale, W/downscale, H/downscale]
        self.UnPS = nn.PixelUnshuffle(downscale)
        self.conv1 = nn.Conv2d((downscale ** 2) * in_size, out_size, 1, 1, 0)

    def forward(self, x):
        x = self.UnPS(x)  # h/2, w/2, 4*c
        x = self.conv1(x)
        return x


class PS_up(nn.Module):
    """
    PixelShuffle up sample
    intput: x=[N,in_size,H,W]
    output: y=[N,out_size,H*upscale,W*upscale]
    """

    def __init__(self, in_size, out_size, upscale):
        super(PS_up, self).__init__()
        # x=[N,C,W,H], PixelShuffle(x)=[N, C / (downscale^2), W x downscale, H x downscale]
        self.PS = nn.PixelShuffle(upscale)
        self.conv1 = nn.Conv2d(in_size // (upscale ** 2), out_size, 1, 1, 0)

    def forward(self, x):
        x = self.PS(x)
        x = self.conv1(x)
        return x


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, in_chn, out_chn, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        print("[Module {}]".format(self.__class__.__name__))
        super().__init__()
        c = in_chn
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        # depth-wise conv
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=out_chn, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class MIANet(nn.Module):
    def __init__(self, in_chn=3, wf=64, depth=4):
        """
        :param in_chn: input channel
        :param wf: number of filters in the first layer
        :param depth: The number of layers in the vertical direction of the network. In this article, it is 4 layers
        """
        super(MIANet, self).__init__()
        self.depth = depth
        self.down_path = nn.ModuleList()
        self.bili_down = bili_resize(0.5)
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)

        # encoder of UNet-64
        prev_channels = 0
        for i in range(depth):  # 0,1,2,3
            downsample = True if (i + 1) < depth else False
            self.down_path.append(UNetConvBlock(prev_channels + wf, (2 ** i) * wf, downsample))
            prev_channels = (2 ** i) * wf

        # decoder of UNet-64
        self.up_path = nn.ModuleList()
        self.skip_conv = nn.ModuleList()
        self.conv_up = nn.ModuleList()

        self.bottom_conv = nn.Conv2d(prev_channels, wf, 3, 1, 1)
        # upsample 8x
        self.bottom_up = bili_resize(2 ** (depth - 1))

        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, (2 ** i) * wf))
            self.skip_conv.append(nn.Conv2d((2 ** i) * wf, (2 ** i) * wf, 3, 1, 1))
            self.conv_up.append(nn.Sequential(*[bili_resize(2 ** i), nn.Conv2d((2 ** i) * wf, wf, 3, 1, 1)]))
            prev_channels = (2 ** i) * wf

        self.final_ff = CBAMBlock(wf)

        self.res_conv3x3 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.conv3x3 = nn.Conv2d(wf, 3, 3, 1, 1)
        self.correction_net = EnhanceNetwork(1, 3, 3)

    def forward(self, x):
        img = x
        # Input data for down sampling
        scale_img = img

        # Use 3x3 convolution to convert 3-channel images into WF channels
        x1 = self.conv_01(img)
        # encs: encoders output
        encs = []
        # UNet-64
        # Down-path (Encoder)
        for i, down in enumerate(self.down_path):
            if i == 0:
                # x1 is the result of down sampling and is sent to the next layer; x1_up is sent to the same layer
                x1, x1_up = down(x1)
                encs.append(x1_up)
            elif (i + 1) < self.depth:
                scale_img = self.bili_down(scale_img)
                left_bar = self.conv_01(scale_img)
                x1 = torch.cat([x1, left_bar], dim=1)
                x1, x1_up = down(x1)
                encs.append(x1_up)
            else:
                scale_img = self.bili_down(scale_img)
                left_bar = self.conv_01(scale_img)
                x1 = torch.cat([x1, left_bar], dim=1)
                x1 = down(x1)

        # Up-path (Decoder)

        ms_result = [self.bottom_up(self.bottom_conv(x1))]
        for i, up in enumerate(self.up_path):
            x1 = up(x1, self.skip_conv[i](encs[-i - 1]))
            ms_result.append(self.conv_up[i](x1))

        # Merge multiple scales
        batch_size, n_feats, H, W = ms_result[1].shape
        inp_feats = torch.cat(ms_result, dim=1)
        # [B,height,C,H,W]
        inp_feats = inp_feats.view(batch_size, self.depth, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        feats_U = torch.sum(inp_feats, dim=1)
        msff_result = self.final_ff(feats_U)

        correct_net_input = msff_result + self.res_conv3x3(img)
        correct_net_input = self.conv3x3(correct_net_input)
        out_1 = self.correction_net(correct_net_input) + img

        return out_1


if __name__ == "__main__":
    from thop import profile, clever_format

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_data = torch.ones(1, 3, 400, 592, dtype=torch.float, requires_grad=False).to(device)
    model = MIANet().to(device)
    print(model)
    out = model(input_data)
    flops, params = profile(model, inputs=(input_data,))
    flops_format, params_format = clever_format([flops, params], "%.3f")

    print('input shape:', input_data.shape)
    print('parameters:', params / 1e6)
    print('flops', flops / 1e9)
    print("flops_format: ", flops_format, "params_format: ", params_format)
    print('output shape', out.shape)
