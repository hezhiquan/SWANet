import math
import torch
import torch.nn as nn


def get_freq_indices(method):
    """
    根据不同的方法, 获取前k个频率分量的下标
    Args:
        method: top-k,bot-k,low-k
                low-k：根据频率由低到高选取
                top-k：根据上文的性能排名由高到低选取
                bot-k：根据上文的性能排名有低到高选取
    Returns:

    """
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    # 被选取的频率分量的个数
    num_freq = int(method[3:])
    # 三种下标的设置原因见 论文V3 中附加材料的图5
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method='top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7
        """
        上述代码的想法很简单，假设输入为n x n，那么其总共有n x n个频谱。但其实在一张14x14的特征图上的(2,2)频率分量,他的绝对频率
        是和一张7x7的特征图上的(1,1)分量是一致的。因此我们为了保证不同stage（也即不同分辨率）的频率都是一致的，我们做了这样的归一化，
        这就导致分辨率不会影响结果。即使在COCO这种本身图像大小不一的数据集上也能work。
        我们设置这个c2wh主要还是为了在固定分辨率和网络结构的情况下（一般是分类网络），提前确定好大小并预先生成对应频谱的DCT 权重，
        而不用每次都resize，这样应该会快一点。
        (https://github.com/cfzd/FcaNet/issues/21)
        """

        self.dct_layer = MultiSpectralDCTLayer(
            dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(
                x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralAttentionLayerV2(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=128, freq_sel_method='top16'):
        super(MultiSpectralAttentionLayerV2, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7
        """
        上述代码的想法很简单，假设输入为n x n，那么其总共有n x n个频谱。但其实在一张14x14的特征图上的(2,2)频率分量,他的绝对频率
        是和一张7x7的特征图上的(1,1)分量是一致的。因此我们为了保证不同stage（也即不同分辨率）的频率都是一致的，我们做了这样的归一化，
        这就导致分辨率不会影响结果。即使在COCO这种本身图像大小不一的数据集上也能work。
        我们设置这个c2wh主要还是为了在固定分辨率和网络结构的情况下（一般是分类网络），提前确定好大小并预先生成对应频谱的DCT 权重，
        而不用每次都resize，这样应该会快一点。
        (https://github.com/cfzd/FcaNet/issues/21)
        """

        self.dct_layer = MultiSpectralDCTLayer(
            dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel * reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel * reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(
                x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0
        # 要选取的频率分量的个数
        self.num_freq = len(mapper_x)
        # 参数有不同的初始化方式(随机或者DCT初始化); 参数可变或者固定; 两两组合得到四种参数设置方式
        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(
            height, width, mapper_x, mapper_y, channel))

        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + \
                                  str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        """
        获取二维DCT basic function B的一半 (B=cos(--)cos(--),这里只获取cos(--),在这个函数的外面组合两个cos的结果)
        :param pos:RGB域中的i或j
        :param freq: 频域中的h或w
        :param POS:RGB域中图像的大小(比如 7)
        :return:
        """
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)
        # 将通道分成 mapper_x 份, 其中一份通道数有 c_part 个通道
        # c_part=c/n
        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    # 更新每一份通道数的
                    # 右边的等式为 basic function B的计算公式(见 论文 v3的公式 7或 论文 final的公式 1)
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)

        return dct_filter


if __name__ == "__main__":
    x = torch.ones(1, 64, 64, 64, dtype=torch.float, requires_grad=False)
    model = MultiSpectralAttentionLayer(64, 64, 64)
    res = model(x)
    print(res.shape)
    print(res)
