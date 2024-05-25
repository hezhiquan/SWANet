import torch
from torch import nn
from torch.autograd import Function


# https://github.com/ZilinGao/Global-Second-order-Pooling-Convolutional-Networks/blob/68264e29acd122540315ec6e78040e4665560b78/torchvision/models/resnet.py
def CovpoolLayer(var):
    return Covpool.apply(var)


class Covpool(Function):
    @staticmethod
    def forward(ctx, input):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        h = x.data.shape[2]
        w = x.data.shape[3]
        M = h * w
        x = x.reshape(batchSize, dim, M)
        I_hat = (-1. / M / M) * torch.ones(M, M, device=x.device) + (1. / M) * torch.eye(M, M, device=x.device)
        I_hat = I_hat.view(1, M, M).repeat(batchSize, 1, 1).type(x.dtype)
        y = x.bmm(I_hat).bmm(x.transpose(1, 2))
        ctx.save_for_backward(input, I_hat)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        input, I_hat = ctx.saved_tensors
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        h = x.data.shape[2]
        w = x.data.shape[3]
        M = h * w
        x = x.reshape(batchSize, dim, M)
        grad_input = grad_output + grad_output.transpose(1, 2)
        grad_input = grad_input.bmm(x).bmm(I_hat)
        grad_input = grad_input.reshape(batchSize, dim, h, w)
        return grad_input


class GsopChannelAttention(nn.Module):

    def __init__(self, in_chn):
        super(GsopChannelAttention, self).__init__()
        reduction = 128
        print("[Module:{} reduction:{}]".format(self.__class__.__name__, reduction))

        self.relu_normal = nn.ReLU(inplace=False)
        self.relu = nn.ReLU(inplace=True)
        if in_chn > 64:
            DR_stride = 1
        else:
            DR_stride = 2

        self.ch_dim = in_chn
        self.conv_for_DR = nn.Conv2d(self.ch_dim, self.ch_dim, kernel_size=1, stride=DR_stride, bias=True)
        # row-wise conv is realized by group conv
        self.row_conv_group = nn.Conv2d(self.ch_dim, 4 * self.ch_dim, kernel_size=(self.ch_dim, 1), groups=self.ch_dim,
                                        bias=True)
        self.fc_adapt_channels = nn.Sequential(
            nn.Conv2d(4 * self.ch_dim, self.ch_dim * reduction, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.ch_dim * reduction, self.ch_dim, 1, padding=0, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # NxCxHxW
        out = self.relu_normal(x)
        out = self.conv_for_DR(out)
        out = self.relu(out)

        out = CovpoolLayer(out)  # Nxdxd
        out = out.view(out.size(0), out.size(1), out.size(2), 1).contiguous()  # Nxdxdx1

        out = self.row_conv_group(out)  # Nx512x1x1

        out = self.fc_adapt_channels(out)  # NxCx1x1
        out = self.sigmoid(out)  # NxCx1x1
        return out * x


if __name__ == "__main__":
    x = torch.ones(1, 64, 64, 64, dtype=torch.float, requires_grad=False)
    model = GsopChannelAttention(64)
    res = model(x)
    print(res.shape)
