import numpy as np
from thop import profile, clever_format
import torch
from torch import nn


def cal_eff_score(net, rounds=30, use_cuda=True):
    # define input tensor
    inp_tensor = torch.rand(1, 3, 512, 512)

    # deploy to cuda
    if use_cuda:
        inp_tensor = inp_tensor.cuda()
        net.cuda()

    # warm up GPU
    with torch.no_grad():
        for _ in range(10):
            _ = net(inp_tensor)

    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = np.zeros(rounds)
    with torch.no_grad():
        for r in range(rounds):
            starter.record()
            _ = net(inp_tensor)
            ender.record()
            torch.cuda.synchronize()
            times[r] = starter.elapsed_time(ender)
    avg_time = times.mean()
    avg_std = times.std()
    print("Test time: {}, {}".format(avg_time, avg_std))

    macs, params = profile(net, inputs=(inp_tensor,))

    macs_g = macs * 1e-9
    params_m = params * 1e-6

    # macs, params = clever_format([macs, params], '%.3f')
    # print('MACs (G): ', macs)
    # print('Params (M): ', params)
    print('MACs (G): ', macs_g)
    print('Params (M): ', params_m)


def cal_SWANet():
    from model.stage1.MIANet import MIANet
    from model.stage2.WNENet import WNENet

    class TwoStageNet(nn.Module):
        def __init__(self):
            super(TwoStageNet, self).__init__()
            self.net1 = MIANet(in_chn=3, wf=64, depth=4)
            self.net2 = WNENet(3, 3, 64)

        def forward(self, x):
            x1 = self.net1(x)
            x2, _ = self.net2(x1)
            return x2

    swa_net = TwoStageNet()
    cal_eff_score(swa_net, use_cuda=True)


if __name__ == "__main__":
    cal_SWANet()
