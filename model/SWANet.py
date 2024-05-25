from torch import nn

from model.stage1.MIANet import MIANet
from model.stage2.WNENet import WNENet
from utils import load_checkpoint


def load_stage1_net(weight1_path):
    net = MIANet(in_chn=3, wf=64, depth=4)
    load_checkpoint(net, weight1_path, map_location="cpu")
    return net


def load_stage2_net(weight2_path):
    net = WNENet(3, 3, 64)
    load_checkpoint(net, weight2_path, map_location="cpu")
    return net


class SWANet(nn.Module):
    def __init__(self, weight1_path, weight2_path):
        super(SWANet, self).__init__()
        self.net1 = load_stage1_net(weight1_path)
        self.net2 = load_stage2_net(weight2_path)

    def forward(self, x):
        x1 = self.net1(x)
        x2, _ = self.net2(x1)
        return x2
