import torch
from torch import nn
import copy

# conv + relu + bn
class CRB(nn.Module):
    def __init__(self, ic, oc, ks, s, p):
        super().__init__()
        self.conv = nn.Conv2d(ic, oc, ks, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oc)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

DEFALT_ARGS = [
    # oc, ks, s, p
    [8, 3, 1, 1], # 112 / 112
    [16, 3, 2, 1], # 112 / 56
    [16, 3, 1, 1], # 56 / 56
    [32, 3, 2, 1], # 56 / 28
    [32, 3, 1, 1], # 28 / 28
    [64, 3, 2, 1], # 28 / 14
    [64, 3, 1, 1], # 14 / 14
    [128, 3, 2, 1], # 14 / 7
    [128, 3, 1, 1], # 7 / 7
    [256, 3, 2, 1], # 7 / 4
    [256, 4, 1, 0], # 4 / 1
]

class SmallVGG(nn.Module):
    def __init__(self, in_channels, feature_dim, net_args=DEFALT_ARGS, alpha=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.net_args = copy.deepcopy(net_args)
        self.net = nn.Sequential()
        for i, arg in enumerate(net_args):
            ic = int(net_args[i-1][0]*alpha) if i > 0 else in_channels
            self.net.add_module(
                'conv_%d'%i,
                CRB(ic, int(arg[0]*alpha), arg[1], arg[2], arg[3])
            )
        self.fc = nn.Linear(int(net_args[-1][0] * alpha), feature_dim)

    def forward(self, x):
        bs = x.shape[0]
        assert x.shape[1] == self.in_channels, 'x.shape=%s, do not meet requirement'%(x.shape)
        x = self.net(x)
        assert x.shape[2] == x.shape[3] and x.shape[2] == 1
        x = x.reshape([bs, -1])
        x = self.fc(x)
        return x
