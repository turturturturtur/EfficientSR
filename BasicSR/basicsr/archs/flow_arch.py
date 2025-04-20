import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer
from basicsr.utils.registry import ARCH_REGISTRY

class NaiveLinear(nn.Module):
    def __init__(self, features=3):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(features))
        self._weight = nn.Parameter(torch.empty(features, features))
        stdv = 1.0 / np.sqrt(8)
        nn.init.uniform_(self._weight, -stdv, stdv)

    def forward(self, x):
        return F.linear(x, self._weight, self.bias), torch.slogdet(self._weight)[1]

    def inverse(self, x):
        return torch.linalg.solve(self._weight, (x - self.bias).T).T

@ARCH_REGISTRY.register()
class Flow(nn.Module):
    def __init__(self, flow_layers=10):
        super().__init__()
        self.n_layers = flow_layers
        self.linears = nn.ModuleList([NaiveLinear(3) for _ in range(flow_layers)])

    def forward(self, x):
        for layer in self.linears:
            x, _ = layer(x)
        return x

    def inverse(self, x):
        for layer in reversed(self.linears):
            x = layer.inverse(x)
        return x
