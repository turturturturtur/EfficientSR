import torch
import torch.nn as nn

from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class UNet(nn.Module):
    def __init__(self, in_chans=64, out_chans=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_chans, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, out_chans, 3, padding=1)
        )

    def forward(self, x):
        return self.encoder(x)
