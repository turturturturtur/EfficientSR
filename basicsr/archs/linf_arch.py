import torch
import torch.nn as nn
from basicsr.archs.edsr_arch import EDSR
from basicsr.archs.flow_arch import Flow
from basicsr.archs.unet_arch import UNet

from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class LinfSRNet(nn.Module):
    def __init__(self, encoder, flow, unet):
        super().__init__()
        self.encoder = encoder
        self.flow = flow
        self.unet = unet

    def forward(self, inp):
        feat = self.encoder(inp)
        flow_out = self.flow(feat)
        out = self.unet(flow_out)
        return out
