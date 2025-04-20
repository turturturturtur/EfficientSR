import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class BicubicUpsampler(nn.Module):
    """Bicubic interpolation upsampler model.

    Args:
        scale (int): Upscaling factor. Typically 2, 3, or 4.
    """
    def __init__(self, upscale=4):
        super(BicubicUpsampler, self).__init__()
        self.scale = upscale
        self.a = nn.Parameter(torch.tensor([1],dtype=torch.float32))

    def forward(self, x):
        # 使用双三次插值进行上采样
        return F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
