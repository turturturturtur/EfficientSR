import torch
from torch import nn
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class BicubicInterpolationSR(nn.Module):
    """Single Image Super-Resolution using Bicubic Interpolation.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 3.
        upscale (int): Upsampling factor. Support integers > 1. Default: 4.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, upscale=4):
        super(BicubicInterpolationSR, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.upscale = upscale

        # Ensure the input and output channels match
        assert num_in_ch == num_out_ch, "num_in_ch and num_out_ch must be equal."

    def forward(self, x):
        """Forward pass for bicubic interpolation.

        Args:
            x (torch.Tensor): Input image tensor with shape (N, C, H, W).

        Returns:
            torch.Tensor: Upscaled image tensor with shape (N, C, H*scale, W*scale).
        """
        # Using PyTorch's interpolate function with bicubic mode
        x_upscaled = torch.nn.functional.interpolate(
            x, scale_factor=self.upscale, mode='bicubic', align_corners=False
        )
        return x_upscaled