import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class Srcnn(nn.Module):
    """端到端 SRCNN 模型，内部包含 bicubic 插值 + SRCNN 卷积恢复。

    Args:
        scale (int): 上采样倍数，支持 2, 3, 4 等。
        num_channels (int): 输入输出通道数，灰度图为1，彩色图为3。
    """
    def __init__(self, upscale=2, num_in_ch=3, num_out_ch=3, num_feats=32):
        super(Srcnn, self).__init__()
        self.scale = upscale

        # SRCNN 结构（与论文一致）
        self.conv1 = nn.Conv2d(num_in_ch, num_feats, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(num_feats, num_feats, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(num_feats, num_out_ch, kernel_size=5, padding=5 // 2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Bicubic 上采样
        x = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)

        # SRCNN 卷积恢复过程
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        return x
