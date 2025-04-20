import torch
from torch import nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY

class Part(nn.Module):
    """一个包含 1x1, 3x3, 5x5 并联卷积的多尺度模块"""

    def __init__(self, in_channels=265, out_channels=88):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.fusion = nn.Conv2d(out_channels * 3, in_channels, kernel_size=1)

    def forward(self, x):
        out1 = self.conv1x1(x)
        out2 = self.conv3x3(x)
        out3 = self.conv5x5(x)
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.fusion(out)
        return out


@ARCH_REGISTRY.register()
class MyArch_conv(nn.Module):
    """MyArch: 多尺度 Inception 风格的轻量级上采样超分模型"""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=265, num_block=2, upscale=2):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, padding=1)

        self.body = nn.ModuleList([Part(num_feat, num_feat // 3) for _ in range(num_block)])

        self.upsample = nn.Sequential(
            nn.Conv2d(num_feat, num_feat * (upscale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale)
        )

        self.conv_last = nn.Conv2d(num_feat, num_out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_first(x)
        for block in self.body:
            x = block(x)
        x = self.upsample(x)
        x = self.conv_last(x)
        return x

@ARCH_REGISTRY.register()
class RFDNLite(nn.Module):
    """轻量级RFDN结构，适用于实时SR任务"""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=48, num_block=4, upscale=2):
        super(RFDNLite, self).__init__()
        self.fea_conv = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.blocks = nn.Sequential(*[RFDB(num_feat) for _ in range(num_block)])
        self.csa = nn.Conv2d(num_block * num_feat, num_feat, 1, 1, 0)
        self.upsample = nn.Sequential(
            nn.Conv2d(num_feat, num_feat * (upscale ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale),
            nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        )

    def forward(self, x):
        fea = self.fea_conv(x)
        outs = []
        out = fea
        for block in self.blocks:
            out = block(out)
            outs.append(out)
        out = self.csa(torch.cat(outs, 1))
        out = self.upsample(out)
        return out

@ARCH_REGISTRY.register()
class RFDN_RES(nn.Module):
    """RFDN with inter-block residuals and global input-output residual"""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=48, num_block=4, upscale=2):
        super().__init__()
        self.fea_conv = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.blocks = nn.ModuleList([RFDB(num_feat) for _ in range(num_block)])
        self.csa = nn.Conv2d(num_block * num_feat, num_feat, 1, 1, 0)

        self.upsample = nn.Sequential(
            nn.Conv2d(num_feat, num_feat * (upscale ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale),
            nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        )

    def forward(self, x):
        fea = self.fea_conv(x)
        outs = []
        out = fea

        for block in self.blocks:
            out = block(out) + out  # 每个块之间残差
            outs.append(out)

        out = self.csa(torch.cat(outs, dim=1))
        out = self.upsample(out)
        return out + F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)  # 首尾大残差


class RFDB(nn.Module):
    def __init__(self, in_channels):
        super(RFDB, self).__init__()
        d = in_channels
        self.c1_d = nn.Conv2d(d, d//2, 1, 1, 0)        # in: d -> d/2
        self.c2_d = nn.Conv2d(d//2, d//2, 3, 1, 1)     # in: d/2 -> d/2
        self.c3_d = nn.Conv2d(d//2, d//2, 3, 1, 1)     # in: d/2 -> d/2
        self.fusion = nn.Conv2d(d + 3*(d//2), d, 1, 1, 0)
        self.act = nn.LeakyReLU(0.05, inplace=True)


    def forward(self, x):   
        out1 = self.act(self.c1_d(x))
        out2 = self.act(self.c2_d(out1))
        out3 = self.act(self.c3_d(out2))
        out = self.fusion(torch.cat([x, out1, out2, out3], 1))
        return out

@ARCH_REGISTRY.register()
class RFDNLiteReparam(nn.Module):
    """基于重参数化和深度可分离卷积的轻量级RFDN"""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=48, num_block=4, upscale=2):
        super().__init__()
        self.fea_conv = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.blocks = nn.Sequential(*[ReparamRFDB(num_feat) for _ in range(num_block)])
        self.csa = nn.Conv2d(num_block * num_feat, num_feat, 1, 1, 0)
        self.upsample = nn.Sequential(
            nn.Conv2d(num_feat, num_feat * (upscale ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale),
            nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        )

    def forward(self, x):
        fea = self.fea_conv(x)
        outs = []
        out = fea
        for block in self.blocks:
            out = block(out)
            outs.append(out)
        out = self.csa(torch.cat(outs, dim=1))
        out = self.upsample(out)
        return out


class ReparamRFDB(nn.Module):
    """使用深度可分离卷积 + 重参数化思想的RFDB模块"""

    def __init__(self, in_channels):
        super().__init__()
        d = in_channels
        self.act = nn.LeakyReLU(0.05, inplace=True)

        self.branch1 = DepthSeparableConv(d, d // 2, 1)
        self.branch2 = DepthSeparableConv(d // 2, d // 2, 3)
        self.branch3 = DepthSeparableConv(d // 2, d // 2, 3)

        self.fusion = nn.Conv2d(d + 3 * (d // 2), d, 1, 1, 0)

    def forward(self, x):
        out1 = self.act(self.branch1(x))
        out2 = self.act(self.branch2(out1))
        out3 = self.act(self.branch3(out2))
        out = self.fusion(torch.cat([x, out1, out2, out3], dim=1))
        return out


class DepthSeparableConv(nn.Module):
    """深度可分离卷积模块"""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=True)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))
    
@ARCH_REGISTRY.register()
class HybridRFDNLite(nn.Module):
    """使用 HybridRFDB 的轻量化 RFDN 网络结构"""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=48, num_block=4, upscale=2):
        super().__init__()
        self.fea_conv = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.blocks = nn.Sequential(*[HybridRFDB(num_feat) for _ in range(num_block)])
        self.csa = nn.Conv2d(num_block * num_feat, num_feat, 1, 1, 0)
        self.upsample = nn.Sequential(
            nn.Conv2d(num_feat, num_feat * (upscale ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale),
            nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        )

    def forward(self, x):
        fea = self.fea_conv(x)
        outs = []
        out = fea
        for block in self.blocks:
            out = block(out)
            outs.append(out)
        out = self.csa(torch.cat(outs, dim=1))
        out = self.upsample(out)
        return out



class HybridRFDB(nn.Module):
    """Hybrid RFDB：标准卷积 + 深度可分离卷积 + SE注意力"""

    def __init__(self, in_channels):
        super().__init__()
        d = in_channels
        self.act = nn.LeakyReLU(0.05, inplace=True)

        # 分支结构
        self.branch1 = nn.Conv2d(d, d // 2, 1, 1, 0)
        self.branch2 = DepthSeparableConv(d // 2, d // 2, 3)
        self.branch3 = DepthSeparableConv(d // 2, d // 2, 3)

        # 融合层
        self.fusion = nn.Conv2d(d + 3 * (d // 2), d, 1, 1, 0)

        # SE注意力
        self.se = SELayer(d)

    def forward(self, x):
        out1 = self.act(self.branch1(x))
        out2 = self.act(self.branch2(out1))
        out3 = self.act(self.branch3(out2))
        concat = torch.cat([x, out1, out2, out3], dim=1)
        out = self.fusion(concat)
        out = self.se(out)
        return out


class DepthSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=True)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class SELayer(nn.Module):
    """Squeeze-and-Excitation 注意力模块"""
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)