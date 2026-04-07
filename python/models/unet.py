"""Small U-Net used for BEV denoising and semantic reconstruction."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


SEMANTIC_CLASS_NAMES = ("drivable", "lane", "vehicle", "pedestrian", "obstacle")
NUM_SEMANTIC_CLASSES = len(SEMANTIC_CLASS_NAMES)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with group norm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class DownBlock(nn.Module):
    """Encoder stage: max-pool followed by feature extraction."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Decoder stage: resize, concatenate skip features, then refine."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat((skip, x), dim=1)
        return self.conv(x)


class SmallUNet(nn.Module):
    """Compact U-Net for noisy-BEV -> semantic-BEV prediction."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = NUM_SEMANTIC_CLASSES,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        bottleneck_channels = base_channels * 16

        self.stem = ConvBlock(in_channels, c1)
        self.down1 = DownBlock(c1, c2)
        self.down2 = DownBlock(c2, c3)
        self.down3 = DownBlock(c3, c4)
        self.bottleneck = DownBlock(c4, bottleneck_channels)

        self.up1 = UpBlock(bottleneck_channels, c4, c4)
        self.up2 = UpBlock(c4, c3, c3)
        self.up3 = UpBlock(c3, c2, c2)
        self.up4 = UpBlock(c2, c1, c1)
        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("expected input shape [batch, channels, height, width]")

        skip1 = self.stem(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        skip4 = self.down3(skip3)
        bottleneck = self.bottleneck(skip4)

        x = self.up1(bottleneck, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)
        return self.head(x)
