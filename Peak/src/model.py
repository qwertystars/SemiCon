from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + self.eps) * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = x.chunk(2, dim=1)
        return left * right


class NAFResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.norm1 = LayerNorm2d(channels)
        self.expand1 = nn.Conv2d(channels, hidden, 1)
        self.depthwise = nn.Conv2d(
            hidden,
            hidden,
            3,
            padding=dilation,
            dilation=dilation,
            groups=hidden,
        )
        self.gate1 = SimpleGate()
        gated = hidden // 2
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(gated, gated, 1),
        )
        self.project1 = nn.Conv2d(gated, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.norm2 = LayerNorm2d(channels)
        self.expand2 = nn.Conv2d(channels, hidden, 1)
        self.gate2 = SimpleGate()
        self.project2 = nn.Conv2d(gated, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.expand1(self.norm1(x))
        y = self.gate1(self.depthwise(y))
        y = y * self.channel_attention(y)
        x = x + self.project1(y) * self.beta

        y = self.gate2(self.expand2(self.norm2(x)))
        return x + self.project2(y) * self.gamma


class RangeAwareNAFSR(nn.Module):
    """A compact grayscale restoration network with a fixed 2x output scale."""

    def __init__(
        self,
        scale: int = 2,
        width: int = 48,
        blocks: int = 12,
        dilations: Iterable[int] = (1, 2, 3),
    ) -> None:
        super().__init__()
        if scale != 2:
            raise ValueError("This submission model supports only scale=2.")
        dilation_list = tuple(int(value) for value in dilations)
        if not dilation_list:
            raise ValueError("dilations must contain at least one value.")

        self.scale = scale
        self.stem = nn.Conv2d(3, width, 3, padding=1)
        self.body = nn.Sequential(
            *[
                NAFResidualBlock(width, dilation_list[index % len(dilation_list)])
                for index in range(blocks)
            ]
        )
        self.body_end = nn.Conv2d(width, width, 3, padding=1)
        self.reconstruction = nn.Sequential(
            nn.Conv2d(width, width * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(width, 1, 3, padding=1),
        )
        nn.init.zeros_(self.reconstruction[-1].weight)
        nn.init.zeros_(self.reconstruction[-1].bias)

    @staticmethod
    def range_features(x: torch.Tensor) -> torch.Tensor:
        clipped = x.clamp(0.0, 1.0)
        overflow = x - clipped
        return torch.cat((x, clipped, overflow), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        baseline = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        shallow = self.stem(self.range_features(x))
        deep = self.body_end(self.body(shallow)) + shallow
        return baseline + self.reconstruction(deep)


def build_model(model_config: dict) -> RangeAwareNAFSR:
    return RangeAwareNAFSR(
        scale=int(model_config.get("scale", 2)),
        width=int(model_config.get("width", 48)),
        blocks=int(model_config.get("blocks", 12)),
        dilations=model_config.get("dilations", [1, 2, 3]),
    )
