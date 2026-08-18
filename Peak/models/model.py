from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        mean = value.mean(1, keepdim=True)
        variance = (value - mean).square().mean(1, keepdim=True)
        return (value - mean) * torch.rsqrt(variance + self.eps) * self.weight + self.bias


def range_features(value: torch.Tensor) -> torch.Tensor:
    clipped = value.clamp(0.0, 1.0)
    negative_overflow = (-value).clamp_min(0.0)
    positive_overflow = (value - 1.0).clamp_min(0.0)
    return torch.cat((value, clipped, negative_overflow, positive_overflow), dim=1)


class DegradationEncoder(nn.Module):
    def __init__(self, width: int, embedding_dim: int) -> None:
        super().__init__()
        hidden = max(width // 2, 16)
        self.encoder = nn.Sequential(
            nn.Conv2d(4, hidden, 3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, width, 3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width, embedding_dim),
            nn.SiLU(inplace=True),
        )
        self.noise_head = nn.Linear(embedding_dim, 2)
        nn.init.zeros_(self.noise_head.weight)
        nn.init.constant_(self.noise_head.bias, -2.2)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(features)
        normalized = torch.sigmoid(self.noise_head(embedding))
        maximum = normalized.new_tensor((0.15, 0.35))
        return embedding, (normalized * maximum).clamp_min(1e-4)


class ConditionalNAFBlock(nn.Module):
    def __init__(self, channels: int, embedding_dim: int, dilation: int) -> None:
        super().__init__()
        hidden = channels * 2
        self.norm1 = LayerNorm2d(channels)
        self.expand1 = nn.Conv2d(channels, hidden, 1)
        self.depthwise = nn.Conv2d(
            hidden, hidden, 3, padding=dilation, dilation=dilation, groups=hidden
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1)
        )
        self.project1 = nn.Conv2d(channels, channels, 1)
        self.norm2 = LayerNorm2d(channels)
        self.expand2 = nn.Conv2d(channels, hidden, 1)
        self.project2 = nn.Conv2d(channels, channels, 1)
        self.condition = nn.Linear(embedding_dim, channels * 4)
        nn.init.zeros_(self.condition.weight)
        nn.init.zeros_(self.condition.bias)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    @staticmethod
    def gate(value: torch.Tensor) -> torch.Tensor:
        left, right = value.chunk(2, dim=1)
        return left * right

    def forward(self, value: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        scale1, shift1, scale2, shift2 = self.condition(embedding).chunk(4, dim=1)
        scale1, shift1, scale2, shift2 = (
            item[:, :, None, None] for item in (scale1, shift1, scale2, shift2)
        )
        branch = self.norm1(value) * (1.0 + scale1) + shift1
        branch = self.gate(self.depthwise(self.expand1(branch)))
        branch = branch * self.channel_attention(branch)
        value = value + self.project1(branch) * self.beta
        branch = self.norm2(value) * (1.0 + scale2) + shift2
        branch = self.gate(self.expand2(branch))
        return value + self.project2(branch) * self.gamma


class RangeConditionedNAFSR(nn.Module):
    def __init__(
        self,
        scale: int = 2,
        width: int = 32,
        blocks: int = 8,
        embedding_dim: int = 64,
        dilations: Sequence[int] = (1, 2, 3),
    ) -> None:
        super().__init__()
        if scale != 2:
            raise ValueError("Only 2x restoration is supported")
        self.scale = scale
        self.degradation_encoder = DegradationEncoder(width, embedding_dim)
        self.stem = nn.Conv2d(4, width, 3, padding=1)
        self.blocks = nn.ModuleList(
            ConditionalNAFBlock(width, embedding_dim, int(dilations[index % len(dilations)]))
            for index in range(blocks)
        )
        self.body_end = nn.Conv2d(width, width, 3, padding=1)
        self.reconstruction = nn.Sequential(
            nn.Conv2d(width, width * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(width, 1, 3, padding=1),
        )
        nn.init.zeros_(self.reconstruction[-1].weight)
        nn.init.zeros_(self.reconstruction[-1].bias)

    def forward(self, value: torch.Tensor, return_aux: bool = False):
        features = range_features(value)
        embedding, noise_parameters = self.degradation_encoder(features)
        shallow = self.stem(features)
        restored = shallow
        for block in self.blocks:
            restored = block(restored, embedding)
        restored = self.body_end(restored) + shallow
        residual = self.reconstruction(restored)
        baseline = F.interpolate(value, scale_factor=self.scale, mode="bicubic", align_corners=False)
        output = baseline + residual
        if return_aux:
            return output, {"noise_parameters": noise_parameters, "embedding": embedding}
        return output


def build_model(config: dict) -> RangeConditionedNAFSR:
    return RangeConditionedNAFSR(
        scale=int(config.get("scale", 2)),
        width=int(config.get("width", 32)),
        blocks=int(config.get("blocks", 8)),
        embedding_dim=int(config.get("embedding_dim", 64)),
        dilations=config.get("dilations", [1, 2, 3]),
    )
