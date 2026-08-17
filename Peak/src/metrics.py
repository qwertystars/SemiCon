from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(prediction, target, reduction="none").flatten(1).mean(1)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def _gaussian_window(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coordinates = torch.arange(size, device=device, dtype=dtype) - size // 2
    kernel = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    return (kernel[:, None] * kernel[None, :]).view(1, 1, size, size)


def ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    window = _gaussian_window(window_size, sigma, prediction.device, prediction.dtype)
    padding = window_size // 2
    mu_x = F.conv2d(prediction, window, padding=padding)
    mu_y = F.conv2d(target, window, padding=padding)
    mu_x2 = mu_x.square()
    mu_y2 = mu_y.square()
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(prediction.square(), window, padding=padding) - mu_x2
    sigma_y2 = F.conv2d(target.square(), window, padding=padding) - mu_y2
    sigma_xy = F.conv2d(prediction * target, window, padding=padding) - mu_xy
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return score.flatten(1).mean(1)
