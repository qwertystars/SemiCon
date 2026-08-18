from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    window_size, sigma = 11, 1.5
    coordinates = torch.arange(window_size, device=prediction.device, dtype=prediction.dtype) - 5
    kernel = torch.exp(-coordinates.square() / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    window = (kernel[:, None] * kernel[None, :]).view(1, 1, window_size, window_size)
    mean_x = F.conv2d(prediction, window, padding=5)
    mean_y = F.conv2d(target, window, padding=5)
    variance_x = F.conv2d(prediction.square(), window, padding=5) - mean_x.square()
    variance_y = F.conv2d(target.square(), window, padding=5) - mean_y.square()
    covariance = F.conv2d(prediction * target, window, padding=5) - mean_x * mean_y
    score = ((2 * mean_x * mean_y + 0.01**2) * (2 * covariance + 0.03**2)) / (
        (mean_x.square() + mean_y.square() + 0.01**2) * (variance_x + variance_y + 0.03**2)
    )
    return score.flatten(1).mean(1)


class CompoundRestorationLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        observed_lq: torch.Tensor,
        noise_parameters: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        difference = prediction - target
        charbonnier = torch.sqrt(difference.square() + 1e-6).mean()
        structure = 1.0 - ssim(prediction.float(), target.float()).mean()
        gradient = (
            F.l1_loss(prediction[..., 1:, :] - prediction[..., :-1, :], target[..., 1:, :] - target[..., :-1, :])
            + F.l1_loss(prediction[..., :, 1:] - prediction[..., :, :-1], target[..., :, 1:] - target[..., :, :-1])
        )
        frequency = F.l1_loss(
            torch.log1p(torch.fft.rfft2(prediction.float()).abs()),
            torch.log1p(torch.fft.rfft2(target.float()).abs()),
        )
        predicted_lq = F.interpolate(prediction.float(), size=observed_lq.shape[-2:], mode="area")
        observed_lq = observed_lq.float()
        noise_parameters = noise_parameters.float()
        gaussian = noise_parameters[:, 0, None, None, None]
        speckle = noise_parameters[:, 1, None, None, None]
        variance = gaussian.square() + (speckle * predicted_lq.clamp(0.0, 1.0)).square() + 1e-6
        residual_square = (observed_lq - predicted_lq).square()
        robust_residual = torch.log1p(residual_square / (3.0 * variance)).mean()
        residual_variance = residual_square.flatten(1).mean(1)
        predicted_variance = variance.flatten(1).mean(1)
        variance_calibration = F.l1_loss(predicted_variance, residual_variance)
        consistency = robust_residual + variance_calibration
        range_penalty = prediction.clamp_max(0.0).square().mean() + (prediction - 1.0).clamp_min(0.0).square().mean()
        total = (
            0.55 * charbonnier
            + 0.18 * structure
            + 0.10 * gradient
            + 0.08 * frequency
            + 0.03 * consistency
            + 0.06 * range_penalty
        )
        return total, {
            "charbonnier": charbonnier.detach(),
            "ssim_loss": structure.detach(),
            "gradient": gradient.detach(),
            "frequency": frequency.detach(),
            "consistency": consistency.detach(),
            "range": range_penalty.detach(),
        }
