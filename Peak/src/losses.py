from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .metrics import ssim


class CompositeRestorationLoss(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.charbonnier_weight = float(config.get("charbonnier", 0.45))
        self.mse_weight = float(config.get("mse", 0.25))
        self.ssim_weight = float(config.get("ssim", 0.20))
        self.edge_weight = float(config.get("edge", 0.10))
        self.epsilon = float(config.get("epsilon", 1e-3))
        laplacian = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        self.register_buffer("laplacian", laplacian.view(1, 1, 3, 3))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        difference = prediction - target
        charbonnier = torch.sqrt(difference.square() + self.epsilon**2).mean()
        mse = difference.square().mean()
        structure = 1.0 - ssim(prediction.float(), target.float()).mean()
        edge_prediction = F.conv2d(prediction, self.laplacian.to(prediction), padding=1)
        edge_target = F.conv2d(target, self.laplacian.to(target), padding=1)
        edge = F.l1_loss(edge_prediction, edge_target)
        total = (
            self.charbonnier_weight * charbonnier
            + self.mse_weight * mse
            + self.ssim_weight * structure
            + self.edge_weight * edge
        )
        components = {
            "total": total.detach(),
            "charbonnier": charbonnier.detach(),
            "mse": mse.detach(),
            "ssim_loss": structure.detach(),
            "edge": edge.detach(),
        }
        return total, components
