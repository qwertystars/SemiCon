from __future__ import annotations

from pathlib import Path

import torch

from .model import build_model


def load_model(weights: str | Path, device: torch.device):
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Checkpoint must contain 'model' and 'model_config'.")
    model_config = checkpoint.get("model_config", {})
    model = build_model(model_config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    return model, model_config
