from __future__ import annotations

import json
from pathlib import Path

import torch

from data import load_array
from losses import CompoundRestorationLoss
from model import build_model


MODEL_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = MODEL_DIR.parent.parent


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    config = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    data = REPOSITORY_DIR / "datasets" / "KLA" / "train"
    batch_size = int(config["batch_size"])
    lq = load_array(data / "NoisyLR" / "000000.npy").unsqueeze(0).repeat(batch_size, 1, 1, 1).cuda()
    gt = load_array(data / "GT" / "000000.npy").unsqueeze(0).repeat(batch_size, 1, 1, 1).cuda()
    model = build_model(config["model"]).cuda()
    loss_function = CompoundRestorationLoss().cuda()
    prediction, auxiliary = model(lq, return_aux=True)
    loss, components = loss_function(prediction, gt, lq, auxiliary["noise_parameters"])
    loss.backward()
    if prediction.shape != gt.shape or not torch.isfinite(prediction).all() or not torch.isfinite(loss):
        raise RuntimeError("Model smoke test failed")
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2
    print(
        f"input={tuple(lq.shape)} output={tuple(prediction.shape)} loss={float(loss):.6f} "
        f"peak_vram_mb={peak_memory:.1f} components={{{', '.join(f'{k}: {float(v):.6f}' for k, v in components.items())}}}"
    )


if __name__ == "__main__":
    main()
