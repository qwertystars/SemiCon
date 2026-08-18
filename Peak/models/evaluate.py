from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import RestorationDataset, discover_pairs, split_pairs
from losses import ssim
from model import build_model


MODEL_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = MODEL_DIR.parent.parent


def metrics(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    error = (prediction - target).square().flatten(1).mean(1).clamp_min(1e-12)
    return -10.0 * torch.log10(error), ssim(prediction, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare bicubic and a trained Peak checkpoint.")
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_DIR / "datasets" / "KLA")
    parser.add_argument("--checkpoint", type=Path, default=MODEL_DIR / "peak_range_conditioned_naf_sr_x2.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    _, validation_pairs = split_pairs(discover_pairs(args.data_dir), 0.1)
    loader = DataLoader(
        RestorationDataset(validation_pairs, training=False), batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().cuda().to(memory_format=torch.channels_last)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    totals = {"bicubic_psnr": 0.0, "bicubic_ssim": 0.0, "model_psnr": 0.0, "model_ssim": 0.0}
    count = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            lq = batch["lq"].cuda(non_blocking=True)
            gt = batch["gt"].cuda(non_blocking=True)
            bicubic = F.interpolate(lq, scale_factor=2, mode="bicubic", align_corners=False).clamp(0.0, 1.0)
            with torch.autocast("cuda", dtype=amp_dtype):
                restored = model(lq)
            restored = restored.float().clamp(0.0, 1.0)
            bicubic_psnr, bicubic_ssim = metrics(bicubic.float(), gt)
            model_psnr, model_ssim = metrics(restored, gt)
            totals["bicubic_psnr"] += float(bicubic_psnr.sum())
            totals["bicubic_ssim"] += float(bicubic_ssim.sum())
            totals["model_psnr"] += float(model_psnr.sum())
            totals["model_ssim"] += float(model_ssim.sum())
            count += lq.shape[0]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    print(f"samples={count} elapsed_seconds={elapsed:.3f}")
    print(f"bicubic psnr={totals['bicubic_psnr']/count:.4f} ssim={totals['bicubic_ssim']/count:.6f}")
    print(f"model    psnr={totals['model_psnr']/count:.4f} ssim={totals['model_ssim']/count:.6f}")


if __name__ == "__main__":
    main()
