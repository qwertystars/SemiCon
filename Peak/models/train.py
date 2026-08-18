from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data import RestorationDataset, discover_pairs, split_pairs
from losses import CompoundRestorationLoss, ssim
from model import build_model


MODEL_DIR = Path(__file__).resolve().parent
SUBMISSION_DIR = MODEL_DIR.parent
REPOSITORY_DIR = SUBMISSION_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Peak's range-aware conditioned NAFNet-SR.")
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_DIR / "datasets" / "KLA")
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_DIR / "artifacts" / "kla_training")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--fresh", action="store_true", help="Ignore an existing last.pt checkpoint")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = (prediction - target).square().flatten(1).mean(1).clamp_min(1e-12)
    return -10.0 * torch.log10(error)


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device, amp_dtype: torch.dtype) -> dict:
    model.eval()
    psnr_sum = ssim_sum = 0.0
    samples = 0
    with torch.inference_mode():
        for batch in loader:
            lq = batch["lq"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                prediction = model(lq)
            prediction = prediction.float().clamp(0.0, 1.0)
            psnr_sum += float(psnr(prediction, gt).sum())
            ssim_sum += float(ssim(prediction, gt).sum())
            samples += lq.shape[0]
    return {"psnr": psnr_sum / samples, "ssim": ssim_sum / samples}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for training")
    config = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    epochs = args.epochs or int(config["epochs"])
    seed_everything(int(config["seed"]))
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    training_pairs, validation_pairs = split_pairs(discover_pairs(args.data_dir), float(config["validation_fraction"]))
    training_data = RestorationDataset(training_pairs, training=True, synthetic_probability=float(config["synthetic_probability"]))
    validation_data = RestorationDataset(validation_pairs, training=False)
    loader_options = {
        "batch_size": int(config["batch_size"]), "num_workers": int(config["workers"]),
        "pin_memory": True, "persistent_workers": int(config["workers"]) > 0,
    }
    training_loader = DataLoader(training_data, shuffle=True, drop_last=True, **loader_options)
    validation_loader = DataLoader(validation_data, shuffle=False, drop_last=False, **loader_options)
    model = build_model(config["model"]).to(device, memory_format=torch.channels_last)
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    loss_function = CompoundRestorationLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=float(config["minimum_learning_rate"]))
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    accumulation = int(config["gradient_accumulation"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_psnr = -math.inf
    first_epoch = 1
    resume_path = args.output_dir / "last.pt"
    if resume_path.is_file() and not args.fresh:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint.get("training_model", checkpoint["model"]))
        ema_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        first_epoch = int(checkpoint["epoch"]) + 1
        best_psnr = float(checkpoint.get("best_psnr", checkpoint.get("metrics", {}).get("psnr", -math.inf)))
        print(f"resumed={resume_path} next_epoch={first_epoch} best_psnr={best_psnr:.4f}", flush=True)
    print(f"gpu={torch.cuda.get_device_name(0)} train={len(training_data)} validation={len(validation_data)} parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    for epoch in range(first_epoch, epochs + 1):
        started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for step, batch in enumerate(training_loader, 1):
            lq = batch["lq"].to(device, non_blocking=True, memory_format=torch.channels_last)
            gt = batch["gt"].to(device, non_blocking=True, memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=amp_dtype):
                prediction, auxiliary = model(lq, return_aux=True)
                loss, _ = loss_function(prediction, gt, lq, auxiliary["noise_parameters"])
            scaler.scale(loss / accumulation).backward()
            if step % accumulation == 0 or step == len(training_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    decay = float(config["ema_decay"])
                    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters(), strict=True):
                        ema_parameter.lerp_(parameter, 1.0 - decay)
            loss_sum += float(loss.detach())
        scheduler.step()
        metrics = validate(ema_model, validation_loader, device, amp_dtype)
        metrics.update(loss=loss_sum / len(training_loader), learning_rate=optimizer.param_groups[0]["lr"])
        training_checkpoint = {
            "model": ema_model.state_dict(),
            "training_model": model.state_dict(),
            "model_config": config["model"],
            "epoch": epoch,
            "metrics": metrics,
            "best_psnr": max(best_psnr, metrics["psnr"]),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        }
        save_checkpoint(args.output_dir / "last.pt", training_checkpoint)
        if metrics["psnr"] > best_psnr:
            best_psnr = metrics["psnr"]
            inference_checkpoint = {
                "model": ema_model.state_dict(), "model_config": config["model"],
                "epoch": epoch, "metrics": metrics,
            }
            save_checkpoint(MODEL_DIR / "peak_range_conditioned_naf_sr_x2.pt", inference_checkpoint)
            save_checkpoint(args.output_dir / "best.pt", inference_checkpoint)
        print(f"epoch={epoch:03d}/{epochs} loss={metrics['loss']:.6f} psnr={metrics['psnr']:.4f} ssim={metrics['ssim']:.6f} seconds={time.perf_counter() - started:.1f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted; the most recent completed checkpoint is preserved.", file=sys.stderr)
        raise
