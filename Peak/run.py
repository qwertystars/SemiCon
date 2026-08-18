from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from models.model import build_model


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "models" / "peak_range_conditioned_naf_sr_x2.pt"


def load_array(path: Path) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Expected grayscale (H,W) or (H,W,1), got {array.shape}: {path}")
    original_dtype = array.dtype
    array = np.asarray(array, dtype=np.float32)
    if np.issubdtype(original_dtype, np.integer):
        array /= float(np.iinfo(original_dtype).max)
    if not np.isfinite(array).all():
        raise ValueError(f"Input contains NaN or Inf: {path}")
    return np.ascontiguousarray(array)


def main() -> None:
    parser = argparse.ArgumentParser(description="Peak KLA blind compound image restoration")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    input_root, output_root = args.input_dir.resolve(), args.output_dir.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    paths = sorted(path for path in input_root.rglob("*.npy") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No .npy inputs found in {input_root}")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Model checkpoint is missing: {CHECKPOINT}")
    if not torch.cuda.is_available():
        raise RuntimeError("An NVIDIA CUDA GPU is required")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().cuda().to(memory_format=torch.channels_last)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    buffers: dict[tuple[int, int], list[tuple[Path, np.ndarray]]] = defaultdict(list)

    def restore(items: list[tuple[Path, np.ndarray]]) -> None:
        tensor = torch.from_numpy(np.stack([array for _, array in items])[:, None]).cuda(
            non_blocking=True, memory_format=torch.channels_last
        )
        with torch.autocast("cuda", dtype=amp_dtype):
            outputs = model(tensor).float().cpu().numpy()[:, 0]
        outputs = np.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=0.0)
        outputs = np.clip(outputs, 0.0, 1.0).astype(np.float32, copy=False)
        for (source, _), output in zip(items, outputs, strict=True):
            destination = output_root / source.relative_to(input_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.save(destination, output)

    with torch.inference_mode():
        for path in paths:
            array = load_array(path)
            bucket = buffers[array.shape]
            bucket.append((path, array))
            if len(bucket) == args.batch_size:
                restore(bucket)
                bucket.clear()
        for bucket in buffers.values():
            if bucket:
                restore(bucket)
    print(f"Restored {len(paths)} file(s) into {output_root}")


if __name__ == "__main__":
    main()
