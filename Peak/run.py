from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.checkpoint import load_model


MODEL_PATH = Path(__file__).resolve().parent / "models" / "kla_renaf_sr_x2.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore every .npy image in a directory.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def load_input(path: Path) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    array = np.asarray(array)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Expected grayscale shape (H, W) or (H, W, 1): {path} has {array.shape}")

    original_dtype = array.dtype
    array = array.astype(np.float32, copy=False)
    if np.issubdtype(original_dtype, np.integer):
        array /= float(np.iinfo(original_dtype).max)
    if not np.isfinite(array).all():
        raise ValueError(f"Input contains NaN or Inf: {path}")
    return np.ascontiguousarray(array)


def main() -> None:
    args = parse_args()
    input_root = args.input_dir.resolve()
    output_root = args.output_dir.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    output_root.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        (path for path in input_root.rglob("*") if path.is_file() and path.suffix.lower() == ".npy"),
        key=lambda path: path.as_posix(),
    )
    if not paths:
        raise FileNotFoundError(f"No .npy files found in: {input_root}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Required model checkpoint is missing: {MODEL_PATH}. "
            "Place the trained checkpoint in models/ before submission."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("An NVIDIA GPU with CUDA support is required.")

    device = torch.device("cuda")
    model, model_config = load_model(MODEL_PATH, device)
    model = model.to(memory_format=torch.channels_last)
    buffers: dict[tuple[int, int], list[tuple[Path, np.ndarray]]] = defaultdict(list)
    processed = 0

    def restore_batch(batch_items: list[tuple[Path, np.ndarray]]) -> int:
        batch = np.stack([array for _, array in batch_items])[:, None]
        tensor = torch.from_numpy(batch).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            restored = model(tensor)
        outputs = restored.float().cpu().numpy()[:, 0]
        outputs = np.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=0.0)
        outputs = np.clip(outputs, 0.0, 1.0).astype(np.float32, copy=False)

        for (input_path, _), output in zip(batch_items, outputs, strict=True):
            output_path = output_root / input_path.relative_to(input_root)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, output)
        return len(batch_items)

    with torch.inference_mode():
        for path in paths:
            array = load_input(path)
            buffer = buffers[array.shape]
            buffer.append((path, array))
            if len(buffer) >= args.batch_size:
                processed += restore_batch(buffer)
                buffer.clear()
        for buffer in buffers.values():
            if buffer:
                processed += restore_batch(buffer)

    scale = int(model_config.get("scale", 2))
    print(f"Restored {processed} file(s) at {scale}x resolution into {output_root}")


if __name__ == "__main__":
    main()
