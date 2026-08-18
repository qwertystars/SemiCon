from __future__ import annotations

import argparse
import itertools
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


RESAMPLERS = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
    "box": Image.Resampling.BOX,
}


def resize(array: np.ndarray, method: Image.Resampling) -> np.ndarray:
    height, width = array.shape
    return np.asarray(
        Image.fromarray(array.astype(np.float32), mode="F").resize((width // 2, height // 2), method),
        dtype=np.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reverse-engineer KLA paired degradations.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=800)
    args = parser.parse_args()
    gt_root, lq_root = args.data_dir / "train" / "GT", args.data_dir / "train" / "NoisyLR"
    names = sorted(path.name for path in gt_root.glob("*.npy"))[: args.limit]
    if not names:
        raise FileNotFoundError(f"No training arrays found below {args.data_dir}")
    errors = Counter()
    winners = Counter()
    bin_edges = np.linspace(0.0, 1.0, 11)
    residual_sum = np.zeros(10, dtype=np.float64)
    residual_count = np.zeros(10, dtype=np.int64)
    lq_min, lq_max = np.inf, -np.inf
    below = above = pixels = 0
    regression_count = 0
    regression_x = regression_y = regression_xx = regression_xy = 0.0
    for name in names:
        gt = np.load(gt_root / name, allow_pickle=False).astype(np.float32, copy=False)
        lq = np.load(lq_root / name, allow_pickle=False).astype(np.float32, copy=False)
        candidates = {label: resize(gt, method) for label, method in RESAMPLERS.items()}
        sample_errors = {label: float(np.mean((candidate - lq) ** 2)) for label, candidate in candidates.items()}
        for label, error in sample_errors.items():
            errors[label] += error
        winner = min(sample_errors, key=sample_errors.get)
        winners[winner] += 1
        baseline = candidates[winner]
        residual_square = (lq - baseline) ** 2
        intensity_square = np.clip(baseline, 0.0, 1.0) ** 2
        regression_count += intensity_square.size
        regression_x += float(intensity_square.sum(dtype=np.float64))
        regression_y += float(residual_square.sum(dtype=np.float64))
        regression_xx += float((intensity_square**2).sum(dtype=np.float64))
        regression_xy += float((intensity_square * residual_square).sum(dtype=np.float64))
        bins = np.clip(np.digitize(baseline, bin_edges) - 1, 0, 9)
        for index in range(10):
            selected = residual_square[bins == index]
            residual_sum[index] += float(selected.sum())
            residual_count[index] += selected.size
        lq_min, lq_max = min(lq_min, float(lq.min())), max(lq_max, float(lq.max()))
        below += int(np.count_nonzero(lq < 0.0))
        above += int(np.count_nonzero(lq > 1.0))
        pixels += lq.size
    print(f"samples={len(names)} lq_range=[{lq_min:.6f}, {lq_max:.6f}]")
    print(f"out_of_range: below_zero={below/pixels:.4%} above_one={above/pixels:.4%}")
    print("mean candidate MSE and per-image wins:")
    for label in RESAMPLERS:
        print(f"  {label:8s} mse={errors[label]/len(names):.8f} wins={winners[label]}")
    denominator = regression_xx - regression_x * regression_x / regression_count
    slope = (regression_xy - regression_x * regression_y / regression_count) / denominator
    intercept = (regression_y - slope * regression_x) / regression_count
    print(
        f"heteroscedastic fit variance ~= sigma_g^2 + sigma_s^2 * I^2: "
        f"sigma_g={np.sqrt(max(intercept, 0.0)):.6f} sigma_s={np.sqrt(max(slope, 0.0)):.6f}"
    )
    print("residual variance proxy by baseline-intensity bin:")
    for index, (lower, upper) in enumerate(itertools.pairwise(bin_edges)):
        value = residual_sum[index] / max(residual_count[index], 1)
        print(f"  [{lower:.1f},{upper:.1f}): {value:.8f} n={residual_count[index]}")


if __name__ == "__main__":
    main()
