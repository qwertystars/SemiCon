from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Pair:
    name: str
    low_resolution: Path
    ground_truth: Path


def load_array(path: Path) -> torch.Tensor:
    array = np.load(path, allow_pickle=False)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"Invalid grayscale array: {path} shape={array.shape}")
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).unsqueeze(0)


def discover_pairs(data_root: Path) -> list[Pair]:
    gt_root = data_root / "train" / "GT"
    lq_root = data_root / "train" / "NoisyLR"
    gt = {path.name: path for path in gt_root.glob("*.npy")}
    lq = {path.name: path for path in lq_root.glob("*.npy")}
    if not gt or gt.keys() != lq.keys():
        raise ValueError("GT and NoisyLR files are empty or do not pair exactly")
    return [Pair(name, lq[name], gt[name]) for name in sorted(gt)]


def split_pairs(pairs: list[Pair], validation_fraction: float) -> tuple[list[Pair], list[Pair]]:
    threshold = round(validation_fraction * 10_000)
    training, validation = [], []
    for pair in pairs:
        bucket = int(hashlib.sha1(pair.name.encode()).hexdigest()[:8], 16) % 10_000
        (validation if bucket < threshold else training).append(pair)
    if not training or not validation:
        raise ValueError("Training and validation splits must both be non-empty")
    return training, validation


def random_noise(
    value: torch.Tensor,
    gaussian: tuple[float, float],
    speckle: tuple[float, float],
    hard_ood: bool,
) -> torch.Tensor:
    gaussian_sigma = random.uniform(*gaussian)
    speckle_sigma = random.uniform(*speckle)
    operations = ["gaussian", "speckle", "downsample"]
    random.shuffle(operations)
    output = value.unsqueeze(0)
    for operation in operations:
        if operation == "gaussian":
            output = output + torch.randn_like(output) * gaussian_sigma
        elif operation == "speckle":
            output = output + output * torch.randn_like(output) * speckle_sigma
        else:
            mode = random.choice(("bicubic", "bilinear", "area")) if hard_ood else random.choices(
                ("area", "bicubic", "bilinear"), weights=(0.8, 0.1, 0.1), k=1
            )[0]
            options = {"scale_factor": 0.5, "mode": mode}
            if mode != "area":
                options.update(align_corners=False, antialias=True)
            output = F.interpolate(output, **options)
    return output.squeeze(0)


class RestorationDataset(Dataset):
    def __init__(self, pairs: list[Pair], training: bool, synthetic_probability: float = 0.35) -> None:
        self.pairs = pairs
        self.training = training
        self.synthetic_probability = synthetic_probability

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        pair = self.pairs[index]
        gt = load_array(pair.ground_truth)
        lq = load_array(pair.low_resolution)
        if gt.shape[-2:] != (lq.shape[-2] * 2, lq.shape[-1] * 2):
            raise ValueError(f"Ground truth is not 2x for {pair.name}")
        if self.training and random.random() < self.synthetic_probability:
            if random.random() < 0.3:
                lq = random_noise(gt, (0.03, 0.12), (0.08, 0.30), hard_ood=True)
            else:
                lq = random_noise(gt, (0.015, 0.060), (0.08, 0.24), hard_ood=False)
        if self.training:
            if random.random() < 0.5:
                lq, gt = lq.flip(-1), gt.flip(-1)
            if random.random() < 0.5:
                lq, gt = lq.flip(-2), gt.flip(-2)
            rotations = random.randrange(4)
            lq, gt = torch.rot90(lq, rotations, (-2, -1)), torch.rot90(gt, rotations, (-2, -1))
        return {"lq": lq.contiguous(), "gt": gt.contiguous(), "name": pair.name}
