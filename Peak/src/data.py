from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .io_utils import SUPPORTED_SUFFIXES, list_images, load_image


@dataclass(frozen=True)
class PairRecord:
    key: str
    lq_path: Path
    gt_path: Path


def _stem_key(path: Path, root: Path) -> str:
    return (path.relative_to(root).parent / path.stem).as_posix()


def discover_pairs(gt_dir: str | Path, lq_dir: str | Path) -> list[PairRecord]:
    gt_root = Path(gt_dir)
    lq_root = Path(lq_dir)
    gt_index = {_stem_key(path, gt_root): path for path in list_images(gt_root)}
    lq_index = {_stem_key(path, lq_root): path for path in list_images(lq_root)}
    common = sorted(set(gt_index) & set(lq_index))
    missing_gt = sorted(set(lq_index) - set(gt_index))
    missing_lq = sorted(set(gt_index) - set(lq_index))
    if missing_gt or missing_lq:
        details = f"missing GT={len(missing_gt)}, missing NoisyLR={len(missing_lq)}"
        raise ValueError(f"Pair discovery failed: {details}.")
    return [PairRecord(key, lq_index[key], gt_index[key]) for key in common]


def deterministic_split(
    records: list[PairRecord],
    val_fraction: float,
    val_source: str | None = None,
) -> tuple[list[PairRecord], list[PairRecord]]:
    if val_source:
        validation = [item for item in records if val_source in Path(item.key).parts]
        training = [item for item in records if item not in validation]
    else:
        threshold = int(val_fraction * 10000)
        validation = []
        training = []
        for item in records:
            bucket = int(hashlib.sha1(item.key.encode("utf-8")).hexdigest()[:8], 16) % 10000
            (validation if bucket < threshold else training).append(item)
    if not training or not validation:
        raise ValueError("The split must contain at least one training pair and one validation pair.")
    return training, validation


def _tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).unsqueeze(0)


def _paired_crop(lq: torch.Tensor, gt: torch.Tensor, hr_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    lr_size = hr_size // 2
    if gt.shape[-2:] != (lq.shape[-2] * 2, lq.shape[-1] * 2):
        raise ValueError(f"GT shape {tuple(gt.shape)} is not 2x LQ shape {tuple(lq.shape)}.")
    if lq.shape[-2] < lr_size or lq.shape[-1] < lr_size:
        return lq, gt
    top = random.randint(0, lq.shape[-2] - lr_size)
    left = random.randint(0, lq.shape[-1] - lr_size)
    return (
        lq[:, top : top + lr_size, left : left + lr_size],
        gt[:, top * 2 : (top + lr_size) * 2, left * 2 : (left + lr_size) * 2],
    )


def _augment(lq: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if random.random() < 0.5:
        lq, gt = lq.flip(-1), gt.flip(-1)
    if random.random() < 0.5:
        lq, gt = lq.flip(-2), gt.flip(-2)
    turns = random.randrange(4)
    return torch.rot90(lq, turns, (-2, -1)), torch.rot90(gt, turns, (-2, -1))


def synthesize_degradation(gt: torch.Tensor, config: dict) -> torch.Tensor:
    gaussian_range = config.get("gaussian_sigma", [0.0, 0.06])
    speckle_range = config.get("speckle_sigma", [0.0, 0.16])
    modes = config.get("downsample_modes", ["bicubic", "bilinear", "area"])
    gaussian_sigma = random.uniform(*map(float, gaussian_range))
    speckle_sigma = random.uniform(*map(float, speckle_range))
    operations = ["gaussian", "speckle", "downsample"]
    random.shuffle(operations)
    value = gt.unsqueeze(0)
    for operation in operations:
        if operation == "gaussian":
            value = value + torch.randn_like(value) * gaussian_sigma
        elif operation == "speckle":
            value = value + value * torch.randn_like(value) * speckle_sigma
        else:
            mode = random.choice(modes)
            kwargs = {"scale_factor": 0.5, "mode": mode}
            if mode in {"bilinear", "bicubic"}:
                kwargs.update({"align_corners": False, "antialias": True})
            value = F.interpolate(value, **kwargs)
    return value.squeeze(0)


class PairedRestorationDataset(Dataset):
    def __init__(
        self,
        records: list[PairRecord],
        hr_patch_size: int,
        training: bool,
        synthetic_probability: float = 0.0,
        degradation_config: dict | None = None,
    ) -> None:
        self.records = records
        self.hr_patch_size = int(hr_patch_size)
        self.training = training
        self.synthetic_probability = float(synthetic_probability)
        self.degradation_config = degradation_config or {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        lq = _tensor(load_image(record.lq_path))
        gt = _tensor(load_image(record.gt_path))
        if self.training:
            lq, gt = _paired_crop(lq, gt, self.hr_patch_size)
            if random.random() < self.synthetic_probability:
                lq = synthesize_degradation(gt, self.degradation_config)
            lq, gt = _augment(lq, gt)
        elif gt.shape[-2:] != (lq.shape[-2] * 2, lq.shape[-1] * 2):
            raise ValueError(f"GT is not 2x for pair: {record.key}")
        return {"lq": lq.contiguous(), "gt": gt.contiguous(), "key": record.key}
