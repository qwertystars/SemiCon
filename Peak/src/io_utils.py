from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import tifffile


SUPPORTED_SUFFIXES = {".npy", ".npz", ".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg"}


def list_images(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root_path}")
    paths = [path for path in root_path.rglob("*") if path.suffix.lower() in SUPPORTED_SUFFIXES]
    paths.sort(key=lambda path: path.as_posix())
    if not paths:
        raise FileNotFoundError(f"No supported images exist in: {root_path}")
    return paths


def _to_grayscale(array: np.ndarray, path: Path) -> np.ndarray:
    array = np.asarray(array)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Expected one grayscale image at {path}, got shape {array.shape}.")
    return array


def load_image(path: str | Path) -> np.ndarray:
    image_path = Path(path)
    suffix = image_path.suffix.lower()
    if suffix == ".npy":
        array = np.load(image_path, allow_pickle=False)
    elif suffix == ".npz":
        archive = np.load(image_path, allow_pickle=False)
        if not archive.files:
            raise ValueError(f"NPZ file contains no arrays: {image_path}")
        array = archive[archive.files[0]]
    elif suffix in {".tif", ".tiff"}:
        array = tifffile.imread(image_path)
    else:
        array = iio.imread(image_path)

    array = _to_grayscale(array, image_path)
    original_dtype = array.dtype
    array = array.astype(np.float32, copy=False)
    if np.issubdtype(original_dtype, np.integer):
        array /= float(np.iinfo(original_dtype).max)
    if not np.isfinite(array).all():
        raise ValueError(f"Image contains NaN or infinity: {image_path}")
    return np.ascontiguousarray(array)


def save_image(path: str | Path, array: np.ndarray) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(array, dtype=np.float32)
    array = np.clip(array, 0.0, 1.0)
    suffix = output_path.suffix.lower()
    if suffix == ".npy":
        np.save(output_path, array)
    elif suffix == ".npz":
        np.savez_compressed(output_path, restored=array)
    elif suffix in {".tif", ".tiff"}:
        tifffile.imwrite(output_path, array, photometric="minisblack")
    elif suffix in {".jpg", ".jpeg"}:
        iio.imwrite(output_path, np.round(array * 255.0).astype(np.uint8), quality=100)
    else:
        iio.imwrite(output_path, np.round(array * 65535.0).astype(np.uint16))


def relative_output_path(input_path: Path, input_root: Path, output_root: Path) -> Path:
    return output_root / input_path.relative_to(input_root)
