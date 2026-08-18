from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the KLA .npy submission contract.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    inputs = sorted(path.relative_to(args.input_dir) for path in args.input_dir.rglob("*.npy"))
    outputs = sorted(path.relative_to(args.output_dir) for path in args.output_dir.rglob("*.npy"))
    if inputs != outputs:
        missing = sorted(set(inputs) - set(outputs))
        extra = sorted(set(outputs) - set(inputs))
        raise ValueError(f"Filename mismatch: missing={missing[:5]} extra={extra[:5]}")
    for relative in inputs:
        source = np.load(args.input_dir / relative, allow_pickle=False)
        restored = np.load(args.output_dir / relative, allow_pickle=False)
        if restored.ndim == 3 and restored.shape[-1] == 1:
            restored = restored[..., 0]
        if restored.ndim != 2 or restored.shape != (source.shape[0] * 2, source.shape[1] * 2):
            raise ValueError(f"Wrong output shape for {relative}: {restored.shape}")
        if not np.isfinite(restored).all() or float(restored.min()) < 0.0 or float(restored.max()) > 1.0:
            raise ValueError(f"Invalid values for {relative}")
    print(f"Validated {len(inputs)} matching, finite, [0,1], 2x grayscale outputs.")


if __name__ == "__main__":
    main()
