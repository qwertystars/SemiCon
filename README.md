# Peak — KLA Blind Compound Image Restoration

Peak restores grayscale `.npy` inputs affected by unknown-order 2× downsampling, Gaussian noise, and multiplicative speckle noise. The submission preserves out-of-range measurements until the final output step.

## Submission structure

```text
Peak/
├── run.py
├── requirements.txt
├── README.md
└── models/
    ├── peak_range_conditioned_naf_sr_x2.pt
    └── supporting model files
```

## Method

The Range-Aware Degradation-Conditioned NAFNet-SR model uses four input views: raw intensity, clipped intensity, negative overflow, and positive overflow. A lightweight degradation encoder produces a latent condition and estimated Gaussian/speckle scales. Conditional NAF blocks use the latent condition to adapt restoration behavior, while a bicubic skip and PixelShuffle residual head generate the 2× output.

Training uses official pairs plus arbitrary-order synthetic Gaussian, speckle, and downsampling operations. The objective combines Charbonnier, SSIM, gradient, frequency, heteroscedastic forward-consistency, and valid-range penalties. LPIPS is excluded from the runtime dependency path so inference never downloads external weights.

Analysis of 800 official pairs found BOX/area to be the lowest-error candidate downsampler for 773 images, with fitted noise scales near `sigma_g=0.0348` and `sigma_s=0.1642`. The in-distribution synthetic branch is centered on those measurements; a separate hard-OOD branch retains wider strengths and kernels.

## Install

Use Python 3.10+ and an NVIDIA CUDA-compatible GPU:

```bash
python -m pip install -r requirements.txt
```

No internet connection, API key, additional download, interaction, or manual configuration is required at runtime.

## Required execution interface

```bash
python run.py <input-dir> <output-dir>
```

The script recursively reads every `.npy` input, creates the output directory, preserves relative filenames, and writes one finite `float32` grayscale array per input. Outputs have shape `(2H, 2W)` and values clipped to `[0,1]`.

## Development training

Training utilities live inside `models/` to preserve the required top-level submission structure. With the dataset located at `../datasets/KLA`:

```bash
python models/analyze_data.py --data-dir ../datasets/KLA
python models/smoke_test.py
python models/train.py
python models/evaluate.py
```

The best validation checkpoint is written directly to `models/peak_range_conditioned_naf_sr_x2.pt`. Training artifacts are written outside the submission folder under `../artifacts/kla_training`.

After inference, the exact file, shape, range, and finite-value contract can be checked with:

```bash
python models/validate_outputs.py <input-dir> <output-dir>
```
