# Peak — KLA Image Restoration Submission

This submission restores every grayscale `.npy` image found in the input directory and its subdirectories. Outputs retain the same relative path and filename, use `float32`, contain finite values in `[0, 1]`, and have twice the input height and width.

## Folder structure

```text
Peak/
├── run.py
├── requirements.txt
├── README.md
├── models/
│   └── kla_renaf_sr_x2.pt
└── src/
```

## Requirements

- Python 3.10 or newer
- NVIDIA GPU with a CUDA-compatible driver
- CUDA 12.4-compatible PyTorch environment
- No internet connection is required at runtime

Install the pinned dependencies before moving the submission to the offline evaluation environment:

```bash
python -m pip install -r requirements.txt
```

## Run

From inside the `Peak` directory:

```bash
python run.py <input-dir> <output-dir>
```

Example:

```bash
python run.py /data/test/NoisyLR /data/test/restored
```

The output directory is created automatically. The program processes only `.npy` files and accepts grayscale arrays shaped `(H, W)` or `(H, W, 1)`.

## Included model

`models/kla_renaf_sr_x2.pt` must be the trained RangeAwareNAFSR checkpoint. It must contain a `model` state dictionary and a `model_config` dictionary. The checkpoint is loaded locally and the program never downloads weights or contacts an external service.
