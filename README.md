# DATA255 U-Net Segmentation

Initial project scaffold for a reproducible DATA255 proposal on U-Net-based semantic segmentation using the Oxford-IIIT Pet dataset.

## Project goal

Build and evaluate a U-Net segmentation baseline on pet images and trimap annotations. The planned evaluation includes region-level metrics (Dice and IoU) and boundary-focused error analysis. Research questions, hypotheses, and final experiment variants will be documented before model training begins.

## Repository structure

```text
.
├── configs/                 # Experiment configuration files
├── docs/                    # Proposal, literature matrix, and reports
├── notebooks/
│   └── data_audit.ipynb    # Dataset inspection and validation
├── results/                 # Generated metrics/figures (not tracked)
├── scripts/                 # Training and evaluation entry points
├── src/
│   ├── dataset.py           # Dataset loading and preprocessing
│   ├── losses/              # Segmentation loss implementations
│   ├── metrics.py           # Dice, IoU, and boundary metrics
│   └── models/              # U-Net and model variants
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

Python 3.10 or newer is recommended. In a virtual environment or Google Colab runtime:

```bash
python -m pip install -r requirements.txt
```

The initial dependency list is intentionally unpinned until the baseline environment is validated. After validation, record the Python, PyTorch, Torchvision, CUDA, and package versions used for the experiments.

## Data policy

Do not commit datasets, downloaded archives, checkpoints, or model weights. Download the Oxford-IIIT Pet dataset directly in the execution environment and keep it under `data/`, which is ignored by Git.

The data audit should verify at minimum:

- dataset source and download method;
- sample counts and train/test split;
- image and trimap dimensions;
- trimap label meanings and pixel counts;
- image-mask alignment;
- missing, unreadable, or duplicate files;
- class imbalance and representative visual examples.

## Proposed workflow

1. Complete and commit the data audit notebook without embedding raw data.
2. Finalize the research question, hypotheses, split policy, preprocessing, and metrics.
3. Add a baseline configuration and a deterministic training script.
4. Train only on the training split; use validation data for model selection.
5. Evaluate once on the held-out test split and save aggregate and boundary-level metrics.
6. Document environment versions, random seeds, configuration, and limitations.

## Week 3 E0 reference baseline

`src/models/small_fcn.py` implements a lightweight fully convolutional network
for a three-class Cross-Entropy reference baseline:

```text
ConvBlock(3, 32) -> MaxPool -> ConvBlock(32, 64) -> MaxPool
-> ConvBlock(64, 128) -> 1x1 Conv(3 classes) -> bilinear upsampling
```

The output contains raw logits (no softmax) and has the same height and width
as the input. The model intentionally does not use U-Net skip connections.

Open `notebooks/e0_small_fcn.ipynb` for the complete experiment workflow:
cleaned train/validation DataLoaders, the visible model definition, forward/loss
checks, Adam training, validation IoU, best-checkpoint reload, and plots.

Run the notebook top to bottom with `RUN_MODE = "smoke"` first. This performs
one epoch of two training batches and two validation batches on real data.
For the full experiment, set `RUN_MODE = "full"` and restart/run all; the
initial configuration in `configs/e0_small_fcn.json` uses 20 epochs, seed 42,
batch size 8, and learning rate 0.001. These are starter settings, not tuned
results. The notebook has a setup cell for Colab; select a GPU runtime and set
`DOWNLOAD_DATA = True` there if needed.

The data root is detected under `data/` or `notebooks/data/`. Alternatively,
set `E0_DATA_ROOT` to a directory containing `oxford-iiit-pet/`. Both train and
validation use `configs/train_val_split_clean.csv`; the test split is reserved.
IoU uses the aggregate validation confusion matrix, includes background and
border, and excludes a class from the mean only when its union is zero.

Each run creates `results/e0_small_fcn/<mode>_<time>_<id>/` with `run.json`,
`history.json`, `best.pt`, `curves.png`, and `prediction.png`. The checkpoint
is selected by highest validation mIoU. Run metadata includes environment
versions, configuration, split/source hashes, Git state, and run mode. These
artifacts are ignored by Git. Download or persist the run directory before
ending a Colab session. Smoke results are pipeline checks, not formal baseline
scores.

Run the model smoke test from the repository root:

```bash
python -m unittest -v tests/test_small_fcn.py
```

The test uses a synthetic batch with shape `[8, 3, 256, 256]`, checks the
output shape, verifies that the logits and Cross-Entropy loss are finite, and
prints the trainable parameter count. It does not train the model or report an
experimental segmentation score.

Run all model, metric, and training checks with:

```bash
python -m unittest discover -s tests -v
```

## Reproducibility checklist

- Keep experiment parameters in `configs/`.
- Set and report random seeds.
- Record the exact environment after the first validated run.
- Save commands and configuration names with each result.
- Keep generated results and weights outside Git; commit only small summaries needed for the report.
