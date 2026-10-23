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

## Reproducibility checklist

- Keep experiment parameters in `configs/`.
- Set and report random seeds.
- Record the exact environment after the first validated run.
- Save commands and configuration names with each result.
- Keep generated results and weights outside Git; commit only small summaries needed for the report.
