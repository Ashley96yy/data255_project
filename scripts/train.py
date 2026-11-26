"""Train one config (E0-E4) for one or more seeds from the command line.

Examples (from the repository root):

    # quick pipeline check on real data: 1 epoch, 2 train + 2 val batches
    python -m scripts.train configs/e1_unet_ce.json --mode smoke

    # full E1 run for the three proposal seeds
    python -m scripts.train configs/e1_unet_ce.json --seeds 42 43 44 --num-workers 4

    # all four loss settings, three seeds each
    for c in e1_unet_ce e2_unet_ce_dice e3_unet_wce e4_unet_wce_dice; do
        python -m scripts.train configs/$c.json --seeds 42 43 44 --num-workers 4
    done

Each run writes ``results/<config name>/<mode>_seed<seed>_<UTC time>_<id>/``
with ``run.json``, ``history.json``, ``best.pt`` (ignored by Git) and
``summary.json`` (selected-epoch validation metrics, parameter count, and
training time). The checkpoint is selected by validation mIoU; the test split
is never loaded here.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from uuid import uuid4

import numpy as np
import pandas as pd
import PIL
import torch
import torchvision
from torch.utils.data import DataLoader

from src.dataset import PetSegmentationDataset
from src.losses import build_loss
from src.metrics import CLASS_NAMES
from src.models import build_model, count_trainable_parameters
from src.training import fit, save_json, seed_everything, seed_worker

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = [
    "src/dataset.py", "src/metrics.py", "src/training.py", "src/losses/__init__.py",
    "src/models/__init__.py", "src/models/small_fcn.py", "src/models/unet.py",
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_state():
    def run(*args):
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True).strip()

    try:
        return {
            "git_commit": run("rev-parse", "HEAD"),
            "git_tracked_changes": bool(run("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_tracked_changes": None}


def environment(device):
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pillow": PIL.__version__,
        "device": str(device),
        "device_name": (torch.cuda.get_device_name() if device.type == "cuda"
                        else platform.machine()),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def prepare_config(config, seed, mode, num_workers=None):
    config = dict(config, seed=seed, mode=mode)
    config.pop("seeds", None)
    if num_workers is not None:
        config["num_workers"] = num_workers
    if mode == "smoke":
        config.update(epochs=1, max_train_batches=2, max_val_batches=2)
    else:
        config.update(max_train_batches=None, max_val_batches=None)
    loss = dict(config.get("loss") or {"name": "ce"})
    if "class_weights_path" in loss:
        loss["class_weights_path"] = str(resolve(loss["class_weights_path"]))
    config["loss"] = loss
    return config


def train_one(config, run_dir, device, train_dataset, val_dataset, extra_metadata=None):
    """Train one seeded run and return its summary dict."""
    seed_everything(config["seed"])
    loader_options = dict(
        batch_size=config["batch_size"], num_workers=config["num_workers"],
        pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
    )
    train_loader = DataLoader(
        train_dataset, shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"]), **loader_options,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    model = build_model(config).to(device)
    criterion = build_loss(config["loss"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    parameters = count_trainable_parameters(model)
    metadata = {
        **environment(device),
        **(extra_metadata or {}),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "class_names": list(CLASS_NAMES),
        "trainable_parameters": parameters,
    }

    start = time.perf_counter()
    history = fit(model, train_loader, val_loader, optimizer, device, config,
                  run_dir, metadata, criterion=criterion)
    seconds = time.perf_counter() - start

    best = max(history, key=lambda row: row["val"]["miou"])
    val = best["val"]
    summary = {
        "experiment": config.get("experiment"),
        "model": config.get("model", "small_fcn"),
        "loss": config["loss"],
        "seed": config["seed"],
        "mode": config["mode"],
        "selected_epoch": best["epoch"],
        "epochs_run": len(history),
        "val_miou": val["miou"],
        "val_per_class_iou": dict(zip(CLASS_NAMES, val["per_class_iou"])),
        "val_macro_dice": val["macro_dice"],
        "val_per_class_dice": dict(zip(CLASS_NAMES, val["per_class_dice"])),
        "trainable_parameters": parameters,
        "train_seconds": round(seconds, 1),
        "seconds_per_epoch": round(seconds / len(history), 1),
    }
    save_json(Path(run_dir) / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="Default: the config's 'seed'.")
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("DATA255_DATA_ROOT", REPO_ROOT / "data")))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config_path = resolve(args.config)
    base_config = json.loads(config_path.read_text(encoding="utf-8"))
    base_config.setdefault("experiment", config_path.stem)
    seeds = args.seeds or [base_config["seed"]]
    device = torch.device(args.device)
    split_csv = resolve(base_config["split_csv"])

    if not args.download and not (args.data_root / "oxford-iiit-pet/images").is_dir():
        raise FileNotFoundError(
            f"No oxford-iiit-pet/ under {args.data_root}. Use --data-root, "
            "DATA255_DATA_ROOT, or --download."
        )
    train_dataset = PetSegmentationDataset(
        args.data_root, split="train", split_csv=split_csv,
        image_size=base_config["image_size"], augment=base_config["augment"],
        download=args.download,
    )
    val_dataset = PetSegmentationDataset(
        args.data_root, split="val", split_csv=split_csv,
        image_size=base_config["image_size"], augment=False,
    )
    shared_metadata = {
        **git_state(),
        "config_file": str(config_path.relative_to(REPO_ROOT)),
        "split_sha256": sha256(split_csv),
        "source_sha256": {name: sha256(REPO_ROOT / name) for name in SOURCE_FILES},
    }
    print(f"{config_path.stem}: train={len(train_dataset)} val={len(val_dataset)} "
          f"device={device} seeds={seeds} mode={args.mode}", flush=True)

    for seed in seeds:
        config = prepare_config(base_config, seed, args.mode, args.num_workers)
        loss = config["loss"]
        if "class_weights_path" in loss:
            shared_metadata["class_weights_sha256"] = sha256(loss["class_weights_path"])
        stamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
        run_dir = args.output_root / config_path.stem / f"{args.mode}_seed{seed}_{stamp}"
        summary = train_one(config, run_dir, device, train_dataset, val_dataset, shared_metadata)
        print(f"seed {seed}: best epoch {summary['selected_epoch']} | "
              f"val mIoU {summary['val_miou']:.4f} | "
              f"border IoU {summary['val_per_class_iou']['border']:.4f} | "
              f"{summary['seconds_per_epoch']}s/epoch | saved {run_dir}", flush=True)


if __name__ == "__main__":
    main()
