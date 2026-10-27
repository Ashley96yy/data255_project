"""Training helpers used by notebooks/e0_small_fcn.ipynb."""

from itertools import islice
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from src.metrics import confusion_matrix, iou_from_confusion


def seed_everything(seed):
    """Seed a fresh run; exact reproducibility still depends on the platform."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    """Seed NumPy and Python augmentation RNGs in each DataLoader worker."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def run_epoch(model, loader, device, optimizer=None, max_batches=None):
    """Run one train or validation pass and aggregate all observed pixels."""
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive or None.")
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    matrix = torch.zeros(3, 3, dtype=torch.int64)
    total_loss = 0.0
    pixels = samples = batches = 0
    iterator = loader if max_batches is None else islice(loader, max_batches)

    with torch.set_grad_enabled(training):
        for images, masks in iterator:
            images = images.to(device)
            masks = masks.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite Cross-Entropy loss.")
            if training:
                loss.backward()
                optimizer.step()

            matrix += confusion_matrix(logits.detach().argmax(dim=1), masks).cpu()
            total_loss += loss.item() * masks.numel()
            pixels += masks.numel()
            samples += images.shape[0]
            batches += 1

    if not pixels:
        raise ValueError("The DataLoader produced no pixels.")
    return {
        "loss": total_loss / pixels,
        **iou_from_confusion(matrix),
        "confusion_matrix": matrix.tolist(),
        "samples": samples,
        "pixels": pixels,
        "batches": batches,
    }


def fit(model, train_loader, val_loader, optimizer, device, config, output_dir, metadata):
    """Select the best checkpoint by validation mIoU in a new run directory.

    Smoke runs use batch limits and must not be compared as formal experiments.
    The caller must re-create the model/optimizer/loaders for each fresh run.
    """
    epochs = config["epochs"]
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    if config["mode"] not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'.")
    train_limit = config.get("max_train_batches")
    val_limit = config.get("max_val_batches")
    if config["mode"] == "full" and (train_limit is not None or val_limit is not None):
        raise ValueError("A full run cannot use batch limits.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    run_record = {
        "config": config,
        "metadata": metadata,
        "checkpoint_selection": "highest validation mIoU; first epoch wins ties",
        "status": "running",
    }
    save_json(output_dir / "run.json", run_record)
    history = []
    best_miou = -math.inf
    best_epoch = None
    model.to(device)

    try:
        for epoch in range(1, epochs + 1):
            train_metrics = run_epoch(
                model, train_loader, device, optimizer, train_limit
            )
            val_metrics = run_epoch(model, val_loader, device, max_batches=val_limit)
            score = val_metrics["miou"]
            if score is None or not math.isfinite(score):
                raise FloatingPointError("Validation mIoU is undefined or non-finite.")
            history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
            if score > best_miou:
                best_miou = score
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_metrics": val_metrics,
                        "config": config,
                        "metadata": metadata,
                    },
                    output_dir / "best.pt",
                )
            save_json(output_dir / "history.json", history)
            print(
                f"[{config['mode']}] epoch {epoch}/{epochs} | "
                f"train CE={train_metrics['loss']:.4f} | "
                f"val CE={val_metrics['loss']:.4f} | val mIoU={score:.4f}",
                flush=True,
            )
        run_record.update(
            status="completed", best_epoch=best_epoch, best_val_miou=best_miou
        )
    except Exception as error:
        run_record.update(status="failed", error=str(error))
        save_json(output_dir / "run.json", run_record)
        raise

    save_json(output_dir / "run.json", run_record)
    return history


def save_json(path, value):
    """Write portable JSON; undefined IoUs use null, never NaN."""
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
