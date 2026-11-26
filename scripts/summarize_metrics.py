"""Summarize validation metrics for an already-selected checkpoint."""

import argparse
import json
from pathlib import Path

import torch

from src.metrics import (
    CLASS_NAMES,
    dice_from_confusion,
    iou_from_confusion,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    with (args.run_dir / "run.json").open() as f:
        run = json.load(f)

    with (args.run_dir / "history.json").open() as f:
        history = json.load(f)

    best_epoch = run["best_epoch"]
    record = next(
        row for row in history if row["epoch"] == best_epoch
    )
    matrix = torch.tensor(record["val"]["confusion_matrix"])

    iou = iou_from_confusion(matrix)
    dice = dice_from_confusion(matrix)

    if abs(iou["miou"] - run["best_val_miou"]) >= 1e-10:
        raise ValueError("Recalculated mIoU differs from the saved result.")

    summary = {
        "split": "validation",
        "selected_epoch": best_epoch,
        "checkpoint_selection": run["checkpoint_selection"],
        "aggregation": "dataset-level confusion matrix",
        "class_names": list(CLASS_NAMES),
        **iou,
        **dice,
        "confusion_matrix": matrix.tolist(),
        "source_files": ["run.json", "history.json"],
    }

    output_path = args.run_dir / "validation_metrics.json"
    output_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"Selected epoch: {best_epoch}")
    print(f"mIoU:       {iou['miou']:.4%}")
    print(f"Macro Dice: {dice['macro_dice']:.4%}")

    for name, class_iou, class_dice in zip(
        CLASS_NAMES,
        iou["per_class_iou"],
        dice["per_class_dice"],
    ):
        print(
            f"{name:12s} IoU: {class_iou:.4%} "
            f"| Dice: {class_dice:.4%}"
        )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()