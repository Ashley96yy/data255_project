"""Compute E3/E4 class weights from the cleaned TRAINING split only.

Pixels are counted after the same 256 x 256 nearest-neighbour mask resize used
in training, without augmentation (a horizontal flip does not change counts).
Validation and test masks are never read.

Usage (from the repository root):

    python -m scripts.compute_class_weights --data-root data
    # writes configs/class_weights.json

Two schemes are saved; the loss config picks one with ``class_weight_scheme``:

* ``median_frequency`` (default, Eigen & Fergus 2015): ``w_c = median(f) / f_c``
* ``inverse_frequency``: ``w_c = 1 / f_c``, rescaled so the weights average 1
"""

import argparse
import hashlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import PetSegmentationDataset
from src.metrics import CLASS_NAMES
from src.training import save_json


def count_pixels(dataset, num_classes=3, batch_size=32, num_workers=0):
    counts = torch.zeros(num_classes, dtype=torch.int64)
    images = 0
    for _, masks in DataLoader(dataset, batch_size=batch_size, num_workers=num_workers):
        counts += torch.bincount(masks.reshape(-1), minlength=num_classes)[:num_classes]
        images += masks.shape[0]
    return counts, images


def weights_from_counts(counts):
    counts = torch.as_tensor(counts, dtype=torch.float64)
    if (counts <= 0).any():
        raise ValueError("Every class needs at least one training pixel.")
    frequency = counts / counts.sum()
    inverse = 1 / frequency
    return {
        "frequency": frequency.tolist(),
        "weights": {
            "median_frequency": (frequency.median() / frequency).tolist(),
            "inverse_frequency": (inverse / inverse.mean()).tolist(),
        },
    }


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split-csv", type=Path, default=Path("configs/train_val_split_clean.csv"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("configs/class_weights.json"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    dataset = PetSegmentationDataset(
        args.data_root, split="train", split_csv=args.split_csv,
        image_size=args.image_size, augment=False, download=args.download,
    )
    counts, images = count_pixels(dataset, num_workers=args.num_workers)
    result = weights_from_counts(counts)
    record = {
        "source_split": "train",
        "num_images": images,
        "image_size": args.image_size,
        "mask_resize": "nearest",
        "class_names": list(CLASS_NAMES),
        "pixel_counts": counts.tolist(),
        **result,
        "default_scheme": "median_frequency",
        "split_csv": str(args.split_csv),
        "split_csv_sha256": sha256(args.split_csv),
    }
    save_json(args.output, record)

    print(f"Counted {images} training images at {args.image_size}x{args.image_size}.")
    for i, name in enumerate(CLASS_NAMES):
        print(
            f"{name:11s} freq={result['frequency'][i]:.4f} "
            f"median_freq_w={result['weights']['median_frequency'][i]:.4f} "
            f"inverse_freq_w={result['weights']['inverse_frequency'][i]:.4f}"
        )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
