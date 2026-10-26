"""Dataset loading and preprocessing for Oxford-IIIT Pet segmentation."""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import functional as TF


class PetSegmentationDataset(Dataset):
    """Oxford-IIIT Pet dataset for three-class semantic segmentation."""

    VALID_SPLITS = {"train", "val", "test"}
    CSV_SPLITS = {"train", "val", "excluded"}

    def __init__(
        self,
        data_root,
        split="train",
        split_csv=None,
        image_size=256,
        augment=False,
        download=False,
    ):
        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"Invalid split {split!r}; expected one of {sorted(self.VALID_SPLITS)}."
            )
        if (
            not isinstance(image_size, int)
            or isinstance(image_size, bool)
            or image_size <= 0
        ):
            raise ValueError("image_size must be a positive integer.")
        if augment and split != "train":
            raise ValueError(
                "Data augmentation is only supported for the training split."
            )

        self.split = split
        self.image_size = image_size
        self.augment = augment

        official_split = "test" if split == "test" else "trainval"
        self.dataset = OxfordIIITPet(
            root=str(Path(data_root)),
            split=official_split,
            target_types="segmentation",
            download=download,
        )

        if split in {"train", "val"}:
            if split_csv is None:
                raise ValueError("split_csv is required for train/val.")
            self.indices = self._load_split_indices(split_csv)
        else:
            # Keep the official test split unchanged.
            self.indices = list(range(len(self.dataset)))

    def _load_split_indices(self, split_csv):
        split_path = Path(split_csv)
        if not split_path.is_file():
            raise FileNotFoundError(f"Split CSV not found: {split_path}")

        split_df = pd.read_csv(split_path)
        required_columns = {"dataset_index", "split"}
        missing_columns = required_columns.difference(split_df.columns)
        if missing_columns:
            raise ValueError(
                f"Split CSV is missing required columns: {sorted(missing_columns)}"
            )

        invalid_splits = set(split_df["split"].dropna().unique()).difference(
            self.CSV_SPLITS
        )
        if split_df["split"].isna().any() or invalid_splits:
            raise ValueError(
                f"Split CSV contains invalid split values: {sorted(invalid_splits)}"
            )

        numeric_indices = pd.to_numeric(split_df["dataset_index"], errors="coerce")
        if numeric_indices.isna().any() or not np.equal(
            numeric_indices, np.floor(numeric_indices)
        ).all():
            raise ValueError("Split CSV contains non-integer dataset indices.")

        split_df = split_df.copy()
        split_df["dataset_index"] = numeric_indices.astype(int)

        if split_df["dataset_index"].duplicated().any():
            raise ValueError("Split CSV contains duplicate dataset indices.")
        if (
            (split_df["dataset_index"] < 0)
            | (split_df["dataset_index"] >= len(self.dataset))
        ).any():
            raise ValueError("Split CSV contains a dataset index out of range.")

        return split_df.loc[
            split_df["split"] == self.split, "dataset_index"
        ].tolist()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        dataset_index = self.indices[index]
        image, mask = self.dataset[dataset_index]

        image = image.convert("RGB")
        image = image.resize(
            (self.image_size, self.image_size),
            resample=Image.Resampling.BILINEAR,
        )
        mask = mask.resize(
            (self.image_size, self.image_size),
            resample=Image.Resampling.NEAREST,
        )

        # Apply the same random transformation to the image and its mask.
        if self.augment and random.random() < 0.5:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(
            image_tensor,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )

        # Convert the original mask labels 1, 2, 3 to class indices 0, 1, 2.
        mask_array = np.asarray(mask, dtype=np.int64)
        if not np.isin(mask_array, [1, 2, 3]).all():
            raise ValueError(f"Unexpected labels in sample {dataset_index}.")

        mask_tensor = torch.from_numpy((mask_array - 1).copy()).long()
        return image_tensor, mask_tensor
