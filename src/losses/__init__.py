"""Segmentation losses for the E1-E4 loss comparison.

All losses take raw logits ``[N, C, H, W]`` and integer class targets
``[N, H, W]`` with values in ``0..C-1`` (foreground, background, border).

Experiment mapping (proposal Section E):

    E1  ce        Cross-Entropy
    E2  ce_dice   Cross-Entropy + Dice
    E3  wce       class-weighted Cross-Entropy
    E4  wce_dice  class-weighted Cross-Entropy + Dice
"""

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


LOSS_NAMES = ("ce", "ce_dice", "wce", "wce_dice")


class SoftDiceLoss(nn.Module):
    """Multi-class soft Dice loss: ``1 - mean_c Dice_c``.

    Softmax probabilities are compared with one-hot targets. Intersections and
    sums are pooled over the whole batch (batch and spatial dimensions) before
    the per-class ratio, so a class that is small in one image still receives a
    stable gradient. ``smooth`` is added to the numerator and denominator.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        if smooth < 0:
            raise ValueError("smooth must be non-negative.")
        self.smooth = float(smooth)

    def forward(self, logits, targets):
        if logits.ndim != 4 or targets.ndim != 3:
            raise ValueError("Expected logits [N, C, H, W] and targets [N, H, W].")
        num_classes = logits.shape[1]
        probabilities = logits.float().softmax(dim=1)
        one_hot = F.one_hot(targets.long(), num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probabilities * one_hot).sum(dims)
        denominator = probabilities.sum(dims) + one_hot.sum(dims)
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        return 1 - dice.mean()


class SegmentationLoss(nn.Module):
    """``ce_weight * CE(weights) + dice_weight * SoftDice``.

    Keeps the last component values in ``self.last_components`` for logging.
    """

    def __init__(self, class_weights=None, ce_weight=1.0, dice_weight=0.0, smooth=1.0):
        super().__init__()
        if ce_weight < 0 or dice_weight < 0 or ce_weight + dice_weight == 0:
            raise ValueError("Loss weights must be non-negative and not both zero.")
        if class_weights is not None:
            class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
            if class_weights.ndim != 1 or (class_weights <= 0).any():
                raise ValueError("class_weights must be a 1-D tensor of positive values.")
        # A buffer moves with model/loss .to(device) and is saved in state_dict.
        self.register_buffer("class_weights", class_weights)
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.dice = SoftDiceLoss(smooth) if dice_weight > 0 else None
        self.last_components = {}

    def forward(self, logits, targets):
        total = logits.new_zeros(())
        components = {}
        if self.ce_weight > 0:
            weight = None
            if self.class_weights is not None:
                weight = self.class_weights.to(device=logits.device, dtype=logits.dtype)
            ce = F.cross_entropy(logits, targets, weight=weight)
            total = total + self.ce_weight * ce
            components["ce"] = ce.item()
        if self.dice is not None:
            dice = self.dice(logits, targets)
            total = total + self.dice_weight * dice
            components["dice"] = dice.item()
        self.last_components = components
        return total


def load_class_weights(path, scheme="median_frequency"):
    """Read train-split class weights written by scripts/compute_class_weights.py."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("source_split") != "train":
        raise ValueError("Class weights must be computed from the training split only.")
    if scheme not in record["weights"]:
        raise KeyError(f"Unknown weighting scheme {scheme!r}; have {sorted(record['weights'])}.")
    return torch.tensor(record["weights"][scheme], dtype=torch.float32)


def build_loss(loss_config=None):
    """Build a loss from a config dict; ``None`` gives plain Cross-Entropy (E0/E1).

    Example::

        {"name": "wce_dice", "dice_weight": 1.0,
         "class_weights_path": "configs/class_weights.json",
         "class_weight_scheme": "median_frequency"}
    """
    loss_config = dict(loss_config or {"name": "ce"})
    name = loss_config.get("name", "ce")
    if name not in LOSS_NAMES:
        raise ValueError(f"Unknown loss {name!r}; expected one of {LOSS_NAMES}.")
    weights = None
    if name.startswith("wce"):
        path = loss_config.get("class_weights_path")
        if path is None:
            raise ValueError(f"{name!r} requires class_weights_path.")
        weights = load_class_weights(
            path, loss_config.get("class_weight_scheme", "median_frequency")
        )
    dice_weight = loss_config.get("dice_weight", 1.0) if name.endswith("_dice") else 0.0
    return SegmentationLoss(
        class_weights=weights,
        ce_weight=loss_config.get("ce_weight", 1.0),
        dice_weight=dice_weight,
        smooth=loss_config.get("dice_smooth", 1.0),
    )
