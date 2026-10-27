"""Dataset-level IoU for the foreground, background, and border classes."""

import torch


CLASS_NAMES = ("foreground", "background", "border")


def confusion_matrix(predictions, targets, num_classes=3):
    """Count pixels with target classes in rows and predictions in columns."""
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have the same shape.")
    if predictions.is_floating_point() or targets.is_floating_point():
        raise ValueError("Predictions and targets must contain integer class IDs.")
    if num_classes < 1:
        raise ValueError("num_classes must be positive.")
    if any(
        ((values < 0) | (values >= num_classes)).any().item()
        for values in (predictions, targets)
    ):
        raise ValueError("Class IDs must be between 0 and num_classes - 1.")
    encoded = targets.reshape(-1).long() * num_classes + predictions.reshape(-1).long()
    return torch.bincount(encoded, minlength=num_classes ** 2).reshape(
        num_classes, num_classes
    )


def iou_from_confusion(matrix):
    """Compute IoUs from aggregate counts, not a mean of batch-level IoUs.

    A class with zero union is undefined (None) and excluded from the mean.
    Background and border are included whenever their union is nonzero.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Expected a square confusion matrix.")
    if (matrix < 0).any().item():
        raise ValueError("Confusion counts cannot be negative.")
    matrix = matrix.detach().to(device="cpu", dtype=torch.float64)
    intersection = matrix.diag()
    union = matrix.sum(dim=0) + matrix.sum(dim=1) - intersection
    per_class = [
        float(hit / total) if total > 0 else None
        for hit, total in zip(intersection, union)
    ]
    defined = [value for value in per_class if value is not None]
    return {
        "miou": sum(defined) / len(defined) if defined else None,
        "per_class_iou": per_class,
    }
