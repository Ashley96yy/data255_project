"""Segmentation model definitions."""

from .small_fcn import ConvBlock, SmallFCN, count_trainable_parameters
from .unet import UNet

MODEL_NAMES = ("small_fcn", "unet")


def build_model(config):
    """Build the model named by ``config["model"]`` (default ``small_fcn`` for E0).

    U-Net options: ``base_channels`` (default 64), ``depth`` (4), ``bilinear`` (False).
    """
    name = config.get("model", "small_fcn")
    num_classes = config.get("num_classes", 3)
    if name == "small_fcn":
        return SmallFCN(num_classes=num_classes)
    if name == "unet":
        return UNet(
            num_classes=num_classes,
            base_channels=config.get("base_channels", 64),
            depth=config.get("depth", 4),
            bilinear=config.get("bilinear", False),
        )
    raise ValueError(f"Unknown model {name!r}; expected one of {MODEL_NAMES}.")


__all__ = [
    "ConvBlock", "SmallFCN", "UNet", "MODEL_NAMES", "build_model",
    "count_trainable_parameters",
]
