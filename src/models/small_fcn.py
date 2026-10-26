"""Small fully convolutional network for the E0 segmentation baseline."""

from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    """Two 3x3 convolutions, each followed by batch normalization and ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class SmallFCN(nn.Module):
    """Lightweight FCN that returns per-class logits at the input resolution.

    The network intentionally has no decoder or skip connections. Two max-pool
    operations reduce the spatial resolution before a 1x1 classifier, and the
    resulting logits are resized to the original height and width.
    """

    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")

        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(64, 128),
        )
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, inputs):
        """Return unnormalized class logits with the input spatial size."""
        if inputs.ndim != 4:
            raise ValueError(
                "Expected inputs with shape [batch, channels, height, width]."
            )

        input_size = inputs.shape[-2:]
        logits = self.classifier(self.features(inputs))
        return F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )


def count_trainable_parameters(model):
    """Return the number of parameters updated during training."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
