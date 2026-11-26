"""U-Net for the E1-E4 loss comparison.

Follows the encoder-decoder design of Ronneberger, Fischer & Brox (2015),
"U-Net: Convolutional Networks for Biomedical Image Segmentation",
https://arxiv.org/abs/1505.04597, with two common modern changes that keep it
consistent with the E0 Small FCN:

* "same" padding (``padding=1``), so logits have the input height and width
  and no output cropping is needed;
* batch normalization after each convolution (the shared ``ConvBlock``).

Defaults match the paper: 4 down-sampling stages, 64 base channels doubling to
1024 at the bottleneck, 2x2 transposed-convolution up-sampling, and skip
connections that concatenate encoder features into the decoder.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .small_fcn import ConvBlock


class Down(nn.Sequential):
    """2x2 max-pool followed by a ConvBlock."""

    def __init__(self, in_channels, out_channels):
        super().__init__(nn.MaxPool2d(kernel_size=2, stride=2), ConvBlock(in_channels, out_channels))


class Up(nn.Module):
    """Up-sample, concatenate the skip connection, then a ConvBlock."""

    def __init__(self, in_channels, skip_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)

    def forward(self, inputs, skip):
        inputs = self.up(inputs)
        # Odd input sizes lose a row/column when pooled; pad back to the skip size.
        diff_h = skip.shape[-2] - inputs.shape[-2]
        diff_w = skip.shape[-1] - inputs.shape[-1]
        if diff_h or diff_w:
            inputs = F.pad(inputs, [diff_w // 2, diff_w - diff_w // 2,
                                    diff_h // 2, diff_h - diff_h // 2])
        return self.conv(torch.cat([skip, inputs], dim=1))


class UNet(nn.Module):
    """U-Net returning raw per-class logits ``[N, num_classes, H, W]``."""

    def __init__(self, in_channels=3, num_classes=3, base_channels=64, depth=4, bilinear=False):
        super().__init__()
        if in_channels <= 0 or num_classes <= 0:
            raise ValueError("in_channels and num_classes must be positive.")
        if base_channels <= 0 or depth < 1:
            raise ValueError("base_channels must be positive and depth at least 1.")

        channels = [base_channels * 2 ** level for level in range(depth + 1)]
        self.depth = depth
        self.stem = ConvBlock(in_channels, channels[0])
        self.down = nn.ModuleList(
            Down(channels[level], channels[level + 1]) for level in range(depth)
        )
        self.up = nn.ModuleList(
            Up(channels[level + 1], channels[level], channels[level], bilinear)
            for level in reversed(range(depth))
        )
        self.classifier = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, inputs):
        if inputs.ndim != 4:
            raise ValueError("Expected inputs with shape [batch, channels, height, width].")
        min_size = 2 ** self.depth
        if min(inputs.shape[-2:]) < min_size:
            raise ValueError(f"Height and width must be at least {min_size}.")

        features = self.stem(inputs)
        skips = [features]
        for block in self.down:
            features = block(features)
            skips.append(features)
        skips.pop()  # the bottleneck output is not a skip connection
        for block in self.up:
            features = block(features, skips.pop())
        return self.classifier(features)
