"""Smoke tests for the E0 Small FCN reference baseline."""

import unittest

import torch
from torch import nn

from src.models.small_fcn import SmallFCN, count_trainable_parameters


class SmallFCNTest(unittest.TestCase):
    def test_forward_and_cross_entropy_loss(self):
        torch.manual_seed(255)
        model = SmallFCN(in_channels=3, num_classes=3)
        images = torch.randn(8, 3, 256, 256)
        masks = torch.randint(0, 3, (8, 256, 256), dtype=torch.long)

        logits = model(images)
        loss = nn.CrossEntropyLoss()(logits, masks)

        self.assertEqual(logits.shape, (8, 3, 256, 256))
        self.assertTrue(torch.isfinite(logits).all().item())
        self.assertTrue(torch.isfinite(loss).item())

        parameter_count = count_trainable_parameters(model)
        self.assertGreater(parameter_count, 0)
        print(f"SmallFCN trainable parameters: {parameter_count:,}")
        print(f"CrossEntropyLoss: {loss.item():.6f}")


if __name__ == "__main__":
    unittest.main()
