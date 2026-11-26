"""Check Dice/weighted-CE values, gradients, config building, and training integration."""

import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.losses import SegmentationLoss, SoftDiceLoss, build_loss, load_class_weights
from src.training import fit, run_epoch, seed_everything


def confident_logits(targets, num_classes=3, scale=50.0):
    return F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float() * scale


class SoftDiceTest(unittest.TestCase):
    def test_perfect_prediction_is_near_zero(self):
        targets = torch.randint(0, 3, (2, 8, 8))
        loss = SoftDiceLoss(smooth=0.0)(confident_logits(targets), targets)
        self.assertLess(loss.item(), 1e-6)

    def test_hand_calculated_uniform_prediction(self):
        # Uniform softmax (1/3 everywhere); targets: 2 px class 0, 2 px class 1, 0 px class 2.
        targets = torch.tensor([[[0, 0], [1, 1]]])
        logits = torch.zeros(1, 3, 2, 2)
        loss = SoftDiceLoss(smooth=0.0)(logits, targets)
        # class 0/1: 2*(2/3)/(4/3+2)=0.4 ; class 2: 0/(4/3)=0
        self.assertAlmostEqual(loss.item(), 1 - (0.4 + 0.4 + 0.0) / 3, places=6)

    def test_gradient_is_finite(self):
        logits = torch.randn(2, 3, 8, 8, requires_grad=True)
        SoftDiceLoss()(logits, torch.randint(0, 3, (2, 8, 8))).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0)


class SegmentationLossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.randn(2, 3, 8, 8)
        self.targets = torch.randint(0, 3, (2, 8, 8))

    def test_ce_only_matches_pytorch(self):
        loss = SegmentationLoss()(self.logits, self.targets)
        self.assertAlmostEqual(loss.item(), F.cross_entropy(self.logits, self.targets).item(), places=6)

    def test_weighted_ce_matches_pytorch(self):
        weights = torch.tensor([1.0, 0.5, 3.0])
        loss = SegmentationLoss(class_weights=weights)(self.logits, self.targets)
        expected = F.cross_entropy(self.logits, self.targets, weight=weights)
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)

    def test_combined_is_sum_of_components(self):
        criterion = SegmentationLoss(dice_weight=0.5)
        loss = criterion(self.logits, self.targets)
        parts = criterion.last_components
        self.assertEqual(set(parts), {"ce", "dice"})
        self.assertAlmostEqual(loss.item(), parts["ce"] + 0.5 * parts["dice"], places=6)

    def test_invalid_settings_raise(self):
        for kwargs in (
            {"ce_weight": 0.0, "dice_weight": 0.0},
            {"dice_weight": -1.0},
            {"class_weights": [1.0, 0.0, 2.0]},
        ):
            with self.assertRaises(ValueError):
                SegmentationLoss(**kwargs)


class BuildLossTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.weights_path = Path(self.directory.name) / "class_weights.json"
        self.weights_path.write_text(json.dumps({
            "source_split": "train",
            "weights": {"median_frequency": [1.0, 0.5, 2.5], "inverse_frequency": [1, 1, 1]},
        }))

    def tearDown(self):
        self.directory.cleanup()

    def test_all_experiment_losses_build(self):
        expectations = {
            "ce": (False, False), "ce_dice": (False, True),
            "wce": (True, False), "wce_dice": (True, True),
        }
        for name, (weighted, has_dice) in expectations.items():
            criterion = build_loss({"name": name, "class_weights_path": self.weights_path})
            self.assertEqual(criterion.class_weights is not None, weighted, name)
            self.assertEqual(criterion.dice is not None, has_dice, name)
        self.assertEqual(build_loss(None).dice_weight, 0.0)

    def test_weighted_loss_requires_weights_and_train_source(self):
        with self.assertRaises(ValueError):
            build_loss({"name": "wce"})
        self.weights_path.write_text(json.dumps({"source_split": "trainval", "weights": {}}))
        with self.assertRaises(ValueError):
            load_class_weights(self.weights_path)

    def test_unknown_loss_raises(self):
        with self.assertRaises(ValueError):
            build_loss({"name": "focal"})


class TrainingIntegrationTest(unittest.TestCase):
    def test_fit_with_combined_loss_logs_components_and_dice(self):
        seed_everything(42)
        model = nn.Conv2d(3, 3, kernel_size=1)
        loader = DataLoader(
            TensorDataset(torch.randn(4, 3, 8, 8), torch.randint(0, 3, (4, 8, 8))),
            batch_size=2,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = SegmentationLoss(class_weights=[1.0, 0.5, 2.0], dice_weight=1.0)
        with tempfile.TemporaryDirectory() as directory:
            history = fit(model, loader, loader, optimizer, "cpu",
                          {"mode": "full", "epochs": 1}, Path(directory) / "run", {},
                          criterion=criterion)
            run = json.loads((Path(directory) / "run" / "run.json").read_text())
        val = history[0]["val"]
        for key in ("loss_ce", "loss_dice", "macro_dice", "per_class_dice"):
            self.assertIn(key, val)
        self.assertAlmostEqual(val["loss"], val["loss_ce"] + val["loss_dice"], places=5)
        self.assertIn("SegmentationLoss", run["loss"])

    def test_default_criterion_is_plain_cross_entropy(self):
        model = nn.Conv2d(3, 3, kernel_size=1)
        images, masks = torch.randn(2, 3, 8, 8), torch.randint(0, 3, (2, 8, 8))
        metrics = run_epoch(model, DataLoader(TensorDataset(images, masks), batch_size=2), "cpu")
        with torch.no_grad():
            expected = F.cross_entropy(model(images), masks).item()
        self.assertAlmostEqual(metrics["loss"], expected, places=6)
        self.assertNotIn("loss_ce", metrics)


if __name__ == "__main__":
    unittest.main()
