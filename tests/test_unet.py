"""U-Net shape/gradient checks and a synthetic end-to-end run of scripts/train.py."""

import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.data import TensorDataset

from scripts.train import prepare_config, train_one
from src.losses import build_loss
from src.models import SmallFCN, UNet, build_model, count_trainable_parameters


class UNetTest(unittest.TestCase):
    def test_paper_configuration_shape_and_size(self):
        model = UNet()
        logits = model(torch.randn(2, 3, 256, 256))
        self.assertEqual(logits.shape, (2, 3, 256, 256))
        parameters = count_trainable_parameters(model)
        # 64->1024 channels with BN: ~31M parameters, like the original U-Net.
        self.assertGreater(parameters, 30_000_000)
        self.assertLess(parameters, 32_000_000)
        print(f"UNet(base_channels=64) trainable parameters: {parameters:,}")

    def test_odd_input_size_and_bilinear_variant(self):
        for bilinear in (False, True):
            model = UNet(base_channels=8, bilinear=bilinear).eval()
            with torch.no_grad():
                logits = model(torch.randn(1, 3, 97, 130))
            self.assertEqual(logits.shape, (1, 3, 97, 130))

    def test_all_parameters_receive_gradients_with_combined_loss(self):
        torch.manual_seed(0)
        model = UNet(base_channels=8)
        images, masks = torch.randn(2, 3, 64, 64), torch.randint(0, 3, (2, 64, 64))
        loss = build_loss({"name": "ce_dice"})(model(images), masks)
        loss.backward()
        missing = [name for name, p in model.named_parameters() if p.grad is None]
        self.assertEqual(missing, [])
        self.assertTrue(torch.isfinite(loss))

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            UNet(base_channels=0)
        with self.assertRaises(ValueError):
            UNet(base_channels=8)(torch.randn(1, 3, 8, 8))  # smaller than 2**depth


class BuildModelTest(unittest.TestCase):
    def test_build_model(self):
        self.assertIsInstance(build_model({}), SmallFCN)
        model = build_model({"model": "unet", "base_channels": 16})
        self.assertIsInstance(model, UNet)
        self.assertEqual(model.stem[0].out_channels, 16)
        with self.assertRaises(ValueError):
            build_model({"model": "deeplab"})

    def test_experiment_configs_build(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("e1_unet_ce", "e2_unet_ce_dice", "e3_unet_wce", "e4_unet_wce_dice"):
            config = json.loads((root / "configs" / f"{name}.json").read_text())
            self.assertIsInstance(build_model(config), UNet, name)


class TrainScriptTest(unittest.TestCase):
    def test_smoke_run_writes_summary(self):
        base = {"model": "unet", "base_channels": 4, "seed": 1, "seeds": [1, 2],
                "batch_size": 2, "num_workers": 0, "epochs": 30, "learning_rate": 1e-3,
                "loss": {"name": "ce_dice"}}
        config = prepare_config(base, seed=7, mode="smoke")
        self.assertEqual((config["epochs"], config["max_train_batches"]), (1, 2))
        self.assertNotIn("seeds", config)
        data = TensorDataset(torch.randn(6, 3, 32, 32), torch.randint(0, 3, (6, 32, 32)))
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            summary = train_one(config, run_dir, torch.device("cpu"), data, data)
            saved = json.loads((run_dir / "summary.json").read_text())
            run = json.loads((run_dir / "run.json").read_text())
            self.assertTrue((run_dir / "best.pt").is_file())
        self.assertEqual(saved, summary)
        self.assertEqual(saved["seed"], 7)
        self.assertEqual(set(saved["val_per_class_iou"]), {"foreground", "background", "border"})
        self.assertIn("SegmentationLoss", run["loss"])
        self.assertEqual(run["metadata"]["trainable_parameters"], saved["trainable_parameters"])


if __name__ == "__main__":
    unittest.main()
