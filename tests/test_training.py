"""Check gradient boundaries, pixel accounting, and checkpoint selection."""

import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training import fit, run_epoch, seed_everything


class TrainingTest(unittest.TestCase):
    def setUp(self):
        seed_everything(42)
        self.model = nn.Conv2d(3, 3, kernel_size=1)
        self.loader = DataLoader(
            TensorDataset(torch.randn(5, 3, 8, 8), torch.randint(0, 3, (5, 8, 8))),
            batch_size=2,
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def test_training_changes_weights_validation_does_not(self):
        before = self.model.weight.detach().clone()
        train = run_epoch(self.model, self.loader, "cpu", self.optimizer, max_batches=2)
        self.assertFalse(torch.equal(before, self.model.weight))
        self.assertEqual(train["samples"], 4)
        before = self.model.weight.detach().clone()
        self.optimizer.zero_grad(set_to_none=True)
        val = run_epoch(self.model, self.loader, "cpu")
        self.assertTrue(torch.equal(before, self.model.weight))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertFalse(self.model.training)
        self.assertEqual(val["samples"], 5)
        self.assertEqual(val["pixels"], 5 * 8 * 8)
        self.assertEqual(sum(map(sum, val["confusion_matrix"])), val["pixels"])

    def test_best_checkpoint_matches_history_and_reloads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            config = {"mode": "full", "epochs": 2}
            history = fit(
                self.model, self.loader, self.loader, self.optimizer,
                "cpu", config, path, {"test": True},
            )
            checkpoint = torch.load(path / "best.pt", weights_only=True)
            best = max(history, key=lambda row: row["val"]["miou"])
            self.assertEqual(checkpoint["epoch"], best["epoch"])
            restored = nn.Conv2d(3, 3, kernel_size=1)
            restored.load_state_dict(checkpoint["model_state_dict"])
            metrics = run_epoch(restored, self.loader, "cpu")
            self.assertEqual(metrics, best["val"])
            self.assertEqual(json.loads((path / "run.json").read_text())["status"], "completed")
            with self.assertRaises(FileExistsError):
                fit(self.model, self.loader, self.loader, self.optimizer,
                    "cpu", config, path, {})

    def test_full_run_rejects_batch_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                fit(self.model, self.loader, self.loader, self.optimizer, "cpu",
                    {"mode": "full", "epochs": 1, "max_val_batches": 1},
                    Path(directory) / "run", {})


if __name__ == "__main__":
    unittest.main()
