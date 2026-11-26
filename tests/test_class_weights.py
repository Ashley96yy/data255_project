"""Check class-weight formulas and train-only pixel counting on synthetic masks."""

import unittest

import torch
from torch.utils.data import TensorDataset

from scripts.compute_class_weights import count_pixels, weights_from_counts


class ClassWeightTest(unittest.TestCase):
    def test_hand_calculated_weights(self):
        result = weights_from_counts([30, 60, 10])
        self.assertEqual([round(f, 6) for f in result["frequency"]], [0.3, 0.6, 0.1])
        median = result["weights"]["median_frequency"]
        for actual, expected in zip(median, [1.0, 0.5, 3.0]):
            self.assertAlmostEqual(actual, expected)
        inverse = result["weights"]["inverse_frequency"]
        self.assertAlmostEqual(sum(inverse) / 3, 1.0)
        self.assertAlmostEqual(inverse[2] / inverse[0], 3.0)

    def test_missing_class_raises(self):
        with self.assertRaises(ValueError):
            weights_from_counts([10, 0, 5])

    def test_count_pixels(self):
        masks = torch.tensor([[[0, 0], [1, 2]], [[1, 1], [1, 2]]])
        counts, images = count_pixels(TensorDataset(torch.zeros(2, 3, 2, 2), masks), batch_size=1)
        self.assertEqual(counts.tolist(), [2, 4, 2])
        self.assertEqual(images, 2)


if __name__ == "__main__":
    unittest.main()
