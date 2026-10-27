"""Hand-calculated checks of the validation metric convention."""

import unittest

import torch

from src.metrics import confusion_matrix, iou_from_confusion


class IoUTest(unittest.TestCase):
    def test_three_classes(self):
        targets = torch.tensor([0, 0, 1, 1, 2, 2])
        predictions = torch.tensor([0, 1, 1, 2, 2, 0])
        matrix = confusion_matrix(predictions, targets)
        self.assertEqual(matrix.tolist(), [[1, 1, 0], [0, 1, 1], [1, 0, 1]])
        result = iou_from_confusion(matrix)
        self.assertAlmostEqual(result["miou"], 1 / 3)
        for value in result["per_class_iou"]:
            self.assertAlmostEqual(value, 1 / 3)

    def test_absent_classes_are_undefined(self):
        result = iou_from_confusion(torch.diag(torch.tensor([5, 0, 0])))
        self.assertEqual(result, {"miou": 1.0, "per_class_iou": [1.0, None, None]})

    def test_aggregate_before_calculating_iou(self):
        # Correct global mean = (9/10 + 1/2)/2 = 0.7, not batch mean 0.625.
        first = confusion_matrix(torch.tensor([0] * 9), torch.tensor([0] * 9))
        second = confusion_matrix(torch.tensor([0, 1]), torch.tensor([1, 1]))
        self.assertAlmostEqual(iou_from_confusion(first + second)["miou"], 0.7)

    def test_invalid_class_raises(self):
        with self.assertRaises(ValueError):
            confusion_matrix(torch.tensor([3]), torch.tensor([0]))


if __name__ == "__main__":
    unittest.main()
