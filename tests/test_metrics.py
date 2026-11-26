"""Hand-calculated checks of the validation metric convention."""

import unittest

import torch

from src.metrics import (
    confusion_matrix,
    iou_from_confusion,
    dice_from_confusion,
)


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

class DiceTest(unittest.TestCase):
    def test_hand_calculated_scores(self):
        matrix = torch.tensor([
            [6, 2, 2],
            [1, 8, 1],
            [1, 0, 9],
        ])

        result = dice_from_confusion(matrix)
        expected = [12 / 18, 16 / 20, 18 / 22]

        for actual, target in zip(result["per_class_dice"], expected):
            self.assertAlmostEqual(actual, target)

        self.assertAlmostEqual(
            result["macro_dice"], sum(expected) / 3
        )

    def test_perfect_predictions(self):
        result = dice_from_confusion(
            torch.diag(torch.tensor([5, 7, 3]))
        )
        self.assertEqual(result["per_class_dice"], [1.0, 1.0, 1.0])
        self.assertEqual(result["macro_dice"], 1.0)

    def test_absent_classes_are_undefined(self):
        result = dice_from_confusion(
            torch.diag(torch.tensor([5, 0, 0]))
        )
        self.assertEqual(result["per_class_dice"], [1.0, None, None])
        self.assertEqual(result["macro_dice"], 1.0)

    def test_missed_class_scores_zero(self):
        # Class 1 exists in targets, but all its pixels become class 0.
        matrix = torch.tensor([
            [5, 0, 0],
            [3, 0, 0],
            [0, 0, 0],
        ])
        result = dice_from_confusion(matrix)
        self.assertEqual(result["per_class_dice"][1], 0.0)

    def test_aggregate_before_calculating_dice(self):
        first = torch.tensor([[9, 0], [0, 0]])
        second = torch.tensor([[0, 0], [1, 1]])

        result = dice_from_confusion(first + second)
        expected = (18 / 19 + 2 / 3) / 2
        self.assertAlmostEqual(result["macro_dice"], expected)

    def test_invalid_matrix_raises(self):
        for matrix in (
            torch.zeros(2, 3),
            torch.tensor([[1, -1], [0, 1]]),
        ):
            with self.assertRaises(ValueError):
                dice_from_confusion(matrix)

if __name__ == "__main__":
    unittest.main()
