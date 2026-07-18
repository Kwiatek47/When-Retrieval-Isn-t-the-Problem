from __future__ import annotations

import unittest
from collections import Counter

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

from scripts.classifier.losses import (
    FocalLoss,
    inverse_class_frequency_alpha,
    is_collapsed_predictions,
    minimum_per_label_recall,
)


@unittest.skipIf(torch is None, "torch is required for classifier loss tests")
class ClassifierLossTests(unittest.TestCase):
    def test_inverse_class_frequency_alpha_caps_extreme_minority_weight(self) -> None:
        counts = Counter({0: 900, 1: 90, 2: 10})
        alpha = inverse_class_frequency_alpha(counts, num_classes=3, max_ratio=5.0)
        self.assertAlmostEqual(min(alpha), 1.0)
        self.assertLessEqual(max(alpha), 5.0)

    def test_focal_loss_mean_reduction_shape(self) -> None:
        logits = torch.randn(4, 3, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 1])
        loss = FocalLoss(num_classes=3, gamma=1.5, label_smoothing=0.05)(logits, targets)
        self.assertEqual(loss.shape, torch.Size([]))
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_focal_loss_uses_true_class_alpha_only(self) -> None:
        logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], requires_grad=False)
        targets = torch.tensor([0, 1])
        alpha = [1.0, 1.0, 8.0]
        loss_high_maybe_alpha = FocalLoss(
            num_classes=3,
            gamma=0.0,
            alpha=alpha,
            label_smoothing=0.0,
        )(logits, targets)
        loss_uniform_alpha = FocalLoss(
            num_classes=3,
            gamma=0.0,
            alpha=[1.0, 1.0, 1.0],
            label_smoothing=0.0,
        )(logits, targets)
        self.assertTrue(torch.allclose(loss_high_maybe_alpha, loss_uniform_alpha, atol=1e-6))

    def test_gamma_zero_matches_manual_smoothed_cross_entropy(self) -> None:
        logits = torch.tensor([[1.2, 0.3, -0.4], [0.1, 1.0, 0.2]])
        targets = torch.tensor([0, 1])
        alpha = [1.0, 2.0, 3.0]
        loss_fn = FocalLoss(
            num_classes=3,
            gamma=0.0,
            alpha=alpha,
            label_smoothing=0.05,
        )
        actual = loss_fn(logits, targets)

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        smoothing = 0.05
        soft_targets = torch.zeros_like(logits)
        soft_targets.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
        soft_targets += smoothing / 3
        ce = -(soft_targets * log_probs).sum(dim=-1)
        alpha_tensor = torch.tensor(alpha, dtype=logits.dtype)
        expected = (alpha_tensor[targets] * ce).mean()
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_collapse_detection(self) -> None:
        predictions = [2] * 90
        self.assertTrue(is_collapsed_predictions(predictions, num_classes=3))
        self.assertFalse(is_collapsed_predictions([0, 1, 2, 0, 1, 2], num_classes=3))

    def test_minimum_per_label_recall_skips_tiny_support(self) -> None:
        metrics = {
            "per_label": {
                "yes": {"support": 200, "recall": 0.8},
                "no": {"support": 200, "recall": 0.5},
                "maybe": {"support": 5, "recall": 0.0},
            }
        }
        self.assertAlmostEqual(
            minimum_per_label_recall(metrics, labels=("yes", "no", "maybe"), min_support=10),
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
