from __future__ import annotations

import math
from collections import Counter
from typing import Literal, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def inverse_class_frequency_alpha(
    counts: Counter[int] | dict[int, int],
    *,
    num_classes: int,
    max_ratio: float = 5.0,
) -> list[float]:
    """Per-class focal weights alpha_t from inverse training frequency.

    Weights are normalized so the smallest class weight is 1.0 and the largest
    is capped at ``max_ratio`` to avoid single-class collapse on tiny minorities.
    """
    total = sum(counts.values()) if counts else 0
    if total <= 0:
        return [1.0] * num_classes
    raw_weights: list[float] = []
    for label_id in range(num_classes):
        count = max(int(counts.get(label_id, 0)), 1)
        raw_weights.append(total / (num_classes * count))
    base = min(raw_weights)
    if base <= 0:
        return [1.0] * num_classes
    cap = max(float(max_ratio), 1.0)
    return [min(weight / base, cap) for weight in raw_weights]


class FocalLoss(nn.Module):
    """Multi-class focal loss with per-sample alpha_t and optional label smoothing.

    Hard-label focal loss:
      FL = -alpha_y * (1 - p_t)^gamma * log(p_t)

  With label smoothing, cross-entropy uses soft targets while the focal modulation
  uses the smoothed probability mass on the target distribution:
      p_t = sum_c q_c * p_c
    """

    def __init__(
        self,
        *,
        num_classes: int,
        gamma: float = 1.5,
        alpha: Tensor | Sequence[float] | None = None,
        label_smoothing: float = 0.05,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        self.num_classes = num_classes
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction
        if alpha is not None:
            alpha_tensor = torch.as_tensor(alpha, dtype=torch.float32)
            if alpha_tensor.numel() != num_classes:
                raise ValueError(f"alpha must have {num_classes} values, got {alpha_tensor.numel()}")
            self.register_buffer("alpha", alpha_tensor)
        else:
            self.alpha = None

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        if logits.dim() != 2:
            raise ValueError("logits must be 2D [batch, num_classes]")
        if targets.dim() != 1:
            raise ValueError("targets must be 1D [batch]")
        if logits.size(-1) != self.num_classes:
            raise ValueError(f"expected {self.num_classes} logits, got {logits.size(-1)}")

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        if self.label_smoothing > 0.0:
            true_dist = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1.0)
            soft_targets = true_dist * (1.0 - self.label_smoothing) + self.label_smoothing / self.num_classes
            cross_entropy = -(soft_targets * log_probs).sum(dim=-1)
            pt = (soft_targets * probs).sum(dim=-1)
        else:
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            cross_entropy = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - pt).clamp_min(1e-6).pow(self.gamma)
        loss = focal_weight * cross_entropy
        if self.alpha is not None:
            alpha = self.alpha.to(device=logits.device, dtype=logits.dtype)
            loss = loss * alpha[targets]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def is_collapsed_predictions(predictions: Sequence[int], *, num_classes: int, threshold: float = 0.95) -> bool:
    """Return True when one class accounts for >= threshold of predictions."""
    if not predictions:
        return False
    counts = Counter(predictions)
    dominant = max(counts.values())
    return dominant / len(predictions) >= threshold


def minimum_per_label_recall(
    metrics: dict[str, object],
    *,
    labels: Sequence[str],
    min_support: int = 1,
) -> float:
    per_label = metrics.get("per_label", {})
    if not isinstance(per_label, dict):
        return 0.0
    recalls: list[float] = []
    for label in labels:
        entry = per_label.get(label, {})
        if not isinstance(entry, dict):
            continue
        support = int(entry.get("support", 0))
        if support < min_support:
            continue
        recalls.append(float(entry.get("recall", 0.0)))
    if not recalls:
        return 1.0
    return min(recalls)
