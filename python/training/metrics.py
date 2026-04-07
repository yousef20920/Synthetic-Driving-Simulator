"""Segmentation metrics for validation reporting."""

from __future__ import annotations

import torch
from torch import Tensor


class SegmentationMetrics:
    """Accumulates a confusion matrix and derives IoU/accuracy metrics."""

    def __init__(self, num_classes: int) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.num_classes = num_classes
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def update(self, logits: Tensor, target: Tensor) -> None:
        predictions = logits.argmax(dim=1).reshape(-1)
        target = target.reshape(-1)
        valid_mask = (target >= 0) & (target < self.num_classes)
        encoded = target[valid_mask] * self.num_classes + predictions[valid_mask]
        counts = torch.bincount(encoded, minlength=self.num_classes * self.num_classes)
        self.confusion += counts.reshape(self.num_classes, self.num_classes).cpu()

    def pixel_accuracy(self) -> float:
        total = self.confusion.sum().item()
        if total == 0:
            return 0.0
        return float(self.confusion.diag().sum().item()) / float(total)

    def per_class_iou(self) -> list[float]:
        diagonal = self.confusion.diag().to(torch.float64)
        false_positive = self.confusion.sum(dim=0).to(torch.float64) - diagonal
        false_negative = self.confusion.sum(dim=1).to(torch.float64) - diagonal
        denominator = diagonal + false_positive + false_negative
        iou = torch.zeros(self.num_classes, dtype=torch.float64)
        valid = denominator > 0
        iou[valid] = diagonal[valid] / denominator[valid]
        return [float(value.item()) for value in iou]

    def mean_iou(self) -> float:
        per_class = self.per_class_iou()
        return sum(per_class) / len(per_class)
