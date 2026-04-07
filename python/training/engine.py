"""Shared train/eval helpers for the U-Net pipeline."""

from __future__ import annotations

import random

import torch
from torch import nn
from torch.utils.data import DataLoader

from models import NUM_SEMANTIC_CLASSES, SEMANTIC_CLASS_NAMES
from training.metrics import SegmentationMetrics


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

    return total_loss / max(total_batches, 1)


def run_eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    prefix: str,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    metrics = SegmentationMetrics(NUM_SEMANTIC_CLASSES)

    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)

            total_loss += float(loss.item())
            total_batches += 1
            metrics.update(logits.cpu(), targets.cpu())

    per_class_iou_values = metrics.per_class_iou()
    per_class_iou = {
        class_name: per_class_iou_values[index]
        for index, class_name in enumerate(SEMANTIC_CLASS_NAMES)
    }
    return {
        f"{prefix}_loss": total_loss / max(total_batches, 1),
        f"{prefix}_pixel_accuracy": metrics.pixel_accuracy(),
        f"{prefix}_mean_iou": metrics.mean_iou(),
        f"{prefix}_per_class_iou": per_class_iou,
    }
