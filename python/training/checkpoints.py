"""Checkpoint save/load helpers for the ML pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from models import NUM_SEMANTIC_CLASSES, SmallUNet


def save_checkpoint(
    path: Path | str,
    *,
    model: SmallUNet,
    epoch: int,
    base_channels: int,
    metadata: dict[str, Any],
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "model_config": {
            "in_channels": 1,
            "num_classes": NUM_SEMANTIC_CLASSES,
            "base_channels": base_channels,
        },
        "metadata": metadata,
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(path: Path | str, map_location: str | torch.device = "cpu") -> tuple[SmallUNet, dict[str, Any]]:
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location=map_location)
    model_config = payload["model_config"]
    model = SmallUNet(
        in_channels=model_config["in_channels"],
        num_classes=model_config["num_classes"],
        base_channels=model_config["base_channels"],
    )
    model.load_state_dict(payload["model_state_dict"])
    return model, payload
