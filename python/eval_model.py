#!/usr/bin/env python3
"""Evaluate a saved U-Net checkpoint on the test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from torch import nn
from torch.utils.data import DataLoader

from training import SplitBevDataset, load_checkpoint, resolve_device, run_eval_epoch


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained U-Net checkpoint on the test split.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Dataset root directory")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint to evaluate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, mps")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument("--metrics-out", type=Path, default=None, help="Optional JSON metrics output path")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset_dir.is_dir():
        raise SystemExit(f"dataset directory not found: {args.dataset_dir}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than zero")
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be zero or greater")


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    validate_args(args)
    device = resolve_device(args.device)
    dataset = SplitBevDataset(args.dataset_dir, "test")
    if len(dataset) == 0:
        raise SystemExit("test split is empty")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model, payload = load_checkpoint(args.checkpoint, map_location=device)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    metrics = run_eval_epoch(model, loader, criterion, device, prefix="test")
    return {
        "checkpoint": str(args.checkpoint),
        "dataset_dir": str(args.dataset_dir),
        "device": str(device),
        "checkpoint_epoch": payload["epoch"],
        **metrics,
    }


def log_metrics(metrics: dict[str, object]) -> None:
    per_class = metrics["test_per_class_iou"]
    class_metrics = " ".join(
        f"iou_{class_name}={per_class[class_name]:.4f}" for class_name in sorted(per_class)
    )
    print(
        f"test_loss={metrics['test_loss']:.4f} "
        f"test_pixel_accuracy={metrics['test_pixel_accuracy']:.4f} "
        f"test_mean_iou={metrics['test_mean_iou']:.4f} "
        f"{class_metrics}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    metrics = evaluate(args)
    log_metrics(metrics)
    if args.metrics_out is not None:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(f"wrote metrics to {args.metrics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
