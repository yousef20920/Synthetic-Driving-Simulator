#!/usr/bin/env python3
"""Train the baseline U-Net on the generated BEV dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from models import NUM_SEMANTIC_CLASSES, SEMANTIC_CLASS_NAMES, SmallUNet
from training import (
    SplitBevDataset,
    resolve_device,
    run_eval_epoch,
    run_train_epoch,
    save_checkpoint,
    set_seed,
)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the baseline U-Net on generated BEV data.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Dataset root directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay")
    parser.add_argument("--base-channels", type=int, default=32, help="U-Net base channel width")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, mps")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument(
        "--class-weighting",
        choices=("balanced", "none"),
        default="balanced",
        help="Class weighting mode for the training loss",
    )
    parser.add_argument("--metrics-out", type=Path, default=None, help="Optional JSON metrics output path")
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        default=None,
        help="Optional path to write the best checkpoint",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise SystemExit("--epochs must be greater than zero")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than zero")
    if args.learning_rate <= 0.0:
        raise SystemExit("--learning-rate must be greater than zero")
    if args.base_channels <= 0:
        raise SystemExit("--base-channels must be greater than zero")
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be zero or greater")
    if not args.dataset_dir.is_dir():
        raise SystemExit(f"dataset directory not found: {args.dataset_dir}")


def create_dataloaders(
    args: argparse.Namespace,
) -> tuple[SplitBevDataset, SplitBevDataset, DataLoader, DataLoader]:
    train_dataset = SplitBevDataset(args.dataset_dir, "train")
    val_dataset = SplitBevDataset(args.dataset_dir, "val")
    if len(train_dataset) == 0:
        raise SystemExit("train split is empty")
    if len(val_dataset) == 0:
        raise SystemExit("val split is empty")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def create_loss(
    args: argparse.Namespace,
    train_dataset: SplitBevDataset,
    device: torch.device,
) -> tuple[nn.Module, dict[str, float] | None]:
    if args.class_weighting == "none":
        return nn.CrossEntropyLoss(), None

    weights = train_dataset.balanced_class_weights().to(device)
    class_weights = {
        class_name: float(weight)
        for class_name, weight in zip(SEMANTIC_CLASS_NAMES, weights.detach().cpu().tolist())
    }
    return nn.CrossEntropyLoss(weight=weights), class_weights


def log_class_weighting(mode: str, class_weights: dict[str, float] | None) -> None:
    if class_weights is None:
        print("class_weighting=none")
        return

    rendered_weights = " ".join(
        f"weight_{class_name}={class_weights[class_name]:.4f}" for class_name in SEMANTIC_CLASS_NAMES
    )
    print(f"class_weighting={mode} {rendered_weights}")


def log_epoch(epoch: int, total_epochs: int, metrics: dict[str, object]) -> None:
    per_class = metrics["val_per_class_iou"]
    class_metrics = " ".join(
        f"iou_{class_name}={per_class[class_name]:.4f}" for class_name in SEMANTIC_CLASS_NAMES
    )
    print(
        f"epoch {epoch}/{total_epochs} "
        f"train_loss={metrics['train_loss']:.4f} "
        f"val_loss={metrics['val_loss']:.4f} "
        f"val_pixel_accuracy={metrics['val_pixel_accuracy']:.4f} "
        f"val_mean_iou={metrics['val_mean_iou']:.4f} "
        f"{class_metrics}"
    )


def train(args: argparse.Namespace) -> dict[str, object]:
    validate_args(args)
    set_seed(args.seed)
    device = resolve_device(args.device)
    train_dataset, _val_dataset, train_loader, val_loader = create_dataloaders(args)

    model = SmallUNet(base_channels=args.base_channels).to(device)
    criterion, class_weights = create_loss(args, train_dataset, device)
    log_class_weighting(args.class_weighting, class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history = []
    best_epoch_metrics: dict[str, object] | None = None
    for epoch in range(1, args.epochs + 1):
        train_loss = run_train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = run_eval_epoch(model, val_loader, criterion, device, prefix="val")
        epoch_metrics = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(epoch_metrics)
        log_epoch(epoch, args.epochs, epoch_metrics)

        if best_epoch_metrics is None or epoch_metrics["val_mean_iou"] > best_epoch_metrics["val_mean_iou"]:
            best_epoch_metrics = epoch_metrics
            if args.checkpoint_out is not None:
                save_checkpoint(
                    args.checkpoint_out,
                    model=model,
                    epoch=epoch,
                    base_channels=args.base_channels,
                    metadata={
                        "dataset_dir": str(args.dataset_dir),
                        "device": str(device),
                        "learning_rate": args.learning_rate,
                        "batch_size": args.batch_size,
                        "weight_decay": args.weight_decay,
                        "class_weighting": args.class_weighting,
                        "class_weights": class_weights,
                        "best_val_mean_iou": epoch_metrics["val_mean_iou"],
                    },
                )

    assert best_epoch_metrics is not None
    return {
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "class_weighting": args.class_weighting,
        "class_weights": class_weights,
        "history": history,
        "best_epoch": best_epoch_metrics["epoch"],
        "best_val_mean_iou": best_epoch_metrics["val_mean_iou"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    result = train(args)
    if args.metrics_out is not None:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"wrote metrics to {args.metrics_out}")
    if args.checkpoint_out is not None:
        print(f"wrote checkpoint to {args.checkpoint_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
