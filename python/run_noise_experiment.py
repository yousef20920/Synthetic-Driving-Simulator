#!/usr/bin/env python3
"""Compare model performance across low-noise and high-noise datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import eval_model
import train_unet


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run low-noise vs high-noise U-Net training/evaluation.")
    parser.add_argument("--low-dataset-dir", type=Path, required=True, help="Low-noise dataset directory")
    parser.add_argument("--high-dataset-dir", type=Path, required=True, help="High-noise dataset directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for experiment outputs")
    parser.add_argument("--epochs", type=int, default=5, help="Epoch count for both runs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for both runs")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for both runs")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for both runs")
    parser.add_argument("--base-channels", type=int, default=32, help="U-Net base channel width")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, mps")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for both runs")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument(
        "--class-weighting",
        choices=("balanced", "none"),
        default="balanced",
        help="Class weighting mode forwarded to training",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not args.low_dataset_dir.is_dir():
        raise SystemExit(f"low-noise dataset directory not found: {args.low_dataset_dir}")
    if not args.high_dataset_dir.is_dir():
        raise SystemExit(f"high-noise dataset directory not found: {args.high_dataset_dir}")
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


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _train_args(dataset_dir: Path, run_dir: Path, args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_dir=dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        base_channels=args.base_channels,
        device=args.device,
        seed=args.seed,
        num_workers=args.num_workers,
        class_weighting=args.class_weighting,
        metrics_out=run_dir / "train_metrics.json",
        checkpoint_out=run_dir / "model.pt",
    )


def _eval_args(dataset_dir: Path, run_dir: Path, args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_dir=dataset_dir,
        checkpoint=run_dir / "model.pt",
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
        metrics_out=run_dir / "eval_metrics.json",
    )


def run_condition(label: str, dataset_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    run_dir = args.output_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)

    train_args = _train_args(dataset_dir, run_dir, args)
    train_result = train_unet.train(train_args)
    _write_json(train_args.metrics_out, train_result)

    eval_args = _eval_args(dataset_dir, run_dir, args)
    eval_result = eval_model.evaluate(eval_args)
    _write_json(eval_args.metrics_out, eval_result)

    return {
        "dataset_dir": str(dataset_dir),
        "train_metrics_path": str(train_args.metrics_out),
        "checkpoint_path": str(train_args.checkpoint_out),
        "eval_metrics_path": str(eval_args.metrics_out),
        "train": train_result,
        "eval": eval_result,
    }


def build_comparison(low_result: dict[str, object], high_result: dict[str, object]) -> dict[str, object]:
    low_eval = low_result["eval"]
    high_eval = high_result["eval"]
    low_train = low_result["train"]
    high_train = high_result["train"]

    return {
        "best_val_mean_iou_delta_low_minus_high": low_train["best_val_mean_iou"] - high_train["best_val_mean_iou"],
        "test_mean_iou_delta_low_minus_high": low_eval["test_mean_iou"] - high_eval["test_mean_iou"],
        "test_pixel_accuracy_delta_low_minus_high": low_eval["test_pixel_accuracy"] - high_eval["test_pixel_accuracy"],
    }


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    low_result = run_condition("low_noise", args.low_dataset_dir, args)
    high_result = run_condition("high_noise", args.high_dataset_dir, args)
    comparison = build_comparison(low_result, high_result)
    return {
        "low_noise": low_result,
        "high_noise": high_result,
        "comparison": comparison,
    }


def log_summary(results: dict[str, object]) -> None:
    comparison = results["comparison"]
    print(
        "low_vs_high "
        f"delta_best_val_mean_iou={comparison['best_val_mean_iou_delta_low_minus_high']:.4f} "
        f"delta_test_mean_iou={comparison['test_mean_iou_delta_low_minus_high']:.4f} "
        f"delta_test_pixel_accuracy={comparison['test_pixel_accuracy_delta_low_minus_high']:.4f}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    results = run_experiment(args)
    comparison_path = args.output_dir / "comparison.json"
    _write_json(comparison_path, results)
    log_summary(results)
    print(f"wrote comparison to {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
