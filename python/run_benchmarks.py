#!/usr/bin/env python3
"""Run the Phase 6 benchmark suite and write a unified machine-readable report."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import benchmark_determinism
import benchmark_generation
import benchmark_ml_summary
import run_noise_experiment
from generate_dataset import DEFAULT_SIM_RUNNER, utc_now


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "benchmarks" / "suite"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all Phase 6 benchmarks from one command.")
    parser.add_argument("--sim-runner", type=Path, default=DEFAULT_SIM_RUNNER, help="Path to sim_runner")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Benchmark suite output dir")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=("generation", "determinism", "ml", "all"),
        default=("all",),
        help="Benchmark groups to run (default: all)",
    )
    parser.add_argument("--seed-start", type=int, default=0, help="First seed for generation benchmark")
    parser.add_argument("--num-scenes", type=int, default=10, help="Scene count for generation benchmark")
    parser.add_argument("--dt", type=float, default=0.05, help="Simulation timestep")
    parser.add_argument("--ticks", type=int, default=60, help="Simulation tick count")
    parser.add_argument("--noise", choices=("low", "high"), default="low", help="Noise preset for generation benchmark")
    parser.add_argument(
        "--metadata-format",
        choices=("json", "csv"),
        default="json",
        help="Metadata format for generation and determinism benchmark",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio")
    parser.add_argument("--split-seed", type=int, default=12345, help="Split assignment seed")
    parser.add_argument("--determinism-seed", type=int, default=42, help="Seed for determinism benchmark")
    parser.add_argument("--determinism-noise", choices=("low", "high"), default="low", help="Noise preset for determinism benchmark")
    parser.add_argument("--low-dataset-dir", type=Path, default=None, help="Low-noise dataset directory for ML summary")
    parser.add_argument("--high-dataset-dir", type=Path, default=None, help="High-noise dataset directory for ML summary")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for low-vs-high experiment when ML benchmark runs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for low-vs-high experiment")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for low-vs-high experiment")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for low-vs-high experiment")
    parser.add_argument("--base-channels", type=int, default=32, help="U-Net base channels for low-vs-high experiment")
    parser.add_argument("--device", default="auto", help="Device for ML benchmark")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for ML benchmark")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count for ML benchmark")
    parser.add_argument(
        "--class-weighting",
        choices=("balanced", "none"),
        default="balanced",
        help="Loss weighting mode for ML benchmark",
    )
    parser.add_argument("--metrics-out", type=Path, default=None, help="Optional explicit suite output path")
    parser.add_argument("--force", action="store_true", help="Replace the suite output dir if it exists")
    return parser.parse_args(argv)


def normalize_benchmarks(requested: Sequence[str]) -> tuple[str, ...]:
    if "all" in requested:
        return ("generation", "determinism", "ml")
    return tuple(requested)


def validate_args(args: argparse.Namespace) -> None:
    if not args.sim_runner.is_file():
        raise SystemExit(f"sim_runner not found: {args.sim_runner}")
    if "ml" in normalize_benchmarks(args.benchmarks):
        if args.low_dataset_dir is None or args.high_dataset_dir is None:
            raise SystemExit("ML benchmark requires --low-dataset-dir and --high-dataset-dir")
        if not args.low_dataset_dir.is_dir():
            raise SystemExit(f"low-noise dataset directory not found: {args.low_dataset_dir}")
        if not args.high_dataset_dir.is_dir():
            raise SystemExit(f"high-noise dataset directory not found: {args.high_dataset_dir}")


def suite_output_path(args: argparse.Namespace) -> Path:
    if args.metrics_out is not None:
        return args.metrics_out
    return args.output_dir / "benchmark_suite.json"


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise SystemExit(f"output directory already exists: {output_dir} (use --force to replace it)")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def run_generation(args: argparse.Namespace) -> dict[str, object]:
    generation_args = SimpleNamespace(
        sim_runner=args.sim_runner,
        output_dir=args.output_dir / "generation",
        seed_start=args.seed_start,
        num_scenes=args.num_scenes,
        dt=args.dt,
        ticks=args.ticks,
        noise=args.noise,
        metadata_format=args.metadata_format,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
        metrics_out=None,
        force=False,
    )
    return benchmark_generation.benchmark(generation_args)


def run_determinism(args: argparse.Namespace) -> dict[str, object]:
    determinism_args = SimpleNamespace(
        sim_runner=args.sim_runner,
        output_dir=args.output_dir / "determinism",
        seed=args.determinism_seed,
        dt=args.dt,
        ticks=args.ticks,
        noise=args.determinism_noise,
        metadata_format=args.metadata_format,
        force=False,
        metrics_out=None,
    )
    return benchmark_determinism.benchmark(determinism_args)


def run_ml(args: argparse.Namespace) -> dict[str, object]:
    experiment_dir = args.output_dir / "noise_experiment"
    experiment_args = SimpleNamespace(
        low_dataset_dir=args.low_dataset_dir,
        high_dataset_dir=args.high_dataset_dir,
        output_dir=experiment_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        base_channels=args.base_channels,
        device=args.device,
        seed=args.seed,
        num_workers=args.num_workers,
        class_weighting=args.class_weighting,
    )
    comparison = run_noise_experiment.run_experiment(experiment_args)
    comparison_path = experiment_dir / "comparison.json"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    summary_args = SimpleNamespace(
        comparison_json=comparison_path,
        output_dir=args.output_dir / "ml_summary",
        metrics_out=None,
        force=False,
    )
    summary = benchmark_ml_summary.summarize(summary_args)
    summary_path = summary_args.output_dir / "benchmark_ml_summary.json"
    summary_args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {
        "comparison_json": str(comparison_path.resolve()),
        "comparison": comparison,
        "summary": summary,
        "summary_json": str(summary_path.resolve()),
    }


def run_suite(args: argparse.Namespace) -> dict[str, object]:
    validate_args(args)
    prepare_output_dir(args.output_dir, args.force)
    selected = normalize_benchmarks(args.benchmarks)
    results: dict[str, object] = {
        "generated_at": utc_now(),
        "sim_runner": str(args.sim_runner.resolve()),
        "selected_benchmarks": list(selected),
    }
    if "generation" in selected:
        results["generation"] = run_generation(args)
    if "determinism" in selected:
        results["determinism"] = run_determinism(args)
    if "ml" in selected:
        results["ml"] = run_ml(args)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    payload = run_suite(args)
    output_path = suite_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("benchmark_suite " + " ".join(f"ran_{name}=true" for name in payload["selected_benchmarks"]))
    print(f"wrote benchmark suite to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
