#!/usr/bin/env python3
"""Benchmark dataset generation throughput and per-scene latency."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter
from typing import Sequence

from generate_dataset import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SIM_RUNNER,
    allocate_split_counts,
    assign_splits,
    generate_scene,
    parse_args as parse_dataset_args,
    prepare_output_dir,
    split_ratios,
    utc_now,
    validate_args as validate_dataset_args,
    write_manifest,
    write_split_indices,
)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark dataset generation time per sample and total scenes/sec.",
    )
    parser.add_argument(
        "--sim-runner",
        type=Path,
        default=DEFAULT_SIM_RUNNER,
        help="Path to the sim_runner binary (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "benchmarks" / "generation",
        help="Directory where generated benchmark artifacts are written (default: %(default)s)",
    )
    parser.add_argument("--seed-start", type=int, default=0, help="First seed to benchmark")
    parser.add_argument("--num-scenes", type=int, default=10, help="Number of scenes to generate")
    parser.add_argument("--dt", type=float, default=0.05, help="Simulation timestep in seconds")
    parser.add_argument("--ticks", type=int, default=60, help="Simulation tick count per scene")
    parser.add_argument(
        "--noise",
        choices=("low", "high"),
        default="low",
        help="Noise preset used for noisy input export",
    )
    parser.add_argument(
        "--metadata-format",
        choices=("json", "csv"),
        default="json",
        help="Metadata export format for each scene",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio")
    parser.add_argument("--split-seed", type=int, default=12345, help="Split assignment seed")
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Optional explicit benchmark metrics output path",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the output directory first if it already exists.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    dataset_args = parse_dataset_args([])
    dataset_args.sim_runner = args.sim_runner
    dataset_args.output_dir = args.output_dir
    dataset_args.seed_start = args.seed_start
    dataset_args.num_scenes = args.num_scenes
    dataset_args.dt = args.dt
    dataset_args.ticks = args.ticks
    dataset_args.noise = args.noise
    dataset_args.metadata_format = args.metadata_format
    dataset_args.train_ratio = args.train_ratio
    dataset_args.val_ratio = args.val_ratio
    dataset_args.test_ratio = args.test_ratio
    dataset_args.split_seed = args.split_seed
    dataset_args.force = args.force
    validate_dataset_args(dataset_args)


def metrics_path(args: argparse.Namespace) -> Path:
    if args.metrics_out is not None:
        return args.metrics_out
    return args.output_dir / "benchmark_generation.json"


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty value list")
    if len(values) == 1:
        return values[0]

    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def benchmark(args: argparse.Namespace) -> dict[str, object]:
    validate_args(args)
    prepare_output_dir(args.output_dir, args.force)

    ratios = split_ratios(args)
    counts = allocate_split_counts(args.num_scenes, ratios)
    assignments = assign_splits(args.num_scenes, counts, args.split_seed)

    records = []
    scene_timings = []
    total_start = perf_counter()
    for index in range(args.num_scenes):
        seed = args.seed_start + index
        split = assignments[index]
        scene_start = perf_counter()
        record = generate_scene(
            sim_runner=args.sim_runner,
            output_dir=args.output_dir,
            index=index,
            seed=seed,
            split=split,
            dt=args.dt,
            ticks=args.ticks,
            noise=args.noise,
            metadata_format=args.metadata_format,
        )
        elapsed = perf_counter() - scene_start
        records.append(record)
        scene_timings.append(
            {
                "index": index,
                "seed": seed,
                "sample_id": record.sample_id,
                "split": split,
                "duration_seconds": elapsed,
            }
        )
        print(
            f"[{index + 1}/{args.num_scenes}] "
            f"{record.sample_id} duration_seconds={elapsed:.4f} split={split}"
        )

    total_duration = perf_counter() - total_start
    write_split_indices(args.output_dir, records)
    manifest_path = write_manifest(
        output_dir=args.output_dir,
        sim_runner=args.sim_runner,
        seed_start=args.seed_start,
        num_scenes=args.num_scenes,
        dt=args.dt,
        ticks=args.ticks,
        noise=args.noise,
        metadata_format=args.metadata_format,
        split_seed=args.split_seed,
        ratios=ratios,
        split_counts=counts,
        records=records,
    )

    durations = [entry["duration_seconds"] for entry in scene_timings]
    return {
        "generated_at": utc_now(),
        "sim_runner": str(args.sim_runner.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "dataset_manifest": str(manifest_path.resolve()),
        "config": {
            "seed_start": args.seed_start,
            "num_scenes": args.num_scenes,
            "dt": args.dt,
            "ticks": args.ticks,
            "noise": args.noise,
            "metadata_format": args.metadata_format,
            "split_seed": args.split_seed,
            "split_ratios": ratios,
        },
        "summary": {
            "scene_count": args.num_scenes,
            "total_duration_seconds": total_duration,
            "mean_scene_duration_seconds": statistics.fmean(durations),
            "median_scene_duration_seconds": statistics.median(durations),
            "p95_scene_duration_seconds": percentile(durations, 0.95),
            "min_scene_duration_seconds": min(durations),
            "max_scene_duration_seconds": max(durations),
            "scenes_per_second": args.num_scenes / total_duration,
        },
        "scene_timings": scene_timings,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    results = benchmark(args)
    output_path = metrics_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    summary = results["summary"]
    print(
        "generation_benchmark "
        f"scenes={summary['scene_count']} "
        f"total_seconds={summary['total_duration_seconds']:.4f} "
        f"mean_scene_seconds={summary['mean_scene_duration_seconds']:.4f} "
        f"scenes_per_second={summary['scenes_per_second']:.4f}"
    )
    print(f"wrote benchmark to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
