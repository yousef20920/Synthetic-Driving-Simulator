#!/usr/bin/env python3
"""Benchmark end-to-end determinism across exported simulator artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from generate_dataset import DEFAULT_SIM_RUNNER, utc_now


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "benchmarks" / "determinism"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that repeated simulator exports are byte-identical for a fixed seed.",
    )
    parser.add_argument("--sim-runner", type=Path, default=DEFAULT_SIM_RUNNER, help="Path to sim_runner")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Seed to verify")
    parser.add_argument("--dt", type=float, default=0.05, help="Simulation timestep in seconds")
    parser.add_argument("--ticks", type=int, default=60, help="Simulation tick count")
    parser.add_argument("--noise", choices=("low", "high"), default="low", help="Noise preset")
    parser.add_argument(
        "--metadata-format",
        choices=("json", "csv"),
        default="json",
        help="Metadata export format",
    )
    parser.add_argument("--force", action="store_true", help="Replace the output directory if it exists")
    parser.add_argument("--metrics-out", type=Path, default=None, help="Optional explicit metrics output path")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not args.sim_runner.is_file():
        raise SystemExit(f"sim_runner not found: {args.sim_runner}")
    if args.seed < 0:
        raise SystemExit("--seed must be zero or greater")
    if args.dt <= 0.0:
        raise SystemExit("--dt must be greater than zero")
    if args.ticks <= 0:
        raise SystemExit("--ticks must be greater than zero")


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise SystemExit(f"output directory already exists: {output_dir} (use --force to replace it)")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def metrics_path(args: argparse.Namespace) -> Path:
    if args.metrics_out is not None:
        return args.metrics_out
    return args.output_dir / "benchmark_determinism.json"


def run_command(command: list[str], label: str) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        return

    detail = [f"{label} failed with exit code {result.returncode}"]
    if result.stdout.strip():
        detail.extend(["stdout:", result.stdout.strip()])
    if result.stderr.strip():
        detail.extend(["stderr:", result.stderr.strip()])
    raise RuntimeError("\n".join(detail))


def export_bundle(run_dir: Path, args: argparse.Namespace) -> dict[str, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "csv": run_dir / "actor_log.csv",
        "clean_bev": run_dir / "clean_bev.ppm",
        "noisy_bev": run_dir / "noisy_bev.pgm",
        "metadata": run_dir / ("metadata.json" if args.metadata_format == "json" else "metadata.csv"),
    }
    base_args = [
        str(args.sim_runner),
        "--seed",
        str(args.seed),
        "--dt",
        str(args.dt),
        "--ticks",
        str(args.ticks),
    ]
    run_command([*base_args, "--out", str(artifacts["csv"])], "csv export")
    run_command([*base_args, "--dump-bev-ppm", str(artifacts["clean_bev"])], "clean BEV export")
    run_command(
        [*base_args, "--noise", args.noise, "--dump-noisy-bev-pgm", str(artifacts["noisy_bev"])],
        "noisy BEV export",
    )
    metadata_flag = "--dump-metadata-json" if args.metadata_format == "json" else "--dump-metadata-csv"
    run_command([*base_args, metadata_flag, str(artifacts["metadata"])], "metadata export")
    return artifacts


def sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_artifacts(run_a: dict[str, Path], run_b: dict[str, Path]) -> dict[str, object]:
    comparisons: dict[str, object] = {}
    all_identical = True
    for artifact_name in run_a:
        sha_a = sha256_bytes(run_a[artifact_name])
        sha_b = sha256_bytes(run_b[artifact_name])
        identical = sha_a == sha_b and run_a[artifact_name].read_bytes() == run_b[artifact_name].read_bytes()
        comparisons[artifact_name] = {
            "run_a_path": str(run_a[artifact_name]),
            "run_b_path": str(run_b[artifact_name]),
            "run_a_size_bytes": run_a[artifact_name].stat().st_size,
            "run_b_size_bytes": run_b[artifact_name].stat().st_size,
            "run_a_sha256": sha_a,
            "run_b_sha256": sha_b,
            "identical": identical,
        }
        all_identical = all_identical and identical
    return {
        "all_identical": all_identical,
        "artifact_count": len(run_a),
        "artifacts": comparisons,
    }


def benchmark(args: argparse.Namespace) -> dict[str, object]:
    validate_args(args)
    prepare_output_dir(args.output_dir, args.force)
    run_a = export_bundle(args.output_dir / "run_a", args)
    run_b = export_bundle(args.output_dir / "run_b", args)
    comparison = compare_artifacts(run_a, run_b)
    return {
        "generated_at": utc_now(),
        "sim_runner": str(args.sim_runner.resolve()),
        "config": {
            "seed": args.seed,
            "dt": args.dt,
            "ticks": args.ticks,
            "noise": args.noise,
            "metadata_format": args.metadata_format,
        },
        "summary": {
            "all_identical": comparison["all_identical"],
            "artifact_count": comparison["artifact_count"],
        },
        "artifacts": comparison["artifacts"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    payload = benchmark(args)
    output_path = metrics_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        "determinism_benchmark "
        f"seed={payload['config']['seed']} "
        f"artifacts={payload['summary']['artifact_count']} "
        f"all_identical={str(payload['summary']['all_identical']).lower()}"
    )
    print(f"wrote benchmark to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
