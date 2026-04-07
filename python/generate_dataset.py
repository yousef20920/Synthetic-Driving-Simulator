#!/usr/bin/env python3
"""Generate per-seed scene samples by invoking sim_runner."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIM_RUNNER = REPO_ROOT / "build" / "bin" / "sim_runner"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "generated"
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class SampleRecord:
    seed: int
    sample_id: str
    split: str
    directory: str
    input_image: str
    label_image: str
    metadata: str


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate raw per-scene dataset samples by running sim_runner across many seeds.",
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
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where generated scenes and manifest are written (default: %(default)s)",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First seed to generate (default: %(default)s)",
    )
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=10,
        help="Number of sequential seeds/scenes to generate (default: %(default)s)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.05,
        help="Simulation timestep in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=60,
        help="Simulation tick count per scene (default: %(default)s)",
    )
    parser.add_argument(
        "--noise",
        choices=("low", "high"),
        default="low",
        help="Noise preset to use for the noisy BEV export (default: %(default)s)",
    )
    parser.add_argument(
        "--metadata-format",
        choices=("json", "csv"),
        default="json",
        help="Metadata export format for each scene (default: %(default)s)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train split ratio (default: %(default)s)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio (default: %(default)s)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio (default: %(default)s)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=12345,
        help="Seed used to shuffle scene assignment into train/val/test splits (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the output directory first if it already exists.",
    )
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def validate_args(args: argparse.Namespace) -> None:
    if args.num_scenes <= 0:
        raise SystemExit("--num-scenes must be greater than zero")
    if args.ticks <= 0:
        raise SystemExit("--ticks must be greater than zero")
    if args.dt <= 0.0:
        raise SystemExit("--dt must be greater than zero")
    if args.seed_start < 0:
        raise SystemExit("--seed-start must be zero or greater")
    if args.split_seed < 0:
        raise SystemExit("--split-seed must be zero or greater")
    if not args.sim_runner.is_file():
        raise SystemExit(f"sim_runner not found: {args.sim_runner}")
    split_ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if any(ratio < 0.0 for ratio in split_ratios):
        raise SystemExit("split ratios must be zero or greater")
    if not math.isclose(sum(split_ratios), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise SystemExit("train/val/test ratios must sum to 1.0")


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise SystemExit(
                f"output directory already exists: {output_dir} (use --force to replace it)"
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "splits").mkdir(parents=True, exist_ok=True)
    for split in SPLIT_NAMES:
        (output_dir / split).mkdir(parents=True, exist_ok=True)


def run_command(command: Sequence[str], label: str) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        return

    detail = [f"{label} failed with exit code {result.returncode}"]
    if result.stdout.strip():
        detail.append("stdout:")
        detail.append(result.stdout.strip())
    if result.stderr.strip():
        detail.append("stderr:")
        detail.append(result.stderr.strip())
    raise RuntimeError("\n".join(detail))


def sample_name(index: int, seed: int) -> str:
    return f"sample_{index:04d}_seed_{seed:06d}"


def split_ratios(args: argparse.Namespace) -> dict[str, float]:
    return {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "test": args.test_ratio,
    }


def allocate_split_counts(num_scenes: int, ratios: dict[str, float]) -> dict[str, int]:
    exact_counts = {split: ratios[split] * num_scenes for split in SPLIT_NAMES}
    counts = {split: int(math.floor(exact_counts[split])) for split in SPLIT_NAMES}
    assigned = sum(counts.values())
    remaining = num_scenes - assigned
    priorities = sorted(
        SPLIT_NAMES,
        key=lambda split: (-(exact_counts[split] - counts[split]), SPLIT_NAMES.index(split)),
    )

    for index in range(remaining):
        counts[priorities[index]] += 1

    return counts


def assign_splits(num_scenes: int, counts: dict[str, int], split_seed: int) -> dict[int, str]:
    shuffled_indices = list(range(num_scenes))
    random.Random(split_seed).shuffle(shuffled_indices)

    assignments: dict[int, str] = {}
    cursor = 0
    for split in SPLIT_NAMES:
        next_cursor = cursor + counts[split]
        for index in shuffled_indices[cursor:next_cursor]:
            assignments[index] = split
        cursor = next_cursor
    return assignments


def generate_scene(
    *,
    sim_runner: Path,
    output_dir: Path,
    index: int,
    seed: int,
    split: str,
    dt: float,
    ticks: int,
    noise: str,
    metadata_format: str,
) -> SampleRecord:
    sample_id = sample_name(index, seed)
    sample_dir = output_dir / split / sample_id
    sample_dir.mkdir(parents=True, exist_ok=False)

    label_path = sample_dir / "label.ppm"
    input_path = sample_dir / "input.pgm"
    metadata_name = "metadata.json" if metadata_format == "json" else "metadata.csv"
    metadata_path = sample_dir / metadata_name

    base_args = [
        str(sim_runner),
        "--seed",
        str(seed),
        "--dt",
        str(dt),
        "--ticks",
        str(ticks),
    ]

    run_command(
        [*base_args, "--dump-bev-ppm", str(label_path)],
        f"{sample_id} clean BEV export",
    )
    run_command(
        [*base_args, "--noise", noise, "--dump-noisy-bev-pgm", str(input_path)],
        f"{sample_id} noisy BEV export",
    )
    metadata_flag = "--dump-metadata-json" if metadata_format == "json" else "--dump-metadata-csv"
    run_command(
        [*base_args, metadata_flag, str(metadata_path)],
        f"{sample_id} metadata export",
    )

    return SampleRecord(
        seed=seed,
        sample_id=sample_id,
        split=split,
        directory=str(sample_dir.relative_to(output_dir).as_posix()),
        input_image=str(input_path.relative_to(output_dir).as_posix()),
        label_image=str(label_path.relative_to(output_dir).as_posix()),
        metadata=str(metadata_path.relative_to(output_dir).as_posix()),
    )


def write_manifest(
    *,
    output_dir: Path,
    sim_runner: Path,
    seed_start: int,
    num_scenes: int,
    dt: float,
    ticks: int,
    noise: str,
    metadata_format: str,
    split_seed: int,
    ratios: dict[str, float],
    split_counts: dict[str, int],
    records: Sequence[SampleRecord],
) -> Path:
    manifest = {
        "generated_at": utc_now(),
        "sim_runner": str(sim_runner.resolve()),
        "config": {
            "seed_start": seed_start,
            "num_scenes": num_scenes,
            "dt": dt,
            "ticks": ticks,
            "noise": noise,
            "metadata_format": metadata_format,
            "split_seed": split_seed,
            "split_ratios": ratios,
        },
        "splits": {
            split: {
                "count": split_counts[split],
                "index_path": f"splits/{split}.json",
            }
            for split in SPLIT_NAMES
        },
        "samples": [asdict(record) for record in records],
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def write_split_indices(output_dir: Path, records: Sequence[SampleRecord]) -> None:
    for split in SPLIT_NAMES:
        split_records = [asdict(record) for record in records if record.split == split]
        index_path = output_dir / "splits" / f"{split}.json"
        payload = {
            "split": split,
            "sample_count": len(split_records),
            "samples": split_records,
        }
        index_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    validate_args(args)
    prepare_output_dir(args.output_dir, args.force)

    ratios = split_ratios(args)
    counts = allocate_split_counts(args.num_scenes, ratios)
    assignments = assign_splits(args.num_scenes, counts, args.split_seed)

    records = []
    for index in range(args.num_scenes):
        seed = args.seed_start + index
        split = assignments[index]
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
        records.append(record)
        print(
            f"[{index + 1}/{args.num_scenes}] generated {record.sample_id} in {record.split} "
            f"({record.directory})"
        )

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
    print(f"Wrote dataset manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
