#!/usr/bin/env python3
"""Summarize low-noise vs high-noise ML metrics into a benchmark report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from generate_dataset import utc_now


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPARISON_JSON = REPO_ROOT / "outputs" / "noise_experiment" / "comparison.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "benchmarks" / "ml_summary"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a benchmark summary table from the low-vs-high noise experiment output.",
    )
    parser.add_argument(
        "--comparison-json",
        type=Path,
        default=DEFAULT_COMPARISON_JSON,
        help="Phase 5 comparison JSON to summarize",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the summary report is written",
    )
    parser.add_argument("--metrics-out", type=Path, default=None, help="Optional explicit output path")
    parser.add_argument("--force", action="store_true", help="Replace the output directory if it exists")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not args.comparison_json.is_file():
        raise SystemExit(f"comparison json not found: {args.comparison_json}")


def metrics_path(args: argparse.Namespace) -> Path:
    if args.metrics_out is not None:
        return args.metrics_out
    return args.output_dir / "benchmark_ml_summary.json"


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists() and force:
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def make_row(noise_label: str, payload: dict[str, object]) -> dict[str, object]:
    train = payload["train"]
    eval_metrics = payload["eval"]
    return {
        "noise_level": noise_label,
        "best_val_mean_iou": train["best_val_mean_iou"],
        "test_mean_iou": eval_metrics["test_mean_iou"],
        "test_pixel_accuracy": eval_metrics["test_pixel_accuracy"],
        "test_per_class_iou": eval_metrics["test_per_class_iou"],
    }


def format_table(rows: list[dict[str, object]]) -> str:
    header = (
        "| noise | best_val_mean_iou | test_mean_iou | test_pixel_accuracy | "
        "iou_drivable | iou_lane | iou_vehicle | iou_pedestrian | iou_obstacle |"
    )
    separator = "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, separator]
    for row in rows:
        per_class = row["test_per_class_iou"]
        lines.append(
            "| "
            f"{row['noise_level']} | "
            f"{row['best_val_mean_iou']:.4f} | "
            f"{row['test_mean_iou']:.4f} | "
            f"{row['test_pixel_accuracy']:.4f} | "
            f"{per_class['drivable']:.4f} | "
            f"{per_class['lane']:.4f} | "
            f"{per_class['vehicle']:.4f} | "
            f"{per_class['pedestrian']:.4f} | "
            f"{per_class['obstacle']:.4f} |"
        )
    return "\n".join(lines)


def summarize(args: argparse.Namespace) -> dict[str, object]:
    validate_args(args)
    comparison = json.loads(args.comparison_json.read_text(encoding="utf-8"))
    rows = [
        make_row("low_noise", comparison["low_noise"]),
        make_row("high_noise", comparison["high_noise"]),
    ]
    return {
        "generated_at": utc_now(),
        "source_comparison_json": str(args.comparison_json.resolve()),
        "rows": rows,
        "deltas": comparison["comparison"],
        "markdown_table": format_table(rows),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    prepare_output_dir(args.output_dir, args.force)
    payload = summarize(args)
    output_path = metrics_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(payload["markdown_table"])
    print(f"wrote benchmark to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
