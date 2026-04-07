from __future__ import annotations

import json
from pathlib import Path


CLASS_COLORS = {
    0: (96, 99, 88),
    1: (245, 208, 97),
    2: (190, 74, 47),
    3: (15, 124, 115),
    4: (43, 33, 23),
}


def write_pgm(path: Path, pixels: list[list[int]]) -> None:
    height = len(pixels)
    width = len(pixels[0])
    rows = [" ".join(str(value) for value in row) for row in pixels]
    path.write_text(f"P2\n{width} {height}\n255\n" + "\n".join(rows) + "\n", encoding="utf-8")


def write_ppm_labels(path: Path, labels: list[list[int]]) -> None:
    height = len(labels)
    width = len(labels[0])
    rows = []
    for row in labels:
        rows.append(" ".join(f"{r} {g} {b}" for r, g, b in (CLASS_COLORS[value] for value in row)))
    path.write_text(f"P3\n{width} {height}\n255\n" + "\n".join(rows) + "\n", encoding="utf-8")


def create_tiny_dataset(dataset_dir: Path, *, image_size: int = 16) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "splits").mkdir(exist_ok=True)
    samples = []
    split_counts = {"train": 0, "val": 0, "test": 0}
    split_plan = {"train": 4, "val": 2, "test": 1}
    sample_index = 0

    for split, count in split_plan.items():
        split_samples = []
        for local_index in range(count):
            sample_id = f"sample_{sample_index:04d}_seed_{100 + sample_index:06d}"
            sample_dir = dataset_dir / split / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            input_pixels = []
            label_pixels = []
            for row in range(image_size):
                input_row = []
                label_row = []
                for col in range(image_size):
                    label = (row // max(image_size // 5, 1) + col // max(image_size // 5, 1) + sample_index) % 5
                    label_row.append(label)
                    input_row.append((label * 40 + row + col) % 256)
                input_pixels.append(input_row)
                label_pixels.append(label_row)

            input_path = sample_dir / "input.pgm"
            label_path = sample_dir / "label.ppm"
            metadata_path = sample_dir / "metadata.json"
            write_pgm(input_path, input_pixels)
            write_ppm_labels(label_path, label_pixels)
            metadata_path.write_text(
                json.dumps({"seed": 100 + sample_index, "tick": 9, "actor_count": 8}) + "\n",
                encoding="utf-8",
            )

            record = {
                "seed": 100 + sample_index,
                "sample_id": sample_id,
                "split": split,
                "directory": f"{split}/{sample_id}",
                "input_image": f"{split}/{sample_id}/input.pgm",
                "label_image": f"{split}/{sample_id}/label.ppm",
                "metadata": f"{split}/{sample_id}/metadata.json",
            }
            split_samples.append(record)
            samples.append(record)
            split_counts[split] += 1
            sample_index += 1

        index_payload = {"split": split, "sample_count": len(split_samples), "samples": split_samples}
        (dataset_dir / "splits" / f"{split}.json").write_text(
            json.dumps(index_payload, indent=2) + "\n",
            encoding="utf-8",
        )

    manifest = {
        "config": {
            "seed_start": 100,
            "num_scenes": sample_index,
            "dt": 0.05,
            "ticks": 10,
            "noise": "low",
            "metadata_format": "json",
            "split_seed": 123,
            "split_ratios": {"train": 4 / 7, "val": 2 / 7, "test": 1 / 7},
        },
        "splits": {
            split: {"count": split_counts[split], "index_path": f"splits/{split}.json"}
            for split in ("train", "val", "test")
        },
        "samples": samples,
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def create_imbalanced_dataset(dataset_dir: Path, *, image_size: int = 16) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "splits").mkdir(exist_ok=True)
    split_plan = {"train": 4, "val": 2, "test": 1}
    sample_index = 0
    split_counts = {"train": 0, "val": 0, "test": 0}
    samples = []

    for split, count in split_plan.items():
        split_samples = []
        for _local_index in range(count):
            sample_id = f"sample_{sample_index:04d}_seed_{200 + sample_index:06d}"
            sample_dir = dataset_dir / split / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            input_pixels = []
            label_pixels = []
            for row in range(image_size):
                input_row = []
                label_row = []
                for col in range(image_size):
                    label = 4
                    if row < image_size // 2:
                        label = 0
                    if row == image_size // 2:
                        label = 1
                    if row in (image_size // 2 - 1, image_size // 2 + 1) and col < max(image_size // 6, 1):
                        label = 2
                    if row >= image_size - 2 and col >= image_size - 2:
                        label = 3

                    label_row.append(label)
                    input_row.append((label * 45 + row * 3 + col * 5) % 256)
                input_pixels.append(input_row)
                label_pixels.append(label_row)

            input_path = sample_dir / "input.pgm"
            label_path = sample_dir / "label.ppm"
            metadata_path = sample_dir / "metadata.json"
            write_pgm(input_path, input_pixels)
            write_ppm_labels(label_path, label_pixels)
            metadata_path.write_text(
                json.dumps({"seed": 200 + sample_index, "tick": 9, "actor_count": 8}) + "\n",
                encoding="utf-8",
            )

            record = {
                "seed": 200 + sample_index,
                "sample_id": sample_id,
                "split": split,
                "directory": f"{split}/{sample_id}",
                "input_image": f"{split}/{sample_id}/input.pgm",
                "label_image": f"{split}/{sample_id}/label.ppm",
                "metadata": f"{split}/{sample_id}/metadata.json",
            }
            split_samples.append(record)
            samples.append(record)
            split_counts[split] += 1
            sample_index += 1

        index_payload = {"split": split, "sample_count": len(split_samples), "samples": split_samples}
        (dataset_dir / "splits" / f"{split}.json").write_text(
            json.dumps(index_payload, indent=2) + "\n",
            encoding="utf-8",
        )

    manifest = {
        "config": {
            "seed_start": 200,
            "num_scenes": sample_index,
            "dt": 0.05,
            "ticks": 10,
            "noise": "low",
            "metadata_format": "json",
            "split_seed": 321,
            "split_ratios": {"train": 4 / 7, "val": 2 / 7, "test": 1 / 7},
        },
        "splits": {
            split: {"count": split_counts[split], "index_path": f"splits/{split}.json"}
            for split in ("train", "val", "test")
        },
        "samples": samples,
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
