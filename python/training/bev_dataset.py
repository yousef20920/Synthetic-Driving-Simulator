"""Dataset loader for the generated BEV training samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor
from torch.utils.data import Dataset

from models import NUM_SEMANTIC_CLASSES, SEMANTIC_CLASS_NAMES


SEMANTIC_CLASS_COLORS = {
    "drivable": (96, 99, 88),
    "lane": (245, 208, 97),
    "vehicle": (190, 74, 47),
    "pedestrian": (15, 124, 115),
    "obstacle": (43, 33, 23),
}
_COLOR_TO_CLASS_INDEX = {
    SEMANTIC_CLASS_COLORS[name]: index for index, name in enumerate(SEMANTIC_CLASS_NAMES)
}


def _netpbm_tokens(path: Path) -> list[str]:
    tokens: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0]
        if line:
            tokens.extend(line.split())
    return tokens


def _load_ascii_pgm(path: Path) -> Tensor:
    tokens = _netpbm_tokens(path)
    if len(tokens) < 4 or tokens[0] != "P2":
        raise ValueError(f"expected ASCII PGM (P2) file: {path}")

    width = int(tokens[1])
    height = int(tokens[2])
    max_value = int(tokens[3])
    pixel_values = [int(token) for token in tokens[4:]]
    if len(pixel_values) != width * height:
        raise ValueError(f"unexpected pixel count in {path}")

    image = torch.tensor(pixel_values, dtype=torch.float32).reshape(height, width)
    return image.unsqueeze(0) / float(max_value)


def _load_ascii_ppm_label(path: Path) -> Tensor:
    tokens = _netpbm_tokens(path)
    if len(tokens) < 4 or tokens[0] != "P3":
        raise ValueError(f"expected ASCII PPM (P3) file: {path}")

    width = int(tokens[1])
    height = int(tokens[2])
    pixel_values = [int(token) for token in tokens[4:]]
    if len(pixel_values) != width * height * 3:
        raise ValueError(f"unexpected RGB pixel count in {path}")

    labels = []
    for index in range(0, len(pixel_values), 3):
        rgb = (pixel_values[index], pixel_values[index + 1], pixel_values[index + 2])
        if rgb not in _COLOR_TO_CLASS_INDEX:
            raise ValueError(f"unknown semantic color {rgb} in {path}")
        labels.append(_COLOR_TO_CLASS_INDEX[rgb])
    return torch.tensor(labels, dtype=torch.long).reshape(height, width)


def load_noisy_input_tensor(path: Path) -> Tensor:
    return _load_ascii_pgm(path)


def load_label_tensor(path: Path) -> Tensor:
    return _load_ascii_ppm_label(path)


def balanced_class_weights_from_histogram(class_histogram: Tensor) -> Tensor:
    histogram = class_histogram.to(dtype=torch.float32)
    if histogram.numel() != NUM_SEMANTIC_CLASSES:
        raise ValueError(
            f"expected class histogram with {NUM_SEMANTIC_CLASSES} entries, got {histogram.numel()}"
        )

    weights = torch.zeros_like(histogram, dtype=torch.float32)
    present_mask = histogram > 0
    if not torch.any(present_mask):
        raise ValueError("cannot compute class weights from an empty histogram")

    inverse_frequency = 1.0 / histogram[present_mask]
    inverse_frequency /= inverse_frequency.mean()
    weights[present_mask] = inverse_frequency
    return weights


class SplitBevDataset(Dataset[dict[str, object]]):
    """Loads one dataset split described by `splits/<split>.json`."""

    def __init__(self, dataset_dir: Path | str, split: str) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        split_path = self.dataset_dir / "splits" / f"{split}.json"
        if not split_path.is_file():
            raise FileNotFoundError(f"split index not found: {split_path}")

        payload = json.loads(split_path.read_text(encoding="utf-8"))
        self.samples = payload.get("samples", [])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        input_path = self.dataset_dir / sample["input_image"]
        label_path = self.dataset_dir / sample["label_image"]
        metadata_path = self.dataset_dir / sample["metadata"]

        return {
            "input": load_noisy_input_tensor(input_path),
            "target": load_label_tensor(label_path),
            "sample_id": sample["sample_id"],
            "split": sample["split"],
            "metadata_path": str(metadata_path),
        }

    def class_histogram(self) -> Tensor:
        histogram = torch.zeros(NUM_SEMANTIC_CLASSES, dtype=torch.long)
        for sample in self.samples:
            label_path = self.dataset_dir / sample["label_image"]
            labels = load_label_tensor(label_path)
            histogram += torch.bincount(labels.reshape(-1), minlength=NUM_SEMANTIC_CLASSES)
        return histogram

    def balanced_class_weights(self) -> Tensor:
        return balanced_class_weights_from_histogram(self.class_histogram())


def list_sample_ids(dataset: Iterable[dict[str, object]]) -> list[str]:
    return [str(sample["sample_id"]) for sample in dataset]
