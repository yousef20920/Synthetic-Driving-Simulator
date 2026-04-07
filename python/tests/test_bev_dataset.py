from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))
TESTS_ROOT = PYTHON_ROOT / "tests"
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from training import SplitBevDataset
from support import create_imbalanced_dataset, create_tiny_dataset


class SplitBevDatasetTest(unittest.TestCase):
    def test_loads_input_and_label_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            create_tiny_dataset(dataset_dir, image_size=10)

            dataset = SplitBevDataset(dataset_dir, "train")
            sample = dataset[0]

            self.assertEqual(sample["input"].shape, torch.Size([1, 10, 10]))
            self.assertEqual(sample["target"].shape, torch.Size([10, 10]))
            self.assertEqual(sample["target"].dtype, torch.long)
            self.assertGreaterEqual(float(sample["input"].min().item()), 0.0)
            self.assertLessEqual(float(sample["input"].max().item()), 1.0)

    def test_split_lengths_match_index_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            create_tiny_dataset(dataset_dir, image_size=8)

            self.assertEqual(len(SplitBevDataset(dataset_dir, "train")), 4)
            self.assertEqual(len(SplitBevDataset(dataset_dir, "val")), 2)
            self.assertEqual(len(SplitBevDataset(dataset_dir, "test")), 1)

    def test_balanced_class_weights_upweight_rare_classes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            create_imbalanced_dataset(dataset_dir, image_size=12)

            dataset = SplitBevDataset(dataset_dir, "train")
            histogram = dataset.class_histogram()
            weights = dataset.balanced_class_weights()

            self.assertEqual(histogram.shape, torch.Size([5]))
            self.assertEqual(weights.shape, torch.Size([5]))
            self.assertGreater(float(weights[3].item()), float(weights[4].item()))
            self.assertGreater(float(weights[1].item()), float(weights[0].item()))
            self.assertGreater(float(weights[2].item()), float(weights[4].item()))


if __name__ == "__main__":
    unittest.main()
