from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "python" / "train_unet.py"
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))
TESTS_ROOT = PYTHON_ROOT / "tests"
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from support import create_tiny_dataset


class TrainUnetScriptTest(unittest.TestCase):
    def test_trains_for_one_epoch_and_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_dir = temp_path / "dataset"
            metrics_path = temp_path / "metrics.json"
            create_tiny_dataset(dataset_dir, image_size=16)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--dataset-dir",
                    str(dataset_dir),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "2",
                    "--base-channels",
                    "4",
                    "--learning-rate",
                    "0.001",
                    "--device",
                    "cpu",
                    "--metrics-out",
                    str(metrics_path),
                    "--checkpoint-out",
                    str(temp_path / "model.pt"),
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            self.assertIn("epoch 1/1", result.stdout)
            self.assertIn("class_weighting=balanced", result.stdout)
            self.assertTrue(metrics_path.is_file())
            self.assertTrue((temp_path / "model.pt").is_file())

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["epochs"], 1)
            self.assertEqual(metrics["class_weighting"], "balanced")
            self.assertEqual(set(metrics["class_weights"].keys()), {
                "drivable",
                "lane",
                "vehicle",
                "pedestrian",
                "obstacle",
            })
            self.assertEqual(len(metrics["history"]), 1)
            epoch_metrics = metrics["history"][0]
            self.assertIn("train_loss", epoch_metrics)
            self.assertIn("val_loss", epoch_metrics)
            self.assertIn("val_pixel_accuracy", epoch_metrics)
            self.assertIn("val_mean_iou", epoch_metrics)
            self.assertEqual(set(epoch_metrics["val_per_class_iou"].keys()), {
                "drivable",
                "lane",
                "vehicle",
                "pedestrian",
                "obstacle",
            })


if __name__ == "__main__":
    unittest.main()
