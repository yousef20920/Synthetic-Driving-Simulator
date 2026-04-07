from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT_PATH = REPO_ROOT / "python" / "train_unet.py"
EVAL_SCRIPT_PATH = REPO_ROOT / "python" / "eval_model.py"
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))
TESTS_ROOT = PYTHON_ROOT / "tests"
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from support import create_tiny_dataset


class EvalModelScriptTest(unittest.TestCase):
    def test_evaluates_saved_checkpoint_on_test_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_dir = temp_path / "dataset"
            checkpoint_path = temp_path / "model.pt"
            metrics_path = temp_path / "eval_metrics.json"
            create_tiny_dataset(dataset_dir, image_size=16)

            subprocess.run(
                [
                    sys.executable,
                    str(TRAIN_SCRIPT_PATH),
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
                    "--checkpoint-out",
                    str(checkpoint_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(EVAL_SCRIPT_PATH),
                    "--dataset-dir",
                    str(dataset_dir),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--batch-size",
                    "2",
                    "--device",
                    "cpu",
                    "--metrics-out",
                    str(metrics_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            self.assertIn("test_loss=", result.stdout)
            self.assertTrue(metrics_path.is_file())

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["checkpoint"], str(checkpoint_path))
            self.assertEqual(metrics["checkpoint_epoch"], 1)
            self.assertIn("test_pixel_accuracy", metrics)
            self.assertIn("test_mean_iou", metrics)
            self.assertEqual(set(metrics["test_per_class_iou"].keys()), {
                "drivable",
                "lane",
                "vehicle",
                "pedestrian",
                "obstacle",
            })
