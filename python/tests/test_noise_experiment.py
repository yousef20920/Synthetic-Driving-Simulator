from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "python" / "run_noise_experiment.py"
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))
TESTS_ROOT = PYTHON_ROOT / "tests"
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from support import create_tiny_dataset


class NoiseExperimentScriptTest(unittest.TestCase):
    def test_runs_low_vs_high_experiment_and_writes_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            low_dataset_dir = temp_path / "low_dataset"
            high_dataset_dir = temp_path / "high_dataset"
            output_dir = temp_path / "experiment"

            create_tiny_dataset(low_dataset_dir, image_size=16)
            create_tiny_dataset(high_dataset_dir, image_size=16)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--low-dataset-dir",
                    str(low_dataset_dir),
                    "--high-dataset-dir",
                    str(high_dataset_dir),
                    "--output-dir",
                    str(output_dir),
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
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            self.assertIn("low_vs_high", result.stdout)
            comparison_path = output_dir / "comparison.json"
            self.assertTrue(comparison_path.is_file())

            comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
            for label in ("low_noise", "high_noise"):
                self.assertTrue((output_dir / label / "train_metrics.json").is_file())
                self.assertTrue((output_dir / label / "eval_metrics.json").is_file())
                self.assertTrue((output_dir / label / "model.pt").is_file())
                self.assertIn("train", comparison[label])
                self.assertIn("eval", comparison[label])

            self.assertIn("best_val_mean_iou_delta_low_minus_high", comparison["comparison"])
            self.assertIn("test_mean_iou_delta_low_minus_high", comparison["comparison"])
            self.assertIn("test_pixel_accuracy_delta_low_minus_high", comparison["comparison"])
