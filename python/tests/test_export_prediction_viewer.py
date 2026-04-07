from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT_PATH = REPO_ROOT / "python" / "train_unet.py"
VIEWER_SCRIPT_PATH = REPO_ROOT / "python" / "export_prediction_viewer.py"
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))
TESTS_ROOT = PYTHON_ROOT / "tests"
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from support import create_tiny_dataset


class ExportPredictionViewerScriptTest(unittest.TestCase):
    def test_writes_self_contained_prediction_html(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_dir = temp_path / "dataset"
            checkpoint_path = temp_path / "model.pt"
            output_path = temp_path / "prediction_viewer.html"
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
                    str(VIEWER_SCRIPT_PATH),
                    "--dataset-dir",
                    str(dataset_dir),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--split",
                    "test",
                    "--num-samples",
                    "1",
                    "--device",
                    "cpu",
                    "--output-path",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            self.assertIn("wrote viewer to", result.stdout)
            self.assertTrue(output_path.is_file())
            html = output_path.read_text(encoding="utf-8")
            self.assertIn("Model Viewer", html)
            self.assertIn("Noisy Input", html)
            self.assertIn("Ground Truth", html)
            self.assertIn("Prediction", html)
            self.assertIn("Extracted State", html)
            self.assertIn("Extracted State Summary", html)
            self.assertIn("Ego Plan Summary", html)
            self.assertIn("lane_tracks", html)
            self.assertIn("Forecast Horizon", html)
            self.assertIn("Control Horizon", html)
            self.assertIn("Play Rollout", html)
            self.assertIn("Scenario Step", html)
            self.assertIn("forecast", html)
            self.assertIn("ego_plan", html)
            self.assertIn("ego_control", html)
            self.assertIn("Plan Strategy", html)
            self.assertIn("Ego Control Summary", html)
            self.assertIn("sample_0006_seed_000106", html)


if __name__ == "__main__":
    unittest.main()
