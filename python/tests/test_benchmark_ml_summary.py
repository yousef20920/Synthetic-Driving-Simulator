from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "python" / "benchmark_ml_summary.py"


def fake_noise_payload(best_val: float, test_mean: float, accuracy: float) -> dict[str, object]:
    return {
        "train": {"best_val_mean_iou": best_val},
        "eval": {
            "test_mean_iou": test_mean,
            "test_pixel_accuracy": accuracy,
            "test_per_class_iou": {
                "drivable": test_mean + 0.1,
                "lane": test_mean + 0.05,
                "vehicle": test_mean - 0.02,
                "pedestrian": test_mean - 0.03,
                "obstacle": test_mean + 0.2,
            },
        },
    }


class BenchmarkMlSummaryScriptTest(unittest.TestCase):
    def test_formats_comparison_into_summary_table(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            comparison_path = temp_path / "comparison.json"
            output_dir = temp_path / "summary_out"
            comparison_path.write_text(
                json.dumps(
                    {
                        "low_noise": fake_noise_payload(0.6, 0.55, 0.94),
                        "high_noise": fake_noise_payload(0.5, 0.48, 0.91),
                        "comparison": {
                            "best_val_mean_iou_delta_low_minus_high": 0.1,
                            "test_mean_iou_delta_low_minus_high": 0.07,
                            "test_pixel_accuracy_delta_low_minus_high": 0.03,
                        },
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--comparison-json",
                    str(comparison_path),
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("| noise |", result.stdout)
            payload = json.loads((output_dir / "benchmark_ml_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(payload["rows"]), 2)
            self.assertEqual(payload["rows"][0]["noise_level"], "low_noise")
            self.assertIn("markdown_table", payload)
            self.assertIn("test_mean_iou_delta_low_minus_high", payload["deltas"])


if __name__ == "__main__":
    unittest.main()
