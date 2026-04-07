from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "python" / "run_benchmarks.py"
SIM_RUNNER_PATH = REPO_ROOT / "build" / "bin" / "sim_runner"
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))
TESTS_ROOT = PYTHON_ROOT / "tests"
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from support import create_tiny_dataset


@unittest.skipUnless(SIM_RUNNER_PATH.is_file(), "sim_runner binary not built")
class RunBenchmarksScriptTest(unittest.TestCase):
    def test_runs_full_suite_and_writes_combined_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            low_dataset_dir = temp_path / "low_dataset"
            high_dataset_dir = temp_path / "high_dataset"
            output_dir = temp_path / "suite_out"
            create_tiny_dataset(low_dataset_dir, image_size=16)
            create_tiny_dataset(high_dataset_dir, image_size=16)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--sim-runner",
                    str(SIM_RUNNER_PATH),
                    "--output-dir",
                    str(output_dir),
                    "--num-scenes",
                    "2",
                    "--ticks",
                    "8",
                    "--low-dataset-dir",
                    str(low_dataset_dir),
                    "--high-dataset-dir",
                    str(high_dataset_dir),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "2",
                    "--base-channels",
                    "4",
                    "--device",
                    "cpu",
                    "--force",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("benchmark_suite", result.stdout)
            payload = json.loads((output_dir / "benchmark_suite.json").read_text(encoding="utf-8"))
            self.assertEqual(set(payload["selected_benchmarks"]), {"generation", "determinism", "ml"})
            self.assertIn("generation", payload)
            self.assertIn("determinism", payload)
            self.assertIn("ml", payload)
            self.assertTrue(payload["determinism"]["summary"]["all_identical"])
            self.assertEqual(len(payload["ml"]["summary"]["rows"]), 2)


if __name__ == "__main__":
    unittest.main()
