from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "python" / "benchmark_generation.py"
SIM_RUNNER_PATH = REPO_ROOT / "build" / "bin" / "sim_runner"


@unittest.skipUnless(SIM_RUNNER_PATH.is_file(), "sim_runner binary not built")
class BenchmarkGenerationScriptTest(unittest.TestCase):
    def test_writes_generation_metrics_and_dataset_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "benchmark_out"
            metrics_path = output_dir / "benchmark_generation.json"

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--sim-runner",
                    str(SIM_RUNNER_PATH),
                    "--output-dir",
                    str(output_dir),
                    "--seed-start",
                    "200",
                    "--num-scenes",
                    "3",
                    "--ticks",
                    "12",
                    "--noise",
                    "low",
                    "--train-ratio",
                    "0.34",
                    "--val-ratio",
                    "0.33",
                    "--test-ratio",
                    "0.33",
                    "--split-seed",
                    "7",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("generation_benchmark", result.stdout)
            self.assertTrue(metrics_path.is_file())
            self.assertTrue((output_dir / "manifest.json").is_file())

            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["config"]["seed_start"], 200)
            self.assertEqual(payload["summary"]["scene_count"], 3)
            self.assertGreater(payload["summary"]["total_duration_seconds"], 0.0)
            self.assertGreater(payload["summary"]["mean_scene_duration_seconds"], 0.0)
            self.assertGreater(payload["summary"]["scenes_per_second"], 0.0)
            self.assertEqual(len(payload["scene_timings"]), 3)

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["samples"]), 3)


if __name__ == "__main__":
    unittest.main()
