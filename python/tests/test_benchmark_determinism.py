from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "python" / "benchmark_determinism.py"
SIM_RUNNER_PATH = REPO_ROOT / "build" / "bin" / "sim_runner"


@unittest.skipUnless(SIM_RUNNER_PATH.is_file(), "sim_runner binary not built")
class BenchmarkDeterminismScriptTest(unittest.TestCase):
    def test_writes_byte_identical_artifact_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "benchmark_out"
            metrics_path = output_dir / "benchmark_determinism.json"

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--sim-runner",
                    str(SIM_RUNNER_PATH),
                    "--output-dir",
                    str(output_dir),
                    "--seed",
                    "77",
                    "--ticks",
                    "12",
                    "--noise",
                    "low",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("determinism_benchmark", result.stdout)
            self.assertTrue(metrics_path.is_file())

            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["config"]["seed"], 77)
            self.assertTrue(payload["summary"]["all_identical"])
            self.assertEqual(payload["summary"]["artifact_count"], 4)
            self.assertEqual(set(payload["artifacts"].keys()), {"csv", "clean_bev", "noisy_bev", "metadata"})
            for artifact in payload["artifacts"].values():
                self.assertTrue(artifact["identical"])


if __name__ == "__main__":
    unittest.main()
