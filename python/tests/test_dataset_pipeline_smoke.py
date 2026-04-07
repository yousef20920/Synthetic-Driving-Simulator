from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "python" / "generate_dataset.py"
SIM_RUNNER_PATH = REPO_ROOT / "build" / "bin" / "sim_runner"
SPLIT_NAMES = ("train", "val", "test")


@unittest.skipUnless(SIM_RUNNER_PATH.is_file(), "sim_runner binary not built")
class DatasetPipelineSmokeTest(unittest.TestCase):
    def test_generates_real_split_dataset_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "dataset_out"

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--sim-runner",
                    str(SIM_RUNNER_PATH),
                    "--output-dir",
                    str(output_dir),
                    "--seed-start",
                    "100",
                    "--num-scenes",
                    "4",
                    "--ticks",
                    "12",
                    "--noise",
                    "low",
                    "--train-ratio",
                    "0.5",
                    "--val-ratio",
                    "0.25",
                    "--test-ratio",
                    "0.25",
                    "--split-seed",
                    "7",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["config"]["seed_start"], 100)
            self.assertEqual(manifest["config"]["num_scenes"], 4)
            self.assertEqual(manifest["config"]["ticks"], 12)
            self.assertEqual(manifest["config"]["noise"], "low")
            self.assertEqual(manifest["splits"]["train"]["count"], 2)
            self.assertEqual(manifest["splits"]["val"]["count"], 1)
            self.assertEqual(manifest["splits"]["test"]["count"], 1)

            samples = manifest["samples"]
            self.assertEqual(len(samples), 4)

            samples_by_id = {sample["sample_id"]: sample for sample in samples}
            indexed_sample_ids: set[str] = set()

            for split in SPLIT_NAMES:
                index_payload = json.loads(
                    (output_dir / manifest["splits"][split]["index_path"]).read_text(encoding="utf-8")
                )
                self.assertEqual(index_payload["split"], split)
                self.assertEqual(index_payload["sample_count"], manifest["splits"][split]["count"])

                for sample in index_payload["samples"]:
                    indexed_sample_ids.add(sample["sample_id"])
                    self.assertIn(sample["sample_id"], samples_by_id)
                    self.assertEqual(sample["split"], split)

            self.assertEqual(indexed_sample_ids, set(samples_by_id))

            for sample in samples:
                input_path = output_dir / sample["input_image"]
                label_path = output_dir / sample["label_image"]
                metadata_path = output_dir / sample["metadata"]

                self.assertTrue(input_path.is_file())
                self.assertTrue(label_path.is_file())
                self.assertTrue(metadata_path.is_file())
                self.assertTrue(sample["directory"].startswith(f"{sample['split']}/"))

                self.assertEqual(input_path.read_text(encoding="utf-8").splitlines()[0], "P2")
                self.assertEqual(label_path.read_text(encoding="utf-8").splitlines()[0], "P3")

                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                self.assertEqual(metadata["seed"], sample["seed"])
                self.assertEqual(metadata["tick"], 11)
                self.assertGreater(metadata["actor_count"], 0)


if __name__ == "__main__":
    unittest.main()
