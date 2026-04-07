from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "python" / "generate_dataset.py"


def write_executable(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR)


class GenerateDatasetScriptTest(unittest.TestCase):
    def test_generates_split_dataset_layout_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            simulator_path = temp_path / "fake_sim_runner.py"
            log_path = temp_path / "commands.log"
            output_dir = temp_path / "dataset_out"

            write_executable(
                simulator_path,
                textwrap.dedent(
                    """\
                    #!/usr/bin/env python3
                    import json
                    import os
                    import sys
                    from pathlib import Path

                    args = sys.argv[1:]
                    seed = int(args[args.index("--seed") + 1]) if "--seed" in args else -1
                    if "--dump-bev-ppm" in args:
                        out_path = Path(args[args.index("--dump-bev-ppm") + 1])
                        out_path.write_text("P3\\n1 1\\n255\\n0 0 0\\n", encoding="utf-8")
                    if "--dump-noisy-bev-pgm" in args:
                        out_path = Path(args[args.index("--dump-noisy-bev-pgm") + 1])
                        out_path.write_text("P2\\n1 1\\n255\\n17\\n", encoding="utf-8")
                    if "--dump-metadata-json" in args:
                        out_path = Path(args[args.index("--dump-metadata-json") + 1])
                        out_path.write_text(json.dumps({"seed": seed}) + "\\n", encoding="utf-8")
                    if "--dump-metadata-csv" in args:
                        out_path = Path(args[args.index("--dump-metadata-csv") + 1])
                        out_path.write_text(f"seed\\n{seed}\\n", encoding="utf-8")
                    with open(os.environ["SIM_LOG_PATH"], "a", encoding="utf-8") as handle:
                        handle.write(" ".join(args) + "\\n")
                    """
                ),
            )

            env = os.environ.copy()
            env["SIM_LOG_PATH"] = str(log_path)

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--sim-runner",
                    str(simulator_path),
                    "--output-dir",
                    str(output_dir),
                    "--seed-start",
                    "10",
                    "--num-scenes",
                    "5",
                    "--ticks",
                    "7",
                    "--dt",
                    "0.1",
                    "--noise",
                    "high",
                    "--train-ratio",
                    "0.6",
                    "--val-ratio",
                    "0.2",
                    "--test-ratio",
                    "0.2",
                    "--split-seed",
                    "9",
                ],
                check=True,
                capture_output=True,
                env=env,
                text=True,
            )

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["config"]["seed_start"], 10)
            self.assertEqual(manifest["config"]["num_scenes"], 5)
            self.assertEqual(manifest["config"]["ticks"], 7)
            self.assertEqual(manifest["config"]["dt"], 0.1)
            self.assertEqual(manifest["config"]["noise"], "high")
            self.assertEqual(manifest["config"]["split_seed"], 9)
            self.assertEqual(manifest["config"]["split_ratios"]["train"], 0.6)

            self.assertEqual(manifest["splits"]["train"]["count"], 3)
            self.assertEqual(manifest["splits"]["val"]["count"], 1)
            self.assertEqual(manifest["splits"]["test"]["count"], 1)

            samples = manifest["samples"]
            self.assertEqual(len(samples), 5)
            self.assertEqual([sample["seed"] for sample in samples], [10, 11, 12, 13, 14])

            split_counts = {"train": 0, "val": 0, "test": 0}
            for sample in samples:
                split_counts[sample["split"]] += 1
                self.assertTrue((output_dir / sample["input_image"]).is_file())
                self.assertTrue((output_dir / sample["label_image"]).is_file())
                self.assertTrue((output_dir / sample["metadata"]).is_file())
                self.assertTrue(sample["directory"].startswith(f"{sample['split']}/"))

            self.assertEqual(split_counts, {"train": 3, "val": 1, "test": 1})
            self.assertTrue((output_dir / "splits" / "train.json").is_file())
            self.assertTrue((output_dir / "splits" / "val.json").is_file())
            self.assertTrue((output_dir / "splits" / "test.json").is_file())

            commands = log_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(commands), 15)
            self.assertTrue(all("--ticks 7" in command for command in commands))
            self.assertTrue(all("--dt 0.1" in command for command in commands))
            self.assertEqual(sum("--noise high" in command for command in commands), 5)

    def test_csv_metadata_mode_uses_csv_exports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            simulator_path = temp_path / "fake_sim_runner.py"
            log_path = temp_path / "commands.log"
            output_dir = temp_path / "dataset_out"

            write_executable(
                simulator_path,
                textwrap.dedent(
                    """\
                    #!/usr/bin/env python3
                    import os
                    import sys
                    from pathlib import Path

                    args = sys.argv[1:]
                    if "--dump-bev-ppm" in args:
                        Path(args[args.index("--dump-bev-ppm") + 1]).write_text("ok\\n", encoding="utf-8")
                    if "--dump-noisy-bev-pgm" in args:
                        Path(args[args.index("--dump-noisy-bev-pgm") + 1]).write_text("ok\\n", encoding="utf-8")
                    if "--dump-metadata-csv" in args:
                        Path(args[args.index("--dump-metadata-csv") + 1]).write_text("seed\\n0\\n", encoding="utf-8")
                    with open(os.environ["SIM_LOG_PATH"], "a", encoding="utf-8") as handle:
                        handle.write(" ".join(args) + "\\n")
                    """
                ),
            )

            env = os.environ.copy()
            env["SIM_LOG_PATH"] = str(log_path)

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--sim-runner",
                    str(simulator_path),
                    "--output-dir",
                    str(output_dir),
                    "--seed-start",
                    "3",
                    "--num-scenes",
                    "1",
                    "--metadata-format",
                    "csv",
                ],
                check=True,
                capture_output=True,
                env=env,
                text=True,
            )

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["config"]["metadata_format"], "csv")
            self.assertEqual(manifest["samples"][0]["metadata"], "train/sample_0000_seed_000003/metadata.csv")

            commands = log_path.read_text(encoding="utf-8")
            self.assertIn("--dump-metadata-csv", commands)
            self.assertNotIn("--dump-metadata-json", commands)
            self.assertTrue((output_dir / "splits" / "train.json").is_file())

    def test_rejects_split_ratios_that_do_not_sum_to_one(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            simulator_path = temp_path / "fake_sim_runner.py"
            output_dir = temp_path / "dataset_out"

            write_executable(
                simulator_path,
                "#!/usr/bin/env python3\n",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--sim-runner",
                    str(simulator_path),
                    "--output-dir",
                    str(output_dir),
                    "--train-ratio",
                    "0.5",
                    "--val-ratio",
                    "0.3",
                    "--test-ratio",
                    "0.3",
                ],
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ratios must sum to 1.0", result.stderr)


if __name__ == "__main__":
    unittest.main()
