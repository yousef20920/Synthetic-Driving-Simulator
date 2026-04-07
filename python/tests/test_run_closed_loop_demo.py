from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import run_closed_loop_demo
from perception import ClosedLoopFrame


class _FakeModel:
    def to(self, _device):
        return self

    def eval(self):
        return self


class RunClosedLoopDemoTest(unittest.TestCase):
    def test_main_writes_summary_and_viewer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint_path = temp_path / "model.pt"
            sim_runner_path = temp_path / "sim_runner"
            output_dir = temp_path / "outputs"
            checkpoint_path.write_text("checkpoint\n", encoding="utf-8")
            sim_runner_path.write_text("runner\n", encoding="utf-8")

            semantic_map = torch.full((8, 8), 4, dtype=torch.long)
            semantic_map[:, 2] = 0
            semantic_map[:, 3] = 1
            semantic_map[:, 4] = 0
            frames = [
                ClosedLoopFrame(
                    tick=index,
                    time_seconds=index * 0.1,
                    semantic_prediction=semantic_map.clone(),
                    metadata={"seed": 42, "tick": index, "time_seconds": index * 0.1, "actors": []},
                )
                for index in range(3)
            ]
            visuals = [
                {
                    "width": 8,
                    "height": 8,
                    "input": [0] * 64,
                    "prediction": semantic_map.reshape(-1).tolist(),
                    "metadata": {"seed": 42, "tick": index, "time_seconds": index * 0.1, "actors": []},
                }
                for index in range(3)
            ]

            with mock.patch.object(
                run_closed_loop_demo,
                "load_checkpoint",
                return_value=(_FakeModel(), {"epoch": 5}),
            ), mock.patch.object(
                run_closed_loop_demo,
                "collect_frames",
                return_value=(frames, visuals),
            ):
                exit_code = run_closed_loop_demo.main(
                    [
                        "--sim-runner",
                        str(sim_runner_path),
                        "--checkpoint",
                        str(checkpoint_path),
                        "--ticks",
                        "3",
                        "--dt",
                        "0.1",
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(exit_code, 0)
            summary_path = output_dir / "closed_loop_run.json"
            viewer_path = output_dir / "closed_loop_viewer.html"
            self.assertTrue(summary_path.is_file())
            self.assertTrue(viewer_path.is_file())
            self.assertIn("Closed-Loop Demo", viewer_path.read_text(encoding="utf-8"))
            self.assertIn("ego_plan", summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
