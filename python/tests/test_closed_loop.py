from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from perception import ClosedLoopFrame, PlanPoint, metadata_collision, run_closed_loop_episode, world_to_plan_point


class ClosedLoopTest(unittest.TestCase):
    def test_world_to_plan_point_matches_bev_mapping(self) -> None:
        point = world_to_plan_point(0.0, 0.0)

        self.assertIsNotNone(point)
        assert point is not None
        self.assertEqual((point.row, point.col), (63, 64))

    def test_metadata_collision_detects_ground_truth_car_overlap(self) -> None:
        ego = world_to_plan_point(0.0, 0.0)
        assert ego is not None
        metadata = {
            "actors": [
                {
                    "type": "car",
                    "x": 0.0,
                    "y": 0.0,
                    "heading": 0.0,
                }
            ]
        }

        self.assertTrue(metadata_collision(metadata, ego))

    def test_closed_loop_episode_advances_ego_along_lane(self) -> None:
        frame_map = torch.full((8, 8), 4, dtype=torch.long)
        frame_map[:, 2] = 0
        frame_map[:, 3] = 1
        frame_map[:, 4] = 0
        frames = [
            ClosedLoopFrame(
                tick=index,
                time_seconds=float(index),
                semantic_prediction=frame_map.clone(),
                metadata={"seed": 7, "tick": index, "time_seconds": float(index), "actors": []},
            )
            for index in range(4)
        ]

        episode = run_closed_loop_episode(frames, dt=1.0, control_horizon=2)

        self.assertFalse(episode.collision)
        self.assertFalse(episode.success)
        self.assertEqual((episode.start.row, episode.start.col), (0, 3))
        self.assertEqual((episode.goal.row, episode.goal.col), (7, 3))
        self.assertEqual((episode.steps[0].ego_position.row, episode.steps[0].ego_position.col), (0, 3))
        self.assertEqual((episode.steps[1].ego_position.row, episode.steps[1].ego_position.col), (1, 3))
        self.assertEqual(episode.steps[0].command["action"], "track_path")

    def test_closed_loop_episode_stops_on_collision(self) -> None:
        frame_map = torch.full((8, 8), 4, dtype=torch.long)
        frame_map[:, 2] = 0
        frame_map[:, 3] = 1
        frame_map[:, 4] = 0
        blocking_point = PlanPoint(row=1, col=3)
        blocking_world_x = float(blocking_point.col) - 64.0 + 0.5
        blocking_world_y = 64.0 - float(blocking_point.row) - 0.5
        frames = [
            ClosedLoopFrame(
                tick=0,
                time_seconds=0.0,
                semantic_prediction=frame_map.clone(),
                metadata={"seed": 9, "tick": 0, "time_seconds": 0.0, "actors": []},
            ),
            ClosedLoopFrame(
                tick=1,
                time_seconds=1.0,
                semantic_prediction=frame_map.clone(),
                metadata={
                    "seed": 9,
                    "tick": 1,
                    "time_seconds": 1.0,
                    "actors": [
                        {
                            "type": "car",
                            "x": blocking_world_x,
                            "y": blocking_world_y,
                            "heading": 0.0,
                        }
                    ],
                },
            ),
        ]

        episode = run_closed_loop_episode(frames, dt=1.0, control_horizon=2)

        self.assertTrue(episode.collision)
        self.assertEqual(episode.collision_tick, 1)
        self.assertEqual((episode.steps[-1].ego_position.row, episode.steps[-1].ego_position.col), (1, 3))


if __name__ == "__main__":
    unittest.main()
