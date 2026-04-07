from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from perception import extract_semantic_state, tensor_from_logits


class StateExtractionTest(unittest.TestCase):
    def test_extracts_free_space_blobs_and_lane_centerlines(self) -> None:
        semantic_map = torch.tensor(
            [
                [4, 4, 0, 1, 0, 4],
                [4, 4, 0, 1, 0, 4],
                [4, 4, 0, 1, 0, 4],
                [4, 4, 0, 1, 0, 4],
                [4, 4, 0, 1, 0, 4],
                [4, 4, 0, 1, 0, 4],
            ],
            dtype=torch.long,
        )

        state = extract_semantic_state(semantic_map)

        self.assertEqual(state.free_space.area, 18)
        self.assertEqual(len(state.lane_regions), 1)
        self.assertEqual(state.lane_regions[0].area, 6)
        self.assertEqual(len(state.lane_tracks), 1)
        self.assertEqual(state.lane_tracks[0].orientation, "vertical")
        self.assertEqual(len(state.lane_tracks[0].centerline), 6)
        self.assertAlmostEqual(state.lane_tracks[0].centerline[0].col, 3.0)

    def test_extracts_vehicle_pedestrian_and_obstacle_components(self) -> None:
        semantic_map = torch.tensor(
            [
                [4, 4, 4, 4, 4, 4, 4],
                [4, 2, 2, 4, 3, 4, 4],
                [4, 2, 2, 1, 3, 4, 4],
                [4, 0, 0, 1, 0, 0, 4],
                [4, 2, 4, 1, 0, 4, 4],
                [4, 2, 4, 4, 4, 4, 4],
                [4, 4, 4, 4, 4, 4, 4],
            ],
            dtype=torch.long,
        )

        state = extract_semantic_state(semantic_map)

        self.assertGreater(state.free_space.area, 0)
        self.assertEqual(len(state.vehicle_blobs), 2)
        self.assertEqual(sorted(blob.area for blob in state.vehicle_blobs), [2, 4])
        self.assertEqual(len(state.pedestrian_blobs), 1)
        self.assertEqual(state.pedestrian_blobs[0].area, 2)
        self.assertEqual(len(state.obstacle_regions), 1)
        self.assertEqual(state.obstacle_regions[0].area, 33)

    def test_converts_logits_to_semantic_tensor(self) -> None:
        logits = torch.zeros((1, 5, 3, 3), dtype=torch.float32)
        logits[:, 4, :, :] = 1.0
        logits[:, 1, 1, :] = 3.0
        logits[:, 2, 0, 0] = 5.0

        semantic_map = tensor_from_logits(logits)

        self.assertEqual(tuple(semantic_map.shape), (3, 3))
        self.assertEqual(int(semantic_map[0, 0].item()), 2)
        self.assertEqual(int(semantic_map[1, 1].item()), 1)
        self.assertEqual(int(semantic_map[2, 2].item()), 4)


if __name__ == "__main__":
    unittest.main()
