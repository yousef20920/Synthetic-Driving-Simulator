from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from perception import forecast_from_semantic_maps


class PredictionTest(unittest.TestCase):
    def test_vehicle_forecast_follows_lane_centerline(self) -> None:
        previous_map = torch.tensor(
            [
                [4, 4, 0, 1, 0, 4],
                [4, 4, 0, 1, 0, 4],
                [4, 4, 0, 1, 0, 4],
                [4, 4, 0, 1, 0, 4],
                [4, 4, 2, 1, 0, 4],
                [4, 4, 2, 1, 0, 4],
                [4, 4, 0, 1, 0, 4],
                [4, 4, 0, 1, 0, 4],
            ],
            dtype=torch.long,
        )
        current_map = torch.tensor(
            [
                [4, 4, 0, 1, 0, 4],
                [4, 4, 0, 1, 0, 4],
                [4, 4, 0, 1, 0, 4],
                [4, 4, 0, 1, 0, 4],
                [4, 4, 0, 1, 0, 4],
                [4, 4, 2, 1, 0, 4],
                [4, 4, 2, 1, 0, 4],
                [4, 4, 0, 1, 0, 4],
            ],
            dtype=torch.long,
        )

        forecast = forecast_from_semantic_maps(previous_map, current_map, dt=1.0, horizon_steps=2)

        self.assertEqual(len(forecast.vehicles), 1)
        vehicle = forecast.vehicles[0]
        self.assertEqual(vehicle.strategy, "lane_following")
        self.assertTrue(vehicle.matched_previous)
        self.assertAlmostEqual(vehicle.velocity_row_per_second, 1.0)
        self.assertAlmostEqual(vehicle.trajectory[0].col, 3.0)
        self.assertGreater(vehicle.trajectory[0].row, vehicle.current_row)
        self.assertGreater(vehicle.trajectory[1].row, vehicle.trajectory[0].row)

    def test_pedestrian_forecast_uses_constant_velocity(self) -> None:
        previous_map = torch.tensor(
            [
                [4, 4, 4, 4, 4],
                [4, 0, 0, 0, 4],
                [4, 4, 3, 4, 4],
                [4, 4, 4, 4, 4],
                [4, 4, 4, 4, 4],
            ],
            dtype=torch.long,
        )
        current_map = torch.tensor(
            [
                [4, 4, 4, 4, 4],
                [4, 0, 0, 0, 4],
                [4, 4, 4, 4, 4],
                [4, 4, 3, 4, 4],
                [4, 4, 4, 4, 4],
            ],
            dtype=torch.long,
        )

        forecast = forecast_from_semantic_maps(previous_map, current_map, dt=1.0, horizon_steps=3)

        self.assertEqual(len(forecast.pedestrians), 1)
        pedestrian = forecast.pedestrians[0]
        self.assertEqual(pedestrian.strategy, "pedestrian_crossing_forecast")
        self.assertAlmostEqual(pedestrian.velocity_row_per_second, 1.0)
        self.assertAlmostEqual(pedestrian.velocity_col_per_second, 0.0)
        self.assertAlmostEqual(pedestrian.trajectory[0].row, pedestrian.current_row + 1.0)
        self.assertAlmostEqual(pedestrian.trajectory[2].row, pedestrian.current_row + 3.0)


if __name__ == "__main__":
    unittest.main()
