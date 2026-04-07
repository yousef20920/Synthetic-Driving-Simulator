from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from perception import ActorForecast, ForecastBundle, ForecastPoint, PlanPoint, plan_ego_route


class PlanningTest(unittest.TestCase):
    def test_plan_ego_route_avoids_forecasted_vehicle_cells(self) -> None:
        semantic_map = torch.tensor(
            [
                [4, 4, 0, 1, 0, 4, 4],
                [4, 4, 0, 1, 0, 4, 4],
                [4, 4, 0, 1, 0, 4, 4],
                [4, 4, 0, 1, 0, 4, 4],
                [4, 4, 0, 1, 0, 4, 4],
                [4, 4, 0, 1, 0, 4, 4],
                [4, 4, 0, 1, 0, 4, 4],
                [4, 4, 0, 1, 0, 4, 4],
            ],
            dtype=torch.long,
        )
        forecast = ForecastBundle(
            dt=1.0,
            horizon_steps=3,
            vehicles=(
                ActorForecast(
                    actor_type="vehicle",
                    strategy="lane_following",
                    matched_previous=True,
                    current_row=1.0,
                    current_col=3.0,
                    velocity_row_per_second=1.0,
                    velocity_col_per_second=0.0,
                    trajectory=(
                        ForecastPoint(step=1, row=2.0, col=3.0),
                        ForecastPoint(step=2, row=3.0, col=3.0),
                        ForecastPoint(step=3, row=4.0, col=3.0),
                    ),
                ),
            ),
            pedestrians=(),
        )

        plan = plan_ego_route(semantic_map, forecast=forecast)
        path_cells = {(point.row, point.col) for point in plan.path}

        self.assertEqual(plan.strategy, "lane_guided_astar")
        self.assertEqual((plan.start.row, plan.start.col), (0, 3))
        self.assertEqual((plan.goal.row, plan.goal.col), (7, 3))
        self.assertNotIn((2, 3), path_cells)
        self.assertNotIn((3, 3), path_cells)
        self.assertNotIn((4, 3), path_cells)
        self.assertTrue(any(point.col != 3 for point in plan.path))

    def test_plan_ego_route_falls_back_without_lane_pixels(self) -> None:
        semantic_map = torch.tensor(
            [
                [4, 4, 4, 4, 4, 4],
                [4, 0, 0, 0, 0, 4],
                [4, 0, 0, 0, 0, 4],
                [4, 0, 0, 0, 0, 4],
                [4, 4, 4, 4, 4, 4],
            ],
            dtype=torch.long,
        )

        plan = plan_ego_route(semantic_map)

        self.assertEqual(plan.start.row, 2)
        self.assertEqual(plan.goal.row, 2)
        self.assertEqual(plan.start.col, 1)
        self.assertEqual(plan.goal.col, 4)
        self.assertGreaterEqual(len(plan.path), 4)
        self.assertTrue(all(point.row in (1, 2, 3) for point in plan.path))

    def test_plan_ego_route_respects_start_and_goal_overrides(self) -> None:
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

        plan = plan_ego_route(
            semantic_map,
            start_override=PlanPoint(row=1, col=2),
            goal_override=PlanPoint(row=4, col=4),
        )

        self.assertEqual((plan.start.row, plan.start.col), (1, 2))
        self.assertEqual((plan.goal.row, plan.goal.col), (4, 4))
        self.assertEqual((plan.path[0].row, plan.path[0].col), (1, 2))
        self.assertEqual((plan.path[-1].row, plan.path[-1].col), (4, 4))


if __name__ == "__main__":
    unittest.main()
