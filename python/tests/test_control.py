from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from perception import (
    ActorForecast,
    EgoPlan,
    ForecastBundle,
    ForecastPoint,
    PlanPoint,
    rollout_ego_control,
)


class ControlTest(unittest.TestCase):
    def test_rollout_tracks_path_to_goal(self) -> None:
        plan = EgoPlan(
            strategy="lane_guided_astar",
            used_fallback=False,
            start=PlanPoint(row=0, col=1),
            goal=PlanPoint(row=3, col=1),
            path=(
                PlanPoint(row=0, col=1),
                PlanPoint(row=1, col=1),
                PlanPoint(row=2, col=1),
                PlanPoint(row=3, col=1),
            ),
            path_cost=3.0,
            blocked_current_cells=0,
            blocked_forecast_cells=0,
        )

        rollout = rollout_ego_control(plan, dt=0.5, horizon_steps=3)

        self.assertTrue(rollout.goal_reached)
        self.assertEqual(len(rollout.states), 4)
        self.assertEqual((rollout.states[-1].row, rollout.states[-1].col), (3, 1))
        self.assertEqual(rollout.commands[0].action, "track_path")
        self.assertEqual(rollout.commands[0].steer, "straight")
        self.assertAlmostEqual(rollout.commands[0].speed_cells_per_second, 2.0)

    def test_rollout_brakes_when_forecast_blocks_next_waypoint(self) -> None:
        plan = EgoPlan(
            strategy="lane_guided_astar",
            used_fallback=False,
            start=PlanPoint(row=0, col=1),
            goal=PlanPoint(row=2, col=1),
            path=(
                PlanPoint(row=0, col=1),
                PlanPoint(row=1, col=1),
                PlanPoint(row=2, col=1),
            ),
            path_cost=2.0,
            blocked_current_cells=0,
            blocked_forecast_cells=1,
        )
        forecast = ForecastBundle(
            dt=1.0,
            horizon_steps=2,
            vehicles=(
                ActorForecast(
                    actor_type="vehicle",
                    strategy="lane_following",
                    matched_previous=True,
                    current_row=1.0,
                    current_col=1.0,
                    velocity_row_per_second=0.0,
                    velocity_col_per_second=0.0,
                    trajectory=(ForecastPoint(step=1, row=1.0, col=1.0),),
                ),
            ),
            pedestrians=(),
        )

        rollout = rollout_ego_control(plan, dt=1.0, forecast=forecast, horizon_steps=2)

        self.assertFalse(rollout.goal_reached)
        self.assertEqual(rollout.commands[0].action, "brake_hold")
        self.assertEqual((rollout.states[1].row, rollout.states[1].col), (0, 1))
        self.assertEqual(rollout.commands[1].action, "track_path")
        self.assertEqual((rollout.states[2].row, rollout.states[2].col), (1, 1))


if __name__ == "__main__":
    unittest.main()
