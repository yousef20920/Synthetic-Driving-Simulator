"""Simple ego control rollout on top of a planned grid path."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .planning import EgoPlan, PlanPoint
from .prediction import ForecastBundle


@dataclass(frozen=True)
class EgoControlCommand:
    step: int
    action: str
    steer: str
    throttle: float
    brake: float
    target_row: int
    target_col: int
    speed_cells_per_second: float


@dataclass(frozen=True)
class EgoState:
    step: int
    row: int
    col: int
    heading: str
    path_index: int


@dataclass(frozen=True)
class EgoControlRollout:
    dt: float
    horizon_steps: int
    goal_reached: bool
    final_path_index: int
    states: tuple[EgoState, ...]
    commands: tuple[EgoControlCommand, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _heading(from_point: PlanPoint, to_point: PlanPoint) -> str:
    row_delta = to_point.row - from_point.row
    col_delta = to_point.col - from_point.col
    if row_delta < 0:
        return "north"
    if row_delta > 0:
        return "south"
    if col_delta > 0:
        return "east"
    if col_delta < 0:
        return "west"
    return "stationary"


def _steer(previous_heading: str, next_heading: str) -> str:
    if previous_heading in ("stationary", next_heading):
        return "straight"

    left_turns = {
        ("north", "west"),
        ("west", "south"),
        ("south", "east"),
        ("east", "north"),
    }
    right_turns = {
        ("north", "east"),
        ("east", "south"),
        ("south", "west"),
        ("west", "north"),
    }
    if (previous_heading, next_heading) in left_turns:
        return "left"
    if (previous_heading, next_heading) in right_turns:
        return "right"
    return "u_turn"


def _forecast_occupancy_by_step(forecast: ForecastBundle | None) -> dict[int, set[tuple[int, int]]]:
    occupancy: dict[int, set[tuple[int, int]]] = {}
    if forecast is None:
        return occupancy

    for actor in (*forecast.vehicles, *forecast.pedestrians):
        for point in actor.trajectory:
            occupancy.setdefault(point.step, set()).add(
                (int(round(point.row)), int(round(point.col)))
            )
    return occupancy


def rollout_ego_control(
    plan: EgoPlan,
    *,
    dt: float,
    forecast: ForecastBundle | None = None,
    horizon_steps: int | None = None,
) -> EgoControlRollout:
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if not plan.path:
        raise ValueError("plan path must not be empty")

    path = plan.path
    max_horizon = len(path) - 1
    if horizon_steps is None:
        forecast_horizon = 0 if forecast is None else forecast.horizon_steps
        rollout_horizon = max(max_horizon, forecast_horizon)
    else:
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive")
        rollout_horizon = horizon_steps

    occupancy = _forecast_occupancy_by_step(forecast)
    current_index = 0
    current_point = path[0]
    current_heading = "stationary"
    states = [
        EgoState(
            step=0,
            row=current_point.row,
            col=current_point.col,
            heading=current_heading,
            path_index=current_index,
        )
    ]
    commands: list[EgoControlCommand] = []

    for step in range(1, rollout_horizon + 1):
        next_index = min(current_index + 1, len(path) - 1)
        next_point = path[next_index]
        blocked_cells = occupancy.get(step, set())

        if next_index == current_index:
            action = "hold_goal"
            steer = "straight"
            throttle = 0.0
            brake = 1.0
            speed = 0.0
        elif (next_point.row, next_point.col) in blocked_cells:
            action = "brake_hold"
            steer = "straight"
            throttle = 0.0
            brake = 1.0
            speed = 0.0
            next_point = current_point
            next_index = current_index
        else:
            next_heading = _heading(current_point, next_point)
            action = "track_path"
            steer = _steer(current_heading, next_heading)
            throttle = 0.65
            brake = 0.0
            speed = 1.0 / dt
            current_heading = next_heading

        commands.append(
            EgoControlCommand(
                step=step,
                action=action,
                steer=steer,
                throttle=throttle,
                brake=brake,
                target_row=next_point.row,
                target_col=next_point.col,
                speed_cells_per_second=speed,
            )
        )
        current_point = next_point
        current_index = next_index
        states.append(
            EgoState(
                step=step,
                row=current_point.row,
                col=current_point.col,
                heading=current_heading,
                path_index=current_index,
            )
        )

    return EgoControlRollout(
        dt=dt,
        horizon_steps=rollout_horizon,
        goal_reached=current_index == len(path) - 1,
        final_path_index=current_index,
        states=tuple(states),
        commands=tuple(commands),
    )
