"""Closed-loop ego driving loop on top of predicted semantic BEV frames."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

import torch
from torch import Tensor

from models import SEMANTIC_CLASS_NAMES

from .control import EgoControlCommand, EgoControlRollout, rollout_ego_control
from .planning import EgoPlan, PlanPoint, plan_ego_route
from .prediction import ForecastBundle, forecast_from_semantic_maps
from .state_extraction import ExtractedSemanticState, extract_semantic_state


GRID_SIZE = 128
WORLD_HALF_EXTENT = 64.0
CAR_LENGTH_M = 4.5
CAR_WIDTH_M = 2.5
PEDESTRIAN_RADIUS_M = 0.75


@dataclass(frozen=True)
class ClosedLoopFrame:
    tick: int
    time_seconds: float
    semantic_prediction: Tensor
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ClosedLoopStep:
    tick: int
    time_seconds: float
    ego_position: PlanPoint
    goal: PlanPoint
    collision: bool
    goal_reached: bool
    predicted_classes: tuple[str, ...]
    forecast: dict[str, object] | None
    extracted_state: dict[str, object]
    ego_plan: dict[str, object]
    ego_control: dict[str, object] | None
    command: dict[str, object] | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ClosedLoopEpisode:
    seed: int | None
    dt: float
    ticks_requested: int
    ticks_completed: int
    success: bool
    collision: bool
    collision_tick: int | None
    goal: PlanPoint
    start: PlanPoint
    steps: tuple[ClosedLoopStep, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _compact_blob(blob: dict[str, object]) -> dict[str, object]:
    return {
        "class_name": blob["class_name"],
        "area": blob["area"],
        "centroid_row": blob["centroid_row"],
        "centroid_col": blob["centroid_col"],
        "bbox": blob["bbox"],
    }


def compact_state(state: ExtractedSemanticState) -> dict[str, object]:
    payload = state.to_dict()
    return {
        "height": payload["height"],
        "width": payload["width"],
        "free_space": {
            "area": payload["free_space"]["area"],
            "bbox": payload["free_space"]["bbox"],
        },
        "lane_regions": [_compact_blob(blob) for blob in payload["lane_regions"]],
        "lane_tracks": [
            {
                "orientation": track["orientation"],
                "region": _compact_blob(track["region"]),
                "centerline": track["centerline"],
            }
            for track in payload["lane_tracks"]
        ],
        "vehicle_blobs": [_compact_blob(blob) for blob in payload["vehicle_blobs"]],
        "pedestrian_blobs": [_compact_blob(blob) for blob in payload["pedestrian_blobs"]],
        "obstacle_regions": [_compact_blob(blob) for blob in payload["obstacle_regions"]],
    }


def world_to_plan_point(x: float, y: float) -> PlanPoint | None:
    col = int(math.floor(x + WORLD_HALF_EXTENT))
    world_row = int(math.floor(y + WORLD_HALF_EXTENT))
    row = GRID_SIZE - 1 - world_row
    if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
        return PlanPoint(row=row, col=col)
    return None


def _image_cell_center(point: PlanPoint) -> tuple[float, float]:
    wx = float(point.col) - WORLD_HALF_EXTENT + 0.5
    wy = WORLD_HALF_EXTENT - float(point.row) - 0.5
    return wx, wy


def _car_occupies_point(actor: dict[str, Any], point: PlanPoint) -> bool:
    wx, wy = _image_cell_center(point)
    dx = wx - float(actor["x"])
    dy = wy - float(actor["y"])
    heading = float(actor["heading"])
    c = math.cos(heading)
    s = math.sin(heading)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    return abs(local_x) <= (CAR_LENGTH_M / 2.0) and abs(local_y) <= (CAR_WIDTH_M / 2.0)


def _pedestrian_occupies_point(actor: dict[str, Any], point: PlanPoint) -> bool:
    wx, wy = _image_cell_center(point)
    dx = wx - float(actor["x"])
    dy = wy - float(actor["y"])
    return (dx * dx + dy * dy) <= (PEDESTRIAN_RADIUS_M * PEDESTRIAN_RADIUS_M)


def actor_occupies_point(actor: dict[str, Any], point: PlanPoint) -> bool:
    actor_type = str(actor.get("type", ""))
    if actor_type == "car":
        return _car_occupies_point(actor, point)
    if actor_type == "pedestrian":
        return _pedestrian_occupies_point(actor, point)
    return False


def metadata_collision(metadata: dict[str, Any], point: PlanPoint) -> bool:
    actors = metadata.get("actors", [])
    if not isinstance(actors, list):
        return False
    return any(actor_occupies_point(actor, point) for actor in actors if isinstance(actor, dict))


def _first_command(control: EgoControlRollout | None) -> EgoControlCommand | None:
    if control is None or not control.commands:
        return None
    return control.commands[0]


def _predicted_class_names(semantic_prediction: Tensor) -> tuple[str, ...]:
    return tuple(sorted({SEMANTIC_CLASS_NAMES[class_id] for class_id in semantic_prediction.unique().tolist()}))


def run_closed_loop_episode(
    frames: Sequence[ClosedLoopFrame],
    *,
    dt: float,
    control_horizon: int,
) -> ClosedLoopEpisode:
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if control_horizon <= 0:
        raise ValueError("control_horizon must be positive")
    if not frames:
        raise ValueError("expected at least one frame")

    previous_prediction: Tensor | None = None
    current_ego: PlanPoint | None = None
    goal: PlanPoint | None = None
    start: PlanPoint | None = None
    steps: list[ClosedLoopStep] = []
    collision_tick: int | None = None
    collision = False
    success = False

    for frame_index, frame in enumerate(frames):
        extracted_state = extract_semantic_state(frame.semantic_prediction)
        forecast: ForecastBundle | None = None
        if previous_prediction is not None:
            forecast = forecast_from_semantic_maps(
                previous_prediction,
                frame.semantic_prediction,
                dt=dt,
                horizon_steps=control_horizon,
            )

        if current_ego is None or goal is None:
            bootstrap_plan = plan_ego_route(
                frame.semantic_prediction,
                extracted_state=extracted_state,
                forecast=forecast,
            )
            current_ego = bootstrap_plan.start
            goal = bootstrap_plan.goal
            start = bootstrap_plan.start

        collision = metadata_collision(frame.metadata, current_ego)
        goal_reached = current_ego == goal
        control: EgoControlRollout | None = None
        command: EgoControlCommand | None = None

        ego_plan = plan_ego_route(
            frame.semantic_prediction,
            extracted_state=extracted_state,
            forecast=forecast,
            start_override=current_ego,
            goal_override=goal,
        )

        if not collision and not goal_reached and frame_index + 1 < len(frames):
            control = rollout_ego_control(
                ego_plan,
                forecast=forecast,
                dt=dt,
                horizon_steps=control_horizon,
            )
            command = _first_command(control)

        steps.append(
            ClosedLoopStep(
                tick=frame.tick,
                time_seconds=frame.time_seconds,
                ego_position=current_ego,
                goal=goal,
                collision=collision,
                goal_reached=goal_reached,
                predicted_classes=_predicted_class_names(frame.semantic_prediction),
                forecast=None if forecast is None else forecast.to_dict(),
                extracted_state=compact_state(extracted_state),
                ego_plan=ego_plan.to_dict(),
                ego_control=None if control is None else control.to_dict(),
                command=None if command is None else asdict(command),
            )
        )

        previous_prediction = frame.semantic_prediction

        if collision:
            collision_tick = frame.tick
            break
        if goal_reached:
            success = True
            break
        if control is not None and len(control.states) > 1:
            next_state = control.states[1]
            current_ego = PlanPoint(row=next_state.row, col=next_state.col)

    assert goal is not None
    assert start is not None
    if not success and steps and steps[-1].goal_reached:
        success = True

    seed_value = frames[0].metadata.get("seed") if isinstance(frames[0].metadata, dict) else None
    seed = int(seed_value) if isinstance(seed_value, int) else None
    return ClosedLoopEpisode(
        seed=seed,
        dt=dt,
        ticks_requested=len(frames),
        ticks_completed=len(steps),
        success=success,
        collision=collision,
        collision_tick=collision_tick,
        goal=goal,
        start=start,
        steps=tuple(steps),
    )
