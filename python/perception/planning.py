"""Simple ego-route planning on top of semantic state and short-horizon forecasts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
from itertools import count
from typing import Iterable

import torch
from torch import Tensor

from .prediction import ForecastBundle
from .state_extraction import (
    DRIVABLE_INDEX,
    LANE_INDEX,
    OBSTACLE_INDEX,
    PEDESTRIAN_INDEX,
    VEHICLE_INDEX,
    ExtractedSemanticState,
    LaneTrack,
    extract_semantic_state,
)


@dataclass(frozen=True)
class PlanPoint:
    row: int
    col: int


@dataclass(frozen=True)
class EgoPlan:
    strategy: str
    used_fallback: bool
    start: PlanPoint
    goal: PlanPoint
    path: tuple[PlanPoint, ...]
    path_cost: float
    blocked_current_cells: int
    blocked_forecast_cells: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _traversable_mask(semantic_map: Tensor) -> Tensor:
    return (semantic_map == DRIVABLE_INDEX) | (semantic_map == LANE_INDEX)


def _lane_mask(semantic_map: Tensor) -> Tensor:
    return semantic_map == LANE_INDEX


def _blocked_current_mask(semantic_map: Tensor) -> Tensor:
    return (
        (semantic_map == VEHICLE_INDEX)
        | (semantic_map == PEDESTRIAN_INDEX)
        | (semantic_map == OBSTACLE_INDEX)
    )


def _blocked_forecast_mask(
    forecast: ForecastBundle | None,
    *,
    height: int,
    width: int,
) -> Tensor:
    mask = torch.zeros((height, width), dtype=torch.bool)
    if forecast is None:
        return mask

    for actor in (*forecast.vehicles, *forecast.pedestrians):
        for point in actor.trajectory:
            row = int(round(point.row))
            col = int(round(point.col))
            if 0 <= row < height and 0 <= col < width:
                mask[row, col] = True
    return mask


def _lane_track_score(track: LaneTrack) -> tuple[int, int]:
    row_span = track.region.bbox.max_row - track.region.bbox.min_row
    col_span = track.region.bbox.max_col - track.region.bbox.min_col
    return len(track.centerline), max(row_span, col_span)


def _primary_lane_track(state: ExtractedSemanticState) -> LaneTrack | None:
    if not state.lane_tracks:
        return None
    return max(state.lane_tracks, key=_lane_track_score)


def _fallback_endpoints(state: ExtractedSemanticState) -> tuple[PlanPoint, PlanPoint]:
    bbox = state.free_space.bbox
    row_span = bbox.max_row - bbox.min_row
    col_span = bbox.max_col - bbox.min_col
    center_row = round(state.free_space.centroid_row)
    center_col = round(state.free_space.centroid_col)

    if row_span >= col_span:
        return (
            PlanPoint(row=bbox.min_row, col=center_col),
            PlanPoint(row=bbox.max_row, col=center_col),
        )
    return (
        PlanPoint(row=center_row, col=bbox.min_col),
        PlanPoint(row=center_row, col=bbox.max_col),
    )


def _candidate_endpoints(
    state: ExtractedSemanticState,
    *,
    start_override: PlanPoint | None = None,
    goal_override: PlanPoint | None = None,
) -> tuple[PlanPoint, PlanPoint]:
    if start_override is not None and goal_override is not None:
        return start_override, goal_override

    primary_lane = _primary_lane_track(state)
    if primary_lane is None or not primary_lane.centerline:
        fallback_start, fallback_goal = _fallback_endpoints(state)
        return (
            start_override if start_override is not None else fallback_start,
            goal_override if goal_override is not None else fallback_goal,
        )

    start_point = primary_lane.centerline[0]
    end_point = primary_lane.centerline[-1]
    default_start = PlanPoint(row=int(round(start_point.row)), col=int(round(start_point.col)))
    default_goal = PlanPoint(row=int(round(end_point.row)), col=int(round(end_point.col)))
    return (
        start_override if start_override is not None else default_start,
        goal_override if goal_override is not None else default_goal,
    )


def _iter_neighbors(row: int, col: int, height: int, width: int) -> Iterable[tuple[int, int]]:
    if row > 0:
        yield row - 1, col
    if row + 1 < height:
        yield row + 1, col
    if col > 0:
        yield row, col - 1
    if col + 1 < width:
        yield row, col + 1


def _nearest_allowed(mask: Tensor, point: PlanPoint) -> PlanPoint | None:
    height, width = mask.shape
    if 0 <= point.row < height and 0 <= point.col < width and bool(mask[point.row, point.col].item()):
        return point

    best_point: PlanPoint | None = None
    best_distance: tuple[int, int] | None = None
    for row, col in torch.nonzero(mask, as_tuple=False).tolist():
        distance = abs(row - point.row) + abs(col - point.col)
        tie_break = abs(row - point.row) + abs(col - point.col)
        score = (distance, tie_break)
        if best_distance is None or score < best_distance:
            best_point = PlanPoint(row=int(row), col=int(col))
            best_distance = score
    return best_point


def _heuristic(point: PlanPoint, goal: PlanPoint) -> float:
    return float(abs(point.row - goal.row) + abs(point.col - goal.col))


def _reconstruct_path(
    came_from: dict[tuple[int, int], tuple[int, int] | None],
    goal: PlanPoint,
) -> tuple[PlanPoint, ...]:
    current: tuple[int, int] | None = (goal.row, goal.col)
    path: list[PlanPoint] = []
    while current is not None:
        path.append(PlanPoint(row=current[0], col=current[1]))
        current = came_from[current]
    path.reverse()
    return tuple(path)


def _astar(
    *,
    traversable_mask: Tensor,
    lane_mask: Tensor,
    blocked_mask: Tensor,
    start: PlanPoint,
    goal: PlanPoint,
) -> tuple[tuple[PlanPoint, ...], float, PlanPoint, PlanPoint] | None:
    allowed = traversable_mask & ~blocked_mask
    if not bool(allowed.any().item()):
        return None

    resolved_start = _nearest_allowed(allowed, start)
    resolved_goal = _nearest_allowed(allowed, goal)
    if resolved_start is None or resolved_goal is None:
        return None

    frontier: list[tuple[float, float, int, int, int]] = []
    insertion_order = count()
    heapq.heappush(
        frontier,
        (
            _heuristic(resolved_start, resolved_goal),
            0.0,
            next(insertion_order),
            resolved_start.row,
            resolved_start.col,
        ),
    )
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {
        (resolved_start.row, resolved_start.col): None
    }
    g_score: dict[tuple[int, int], float] = {
        (resolved_start.row, resolved_start.col): 0.0
    }

    height, width = traversable_mask.shape
    while frontier:
        _f_score, current_cost, _order, row, col = heapq.heappop(frontier)
        if (row, col) == (resolved_goal.row, resolved_goal.col):
            path = _reconstruct_path(came_from, resolved_goal)
            return path, current_cost, resolved_start, resolved_goal

        for next_row, next_col in _iter_neighbors(row, col, height, width):
            if not bool(allowed[next_row, next_col].item()):
                continue

            lane_penalty = 0.0 if bool(lane_mask[next_row, next_col].item()) else 0.18
            tentative_cost = current_cost + 1.0 + lane_penalty
            key = (next_row, next_col)
            if tentative_cost >= g_score.get(key, float("inf")):
                continue

            g_score[key] = tentative_cost
            came_from[key] = (row, col)
            next_point = PlanPoint(row=next_row, col=next_col)
            heapq.heappush(
                frontier,
                (
                    tentative_cost + _heuristic(next_point, resolved_goal),
                    tentative_cost,
                    next(insertion_order),
                    next_row,
                    next_col,
                ),
            )

    return None


def plan_ego_route(
    semantic_map: Tensor,
    *,
    extracted_state: ExtractedSemanticState | None = None,
    forecast: ForecastBundle | None = None,
    start_override: PlanPoint | None = None,
    goal_override: PlanPoint | None = None,
) -> EgoPlan:
    if semantic_map.ndim != 2:
        raise ValueError("expected semantic map with shape [height, width]")

    state = extracted_state or extract_semantic_state(semantic_map)
    traversable_mask = _traversable_mask(semantic_map)
    lane_mask = _lane_mask(semantic_map)
    blocked_current = _blocked_current_mask(semantic_map)
    blocked_forecast = _blocked_forecast_mask(
        forecast,
        height=int(semantic_map.shape[0]),
        width=int(semantic_map.shape[1]),
    )
    start, goal = _candidate_endpoints(
        state,
        start_override=start_override,
        goal_override=goal_override,
    )

    attempts = (
        ("lane_guided_astar", blocked_current | blocked_forecast, False),
        ("lane_guided_astar_current_only", blocked_current, True),
        ("lane_guided_astar_free_space_only", torch.zeros_like(blocked_current), True),
    )
    for strategy, blocked_mask, used_fallback in attempts:
        result = _astar(
            traversable_mask=traversable_mask,
            lane_mask=lane_mask,
            blocked_mask=blocked_mask,
            start=start,
            goal=goal,
        )
        if result is None:
            continue
        path, path_cost, resolved_start, resolved_goal = result
        return EgoPlan(
            strategy=strategy,
            used_fallback=used_fallback,
            start=resolved_start,
            goal=resolved_goal,
            path=path,
            path_cost=path_cost,
            blocked_current_cells=int(blocked_current.sum().item()),
            blocked_forecast_cells=int(blocked_forecast.sum().item()),
        )

    raise RuntimeError("unable to plan a route through the current semantic map")
