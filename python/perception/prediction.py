"""Simple multi-tick motion forecasting from extracted semantic state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .state_extraction import ExtractedSemanticState, GridBlob, LaneTrack, extract_semantic_state


@dataclass(frozen=True)
class ForecastPoint:
    step: int
    row: float
    col: float


@dataclass(frozen=True)
class ActorForecast:
    actor_type: str
    strategy: str
    matched_previous: bool
    current_row: float
    current_col: float
    velocity_row_per_second: float
    velocity_col_per_second: float
    trajectory: tuple[ForecastPoint, ...]


@dataclass(frozen=True)
class ForecastBundle:
    dt: float
    horizon_steps: int
    vehicles: tuple[ActorForecast, ...]
    pedestrians: tuple[ActorForecast, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _distance(row_a: float, col_a: float, row_b: float, col_b: float) -> float:
    return math.hypot(row_a - row_b, col_a - col_b)


def _match_blobs(
    previous: tuple[GridBlob, ...],
    current: tuple[GridBlob, ...],
    *,
    max_distance: float,
) -> dict[int, GridBlob]:
    matches: dict[int, GridBlob] = {}
    remaining_previous = set(range(len(previous)))

    for current_index, current_blob in enumerate(current):
        best_index: int | None = None
        best_distance = max_distance
        for previous_index in remaining_previous:
            previous_blob = previous[previous_index]
            distance = _distance(
                current_blob.centroid_row,
                current_blob.centroid_col,
                previous_blob.centroid_row,
                previous_blob.centroid_col,
            )
            if distance <= best_distance:
                best_index = previous_index
                best_distance = distance
        if best_index is not None:
            matches[current_index] = previous[best_index]
            remaining_previous.remove(best_index)

    return matches


def _velocity(previous_blob: GridBlob | None, current_blob: GridBlob, dt: float) -> tuple[float, float]:
    if previous_blob is None:
        return 0.0, 0.0
    return (
        (current_blob.centroid_row - previous_blob.centroid_row) / dt,
        (current_blob.centroid_col - previous_blob.centroid_col) / dt,
    )


def _polyline_progress(points: list[tuple[float, float]]) -> list[float]:
    progress = [0.0]
    for index in range(1, len(points)):
        segment = _distance(*points[index - 1], *points[index])
        progress.append(progress[-1] + segment)
    return progress


def _interpolate_polyline(points: list[tuple[float, float]], progress: list[float], target: float) -> tuple[float, float]:
    if len(points) == 1:
        return points[0]
    if target <= progress[0]:
        return points[0]
    if target >= progress[-1]:
        return points[-1]

    for index in range(1, len(progress)):
        if target > progress[index]:
            continue
        start = progress[index - 1]
        end = progress[index]
        ratio = 0.0 if end == start else (target - start) / (end - start)
        row = points[index - 1][0] + (points[index][0] - points[index - 1][0]) * ratio
        col = points[index - 1][1] + (points[index][1] - points[index - 1][1]) * ratio
        return row, col

    return points[-1]


def _nearest_lane_track(blob: GridBlob, lane_tracks: tuple[LaneTrack, ...]) -> LaneTrack | None:
    best_track: LaneTrack | None = None
    best_distance = float("inf")
    for track in lane_tracks:
        for point in track.centerline:
            distance = _distance(blob.centroid_row, blob.centroid_col, point.row, point.col)
            if distance < best_distance:
                best_distance = distance
                best_track = track
    return best_track


def _project_to_lane(track: LaneTrack, row: float, col: float) -> tuple[float, int]:
    points = [(point.row, point.col) for point in track.centerline]
    progress = _polyline_progress(points)
    best_index = 0
    best_distance = float("inf")
    for index, point in enumerate(points):
        distance = _distance(row, col, point[0], point[1])
        if distance < best_distance:
            best_distance = distance
            best_index = index
    return progress[best_index], best_index


def _forecast_vehicle(
    current_blob: GridBlob,
    previous_blob: GridBlob | None,
    lane_tracks: tuple[LaneTrack, ...],
    dt: float,
    horizon_steps: int,
) -> ActorForecast:
    velocity_row, velocity_col = _velocity(previous_blob, current_blob, dt)
    lane_track = _nearest_lane_track(current_blob, lane_tracks)
    if lane_track is None or not lane_track.centerline:
        trajectory = tuple(
            ForecastPoint(
                step=step,
                row=current_blob.centroid_row + velocity_row * dt * step,
                col=current_blob.centroid_col + velocity_col * dt * step,
            )
            for step in range(1, horizon_steps + 1)
        )
        return ActorForecast(
            actor_type="vehicle",
            strategy="constant_velocity",
            matched_previous=previous_blob is not None,
            current_row=current_blob.centroid_row,
            current_col=current_blob.centroid_col,
            velocity_row_per_second=velocity_row,
            velocity_col_per_second=velocity_col,
            trajectory=trajectory,
        )

    if lane_track.orientation == "vertical":
        signed_speed = velocity_row if abs(velocity_row) > 1e-6 else 1.0 / dt
    elif lane_track.orientation == "horizontal":
        signed_speed = velocity_col if abs(velocity_col) > 1e-6 else 1.0 / dt
    else:
        dominant = velocity_row if abs(velocity_row) >= abs(velocity_col) else velocity_col
        signed_speed = dominant if abs(dominant) > 1e-6 else 1.0 / dt

    points = [(point.row, point.col) for point in lane_track.centerline]
    progress = _polyline_progress(points)
    current_progress, _nearest_index = _project_to_lane(lane_track, current_blob.centroid_row, current_blob.centroid_col)
    trajectory = []
    for step in range(1, horizon_steps + 1):
        target_progress = current_progress + signed_speed * dt * step
        row, col = _interpolate_polyline(points, progress, target_progress)
        trajectory.append(ForecastPoint(step=step, row=row, col=col))

    return ActorForecast(
        actor_type="vehicle",
        strategy="lane_following",
        matched_previous=previous_blob is not None,
        current_row=current_blob.centroid_row,
        current_col=current_blob.centroid_col,
        velocity_row_per_second=velocity_row,
        velocity_col_per_second=velocity_col,
        trajectory=tuple(trajectory),
    )


def _forecast_pedestrian(current_blob: GridBlob, previous_blob: GridBlob | None, dt: float, horizon_steps: int) -> ActorForecast:
    velocity_row, velocity_col = _velocity(previous_blob, current_blob, dt)
    trajectory = tuple(
        ForecastPoint(
            step=step,
            row=current_blob.centroid_row + velocity_row * dt * step,
            col=current_blob.centroid_col + velocity_col * dt * step,
        )
        for step in range(1, horizon_steps + 1)
    )
    return ActorForecast(
        actor_type="pedestrian",
        strategy="pedestrian_crossing_forecast",
        matched_previous=previous_blob is not None,
        current_row=current_blob.centroid_row,
        current_col=current_blob.centroid_col,
        velocity_row_per_second=velocity_row,
        velocity_col_per_second=velocity_col,
        trajectory=trajectory,
    )


def forecast_from_states(
    previous_state: ExtractedSemanticState,
    current_state: ExtractedSemanticState,
    *,
    dt: float,
    horizon_steps: int,
    max_match_distance: float = 6.0,
) -> ForecastBundle:
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")

    vehicle_matches = _match_blobs(
        previous_state.vehicle_blobs,
        current_state.vehicle_blobs,
        max_distance=max_match_distance,
    )
    pedestrian_matches = _match_blobs(
        previous_state.pedestrian_blobs,
        current_state.pedestrian_blobs,
        max_distance=max_match_distance,
    )

    vehicles = tuple(
        _forecast_vehicle(
            current_blob=current_blob,
            previous_blob=vehicle_matches.get(index),
            lane_tracks=current_state.lane_tracks,
            dt=dt,
            horizon_steps=horizon_steps,
        )
        for index, current_blob in enumerate(current_state.vehicle_blobs)
    )
    pedestrians = tuple(
        _forecast_pedestrian(
            current_blob=current_blob,
            previous_blob=pedestrian_matches.get(index),
            dt=dt,
            horizon_steps=horizon_steps,
        )
        for index, current_blob in enumerate(current_state.pedestrian_blobs)
    )

    return ForecastBundle(
        dt=dt,
        horizon_steps=horizon_steps,
        vehicles=vehicles,
        pedestrians=pedestrians,
    )


def forecast_from_semantic_maps(previous_map, current_map, *, dt: float, horizon_steps: int) -> ForecastBundle:
    previous_state = extract_semantic_state(previous_map)
    current_state = extract_semantic_state(current_map)
    return forecast_from_states(previous_state, current_state, dt=dt, horizon_steps=horizon_steps)
