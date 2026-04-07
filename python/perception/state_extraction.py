"""Extract planner-friendly state from semantic BEV maps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch
from torch import Tensor

from models import NUM_SEMANTIC_CLASSES, SEMANTIC_CLASS_NAMES


DRIVABLE_INDEX = SEMANTIC_CLASS_NAMES.index("drivable")
LANE_INDEX = SEMANTIC_CLASS_NAMES.index("lane")
VEHICLE_INDEX = SEMANTIC_CLASS_NAMES.index("vehicle")
PEDESTRIAN_INDEX = SEMANTIC_CLASS_NAMES.index("pedestrian")
OBSTACLE_INDEX = SEMANTIC_CLASS_NAMES.index("obstacle")


@dataclass(frozen=True)
class GridPoint:
    row: int
    col: int


@dataclass(frozen=True)
class BoundingBox:
    min_row: int
    min_col: int
    max_row: int
    max_col: int


@dataclass(frozen=True)
class GridBlob:
    class_name: str
    area: int
    centroid_row: float
    centroid_col: float
    bbox: BoundingBox
    cells: tuple[GridPoint, ...]


@dataclass(frozen=True)
class LaneCenterPoint:
    row: float
    col: float


@dataclass(frozen=True)
class LaneTrack:
    orientation: str
    region: GridBlob
    centerline: tuple[LaneCenterPoint, ...]


@dataclass(frozen=True)
class ExtractedSemanticState:
    height: int
    width: int
    free_space: GridBlob
    lane_regions: tuple[GridBlob, ...]
    lane_tracks: tuple[LaneTrack, ...]
    vehicle_blobs: tuple[GridBlob, ...]
    pedestrian_blobs: tuple[GridBlob, ...]
    obstacle_regions: tuple[GridBlob, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def tensor_from_logits(logits: Tensor) -> Tensor:
    if logits.ndim != 4:
        raise ValueError("expected logits with shape [batch, classes, height, width]")
    if logits.shape[0] != 1:
        raise ValueError("expected batch size 1 for state extraction")
    if logits.shape[1] != NUM_SEMANTIC_CLASSES:
        raise ValueError(
            f"expected {NUM_SEMANTIC_CLASSES} semantic classes, got {logits.shape[1]}"
        )
    return torch.argmax(logits, dim=1).squeeze(0)


def _neighbors(row: int, col: int, height: int, width: int) -> Iterable[tuple[int, int]]:
    if row > 0:
        yield row - 1, col
    if row + 1 < height:
        yield row + 1, col
    if col > 0:
        yield row, col - 1
    if col + 1 < width:
        yield row, col + 1


def _make_bbox(rows: list[int], cols: list[int]) -> BoundingBox:
    return BoundingBox(
        min_row=min(rows),
        min_col=min(cols),
        max_row=max(rows),
        max_col=max(cols),
    )


def _make_blob(class_name: str, cells: list[tuple[int, int]]) -> GridBlob:
    rows = [row for row, _col in cells]
    cols = [col for _row, col in cells]
    ordered_cells = tuple(GridPoint(row=row, col=col) for row, col in sorted(cells))
    return GridBlob(
        class_name=class_name,
        area=len(cells),
        centroid_row=sum(rows) / len(rows),
        centroid_col=sum(cols) / len(cols),
        bbox=_make_bbox(rows, cols),
        cells=ordered_cells,
    )


def _connected_components(mask: Tensor, class_name: str) -> tuple[GridBlob, ...]:
    if mask.ndim != 2:
        raise ValueError("expected a 2D semantic mask")

    height, width = mask.shape
    visited = torch.zeros((height, width), dtype=torch.bool)
    blobs: list[GridBlob] = []

    for row in range(height):
        for col in range(width):
            if visited[row, col] or not bool(mask[row, col].item()):
                continue

            queue = [(row, col)]
            visited[row, col] = True
            cells: list[tuple[int, int]] = []
            while queue:
                current_row, current_col = queue.pop()
                cells.append((current_row, current_col))
                for next_row, next_col in _neighbors(current_row, current_col, height, width):
                    if visited[next_row, next_col] or not bool(mask[next_row, next_col].item()):
                        continue
                    visited[next_row, next_col] = True
                    queue.append((next_row, next_col))

            blobs.append(_make_blob(class_name, cells))

    return tuple(blobs)


def _free_space_blob(semantic_map: Tensor) -> GridBlob:
    free_space_mask = (semantic_map == DRIVABLE_INDEX) | (semantic_map == LANE_INDEX)
    free_space_cells = [
        (int(row), int(col)) for row, col in torch.nonzero(free_space_mask, as_tuple=False).tolist()
    ]
    if not free_space_cells:
        raise ValueError("semantic map contains no free-space pixels")
    return _make_blob("free_space", free_space_cells)


def _lane_track(blob: GridBlob) -> LaneTrack:
    row_span = blob.bbox.max_row - blob.bbox.min_row
    col_span = blob.bbox.max_col - blob.bbox.min_col
    row_to_cols: dict[int, list[int]] = {}
    col_to_rows: dict[int, list[int]] = {}
    for cell in blob.cells:
        row_to_cols.setdefault(cell.row, []).append(cell.col)
        col_to_rows.setdefault(cell.col, []).append(cell.row)

    if row_span > col_span:
        orientation = "vertical"
        centerline = tuple(
            LaneCenterPoint(row=float(row), col=sum(cols) / len(cols))
            for row, cols in sorted(row_to_cols.items())
        )
    elif col_span > row_span:
        orientation = "horizontal"
        centerline = tuple(
            LaneCenterPoint(row=sum(rows) / len(rows), col=float(col))
            for col, rows in sorted(col_to_rows.items())
        )
    else:
        orientation = "ambiguous"
        centerline = tuple(
            LaneCenterPoint(row=float(row), col=sum(cols) / len(cols))
            for row, cols in sorted(row_to_cols.items())
        )

    return LaneTrack(orientation=orientation, region=blob, centerline=centerline)


def _semantic_mask(semantic_map: Tensor, class_index: int) -> Tensor:
    return semantic_map == class_index


def extract_semantic_state(semantic_map: Tensor) -> ExtractedSemanticState:
    if semantic_map.ndim != 2:
        raise ValueError("expected semantic map with shape [height, width]")
    if semantic_map.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.long):
        raise ValueError("expected integer semantic class indices")

    height, width = semantic_map.shape
    free_space = _free_space_blob(semantic_map)
    lane_regions = _connected_components(_semantic_mask(semantic_map, LANE_INDEX), "lane")
    lane_tracks = tuple(_lane_track(blob) for blob in lane_regions)
    vehicle_blobs = _connected_components(_semantic_mask(semantic_map, VEHICLE_INDEX), "vehicle")
    pedestrian_blobs = _connected_components(_semantic_mask(semantic_map, PEDESTRIAN_INDEX), "pedestrian")
    obstacle_regions = _connected_components(_semantic_mask(semantic_map, OBSTACLE_INDEX), "obstacle")

    return ExtractedSemanticState(
        height=height,
        width=width,
        free_space=free_space,
        lane_regions=lane_regions,
        lane_tracks=lane_tracks,
        vehicle_blobs=vehicle_blobs,
        pedestrian_blobs=pedestrian_blobs,
        obstacle_regions=obstacle_regions,
    )
