#include "sim/WorldGrid.h"

#include <cmath>

namespace sim {

WorldGrid::WorldGrid() {
    buildIntersection();
}

int WorldGrid::worldToCol(float wx) {
    return static_cast<int>(std::floor(wx + WORLD_HALF_EXTENT));
}

int WorldGrid::worldToRow(float wy) {
    return static_cast<int>(std::floor(wy + WORLD_HALF_EXTENT));
}

TileType WorldGrid::tileAtIndex(int col, int row) const {
    if (col < 0 || col >= GRID_SIZE || row < 0 || row >= GRID_SIZE) {
        return TileType::OUT_OF_BOUNDS;
    }
    return tiles_[row][col];
}

TileType WorldGrid::tileAt(float wx, float wy) const {
    if (wx < -WORLD_HALF_EXTENT || wx >= WORLD_HALF_EXTENT ||
        wy < -WORLD_HALF_EXTENT || wy >= WORLD_HALF_EXTENT) {
        return TileType::OUT_OF_BOUNDS;
    }

    return tileAtIndex(worldToCol(wx), worldToRow(wy));
}

void WorldGrid::buildIntersection() {
    for (auto& row : tiles_) {
        row.fill(TileType::OBSTACLE);
    }

    const float sw_outer = ROAD_HALF_WIDTH + SIDEWALK_WIDTH_M;
    const float lane_mark_half_width = 0.25f;

    const auto tileOverlapsStripe = [lane_mark_half_width](float coord, float stripe_center) {
        const float tile_min = coord - 0.5f;
        const float tile_max = coord + 0.5f;
        const float stripe_min = stripe_center - lane_mark_half_width;
        const float stripe_max = stripe_center + lane_mark_half_width;
        return tile_max > stripe_min && tile_min < stripe_max;
    };

    for (int row = 0; row < GRID_SIZE; ++row) {
        for (int col = 0; col < GRID_SIZE; ++col) {
            const float wx = static_cast<float>(col - GRID_SIZE / 2) + 0.5f;
            const float wy = static_cast<float>(row - GRID_SIZE / 2) + 0.5f;
            const float ax = std::fabs(wx);
            const float ay = std::fabs(wy);

            const bool on_ns_arm = ax < ROAD_HALF_WIDTH;
            const bool on_ew_arm = ay < ROAD_HALF_WIDTH;

            if (on_ns_arm || on_ew_arm) {
                tiles_[row][col] = TileType::DRIVABLE;

                if (on_ns_arm) {
                    const bool center_line = tileOverlapsStripe(ax, 0.0f);
                    const bool inner_boundary = tileOverlapsStripe(ax, LANE_WIDTH_M);
                    if (center_line || inner_boundary) {
                        tiles_[row][col] = TileType::LANE;
                    }
                }

                if (on_ew_arm && !on_ns_arm) {
                    const bool center_line = tileOverlapsStripe(ay, 0.0f);
                    const bool inner_boundary = tileOverlapsStripe(ay, LANE_WIDTH_M);
                    if (center_line || inner_boundary) {
                        tiles_[row][col] = TileType::LANE;
                    }
                }

                continue;
            }

            const bool ns_sidewalk = ax >= ROAD_HALF_WIDTH && ax < sw_outer;
            const bool ew_sidewalk = ay >= ROAD_HALF_WIDTH && ay < sw_outer;

            if (ns_sidewalk || ew_sidewalk) {
                tiles_[row][col] = TileType::SIDEWALK;
            }
        }
    }
}

}  // namespace sim
