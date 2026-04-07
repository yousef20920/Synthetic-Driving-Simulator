#pragma once

#include "sim/TileType.h"

#include <array>

namespace sim {

static constexpr int GRID_SIZE = 128;
static constexpr float WORLD_HALF_EXTENT = 64.0f;
static constexpr float LANE_WIDTH_M = 3.5f;
static constexpr int LANES_PER_DIRECTION = 2;
static constexpr float ROAD_HALF_WIDTH = LANES_PER_DIRECTION * LANE_WIDTH_M;
static constexpr float SIDEWALK_WIDTH_M = 2.0f;

class WorldGrid {
public:
    WorldGrid();

    TileType tileAt(float wx, float wy) const;
    TileType tileAtIndex(int col, int row) const;

    const std::array<std::array<TileType, GRID_SIZE>, GRID_SIZE>& tiles() const {
        return tiles_;
    }

private:
    std::array<std::array<TileType, GRID_SIZE>, GRID_SIZE> tiles_{};

    void buildIntersection();
    static int worldToCol(float wx);
    static int worldToRow(float wy);
};

}  // namespace sim
