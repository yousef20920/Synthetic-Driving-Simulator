#pragma once

#include <cstdint>

namespace sim {

// Static terrain classes for the 1m x 1m world grid.
enum class TileType : std::uint8_t {
    OUT_OF_BOUNDS = 0,
    DRIVABLE = 1,
    LANE = 2,
    SIDEWALK = 3,
    OBSTACLE = 4,
};

}  // namespace sim
