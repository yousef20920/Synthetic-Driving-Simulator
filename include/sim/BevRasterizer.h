#pragma once

#include "sim/Scene.h"

#include <array>
#include <cstdint>

namespace sim {

enum class SemanticClass : std::uint8_t {
    DRIVABLE = 0,
    LANE = 1,
    VEHICLE = 2,
    PEDESTRIAN = 3,
    OBSTACLE = 4,
};

struct BevImage {
    std::array<std::array<SemanticClass, GRID_SIZE>, GRID_SIZE> pixels{};

    SemanticClass pixelAtIndex(int col, int row) const;
    SemanticClass pixelAtWorld(float wx, float wy) const;
};

class BevRasterizer {
public:
    BevImage rasterize(const Scene& scene) const;

private:
    static SemanticClass terrainClass(TileType tile);
    static bool carOccupiesPixel(const Actor& actor, float wx, float wy);
    static bool pedestrianOccupiesPixel(const Actor& actor, float wx, float wy);
};

}  // namespace sim
