#include "sim/BevRasterizer.h"

#include <cmath>

namespace sim {
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kCarLengthM = 4.5f;
constexpr float kCarWidthM = 2.5f;
constexpr float kPedestrianRadiusM = 0.75f;

int worldToCol(float wx) {
    return static_cast<int>(std::floor(wx + WORLD_HALF_EXTENT));
}

int worldToImageRow(float wy) {
    const int world_row = static_cast<int>(std::floor(wy + WORLD_HALF_EXTENT));
    return GRID_SIZE - 1 - world_row;
}

bool worldInBounds(float wx, float wy) {
    return wx >= -WORLD_HALF_EXTENT && wx < WORLD_HALF_EXTENT &&
           wy >= -WORLD_HALF_EXTENT && wy < WORLD_HALF_EXTENT;
}

float wrapHeading(float heading) {
    while (heading > kPi) {
        heading -= 2.0f * kPi;
    }
    while (heading < -kPi) {
        heading += 2.0f * kPi;
    }
    return heading;
}

}  // namespace

SemanticClass BevImage::pixelAtIndex(int col, int row) const {
    if (col < 0 || col >= GRID_SIZE || row < 0 || row >= GRID_SIZE) {
        return SemanticClass::OBSTACLE;
    }
    return pixels[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)];
}

SemanticClass BevImage::pixelAtWorld(float wx, float wy) const {
    if (!worldInBounds(wx, wy)) {
        return SemanticClass::OBSTACLE;
    }
    return pixelAtIndex(worldToCol(wx), worldToImageRow(wy));
}

SemanticClass BevRasterizer::terrainClass(TileType tile) {
    switch (tile) {
        case TileType::DRIVABLE:
            return SemanticClass::DRIVABLE;
        case TileType::LANE:
            return SemanticClass::LANE;
        case TileType::SIDEWALK:
        case TileType::OBSTACLE:
        case TileType::OUT_OF_BOUNDS:
            return SemanticClass::OBSTACLE;
    }

    return SemanticClass::OBSTACLE;
}

bool BevRasterizer::carOccupiesPixel(const Actor& actor, float wx, float wy) {
    const float dx = wx - actor.x;
    const float dy = wy - actor.y;
    const float heading = wrapHeading(actor.heading);
    const float c = std::cos(heading);
    const float s = std::sin(heading);
    const float local_x = c * dx + s * dy;
    const float local_y = -s * dx + c * dy;

    return std::fabs(local_x) <= (kCarLengthM / 2.0f) &&
           std::fabs(local_y) <= (kCarWidthM / 2.0f);
}

bool BevRasterizer::pedestrianOccupiesPixel(const Actor& actor, float wx, float wy) {
    const float dx = wx - actor.x;
    const float dy = wy - actor.y;
    return (dx * dx + dy * dy) <= (kPedestrianRadiusM * kPedestrianRadiusM);
}

BevImage BevRasterizer::rasterize(const Scene& scene) const {
    BevImage image{};

    for (int image_row = 0; image_row < GRID_SIZE; ++image_row) {
        const int world_row = GRID_SIZE - 1 - image_row;
        for (int col = 0; col < GRID_SIZE; ++col) {
            image.pixels[static_cast<std::size_t>(image_row)][static_cast<std::size_t>(col)] =
                terrainClass(scene.world().tileAtIndex(col, world_row));
        }
    }

    for (int image_row = 0; image_row < GRID_SIZE; ++image_row) {
        for (int col = 0; col < GRID_SIZE; ++col) {
            const float wx = static_cast<float>(col) - WORLD_HALF_EXTENT + 0.5f;
            const float wy = WORLD_HALF_EXTENT - static_cast<float>(image_row) - 0.5f;

            for (const auto& actor : scene.actors()) {
                if (actor.type == ActorType::CAR && carOccupiesPixel(actor, wx, wy)) {
                    image.pixels[static_cast<std::size_t>(image_row)][static_cast<std::size_t>(col)] =
                        SemanticClass::VEHICLE;
                }
            }

            for (const auto& actor : scene.actors()) {
                if (actor.type == ActorType::PEDESTRIAN && pedestrianOccupiesPixel(actor, wx, wy)) {
                    image.pixels[static_cast<std::size_t>(image_row)][static_cast<std::size_t>(col)] =
                        SemanticClass::PEDESTRIAN;
                }
            }
        }
    }

    return image;
}

}  // namespace sim
