#include "sim/NoiseInjector.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace sim {
namespace {

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

std::uint8_t clampToByte(int value) {
    return static_cast<std::uint8_t>(std::clamp(value, 0, 255));
}

}  // namespace

std::uint8_t NoisyBevImage::pixelAtIndex(int col, int row) const {
    if (col < 0 || col >= GRID_SIZE || row < 0 || row >= GRID_SIZE) {
        return 0;
    }
    return pixels[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)];
}

std::uint8_t NoisyBevImage::pixelAtWorld(float wx, float wy) const {
    if (!worldInBounds(wx, wy)) {
        return 0;
    }
    return pixelAtIndex(worldToCol(wx), worldToImageRow(wy));
}

NoiseConfig NoiseInjector::presetConfig(NoisePreset preset) {
    switch (preset) {
        case NoisePreset::LOW:
            return NoiseConfig{
                .obstacle_intensity = 20,
                .drivable_intensity = 110,
                .lane_intensity = 190,
                .vehicle_intensity = 245,
                .pedestrian_intensity = 225,
                .speckle_intensity_min = 180,
                .speckle_intensity_max = 255,
                .jitter_amplitude = 8,
                .dropout_probability = 0.01f,
                .speckle_probability = 0.005f,
            };
        case NoisePreset::HIGH:
            return NoiseConfig{
                .obstacle_intensity = 20,
                .drivable_intensity = 110,
                .lane_intensity = 190,
                .vehicle_intensity = 245,
                .pedestrian_intensity = 225,
                .speckle_intensity_min = 160,
                .speckle_intensity_max = 255,
                .jitter_amplitude = 28,
                .dropout_probability = 0.06f,
                .speckle_probability = 0.03f,
            };
    }

    return presetConfig(NoisePreset::LOW);
}

std::uint8_t NoiseInjector::baseIntensity(SemanticClass semantic_class, const NoiseConfig& config) {
    switch (semantic_class) {
        case SemanticClass::DRIVABLE:
            return config.drivable_intensity;
        case SemanticClass::LANE:
            return config.lane_intensity;
        case SemanticClass::VEHICLE:
            return config.vehicle_intensity;
        case SemanticClass::PEDESTRIAN:
            return config.pedestrian_intensity;
        case SemanticClass::OBSTACLE:
            return config.obstacle_intensity;
    }

    return config.obstacle_intensity;
}

NoisyBevImage NoiseInjector::render(const BevImage& clean_image, uint32_t seed, const NoiseConfig& config) const {
    NoisyBevImage image{};
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> probability_dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> jitter_dist(
        -static_cast<int>(config.jitter_amplitude),
        static_cast<int>(config.jitter_amplitude));
    std::uniform_int_distribution<int> speckle_dist(
        static_cast<int>(config.speckle_intensity_min),
        static_cast<int>(config.speckle_intensity_max));

    for (int row = 0; row < GRID_SIZE; ++row) {
        for (int col = 0; col < GRID_SIZE; ++col) {
            int intensity = static_cast<int>(
                baseIntensity(clean_image.pixelAtIndex(col, row), config));

            if (probability_dist(rng) < config.dropout_probability) {
                intensity = static_cast<int>(config.obstacle_intensity);
            } else {
                intensity += jitter_dist(rng);
                if (probability_dist(rng) < config.speckle_probability) {
                    intensity = speckle_dist(rng);
                }
            }

            image.pixels[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] =
                clampToByte(intensity);
        }
    }

    return image;
}

}  // namespace sim
