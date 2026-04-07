#pragma once

#include "sim/BevRasterizer.h"

#include <array>
#include <cstdint>

namespace sim {

enum class NoisePreset : std::uint8_t {
    LOW = 0,
    HIGH = 1,
};

struct NoiseConfig {
    std::uint8_t obstacle_intensity = 20;
    std::uint8_t drivable_intensity = 110;
    std::uint8_t lane_intensity = 190;
    std::uint8_t vehicle_intensity = 245;
    std::uint8_t pedestrian_intensity = 225;
    std::uint8_t speckle_intensity_min = 180;
    std::uint8_t speckle_intensity_max = 255;
    std::uint8_t jitter_amplitude = 0;
    float dropout_probability = 0.0f;
    float speckle_probability = 0.0f;
};

struct NoisyBevImage {
    std::array<std::array<std::uint8_t, GRID_SIZE>, GRID_SIZE> pixels{};

    std::uint8_t pixelAtIndex(int col, int row) const;
    std::uint8_t pixelAtWorld(float wx, float wy) const;
};

class NoiseInjector {
public:
    static NoiseConfig presetConfig(NoisePreset preset);

    NoisyBevImage render(const BevImage& clean_image, uint32_t seed, const NoiseConfig& config) const;

private:
    static std::uint8_t baseIntensity(SemanticClass semantic_class, const NoiseConfig& config);
};

}  // namespace sim
