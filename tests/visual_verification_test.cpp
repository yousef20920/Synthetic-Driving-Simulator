#include <gtest/gtest.h>

#include "sim/BevRasterizer.h"
#include "sim/MetadataExporter.h"
#include "sim/NoiseConfigLoader.h"
#include "sim/NoiseInjector.h"

#include <array>
#include <cmath>
#include <sstream>
#include <string>

using namespace sim;

namespace {

Scene captureScene(uint32_t seed, float dt, int num_ticks) {
    Scene scene(seed);
    for (int tick = 0; tick + 1 < num_ticks; ++tick) {
        scene.advance(dt, tick);
    }
    return scene;
}

std::array<int, 5> countSemanticClasses(const BevImage& image) {
    std::array<int, 5> counts{};
    for (const auto& row : image.pixels) {
        for (SemanticClass pixel : row) {
            ++counts[static_cast<std::size_t>(pixel)];
        }
    }
    return counts;
}

std::uint8_t baseIntensity(SemanticClass semantic_class, const NoiseConfig& config) {
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

double meanAbsoluteErrorFromClean(const BevImage& clean, const NoisyBevImage& noisy, const NoiseConfig& config) {
    double total_error = 0.0;
    const double pixel_count = static_cast<double>(GRID_SIZE * GRID_SIZE);

    for (int row = 0; row < GRID_SIZE; ++row) {
        for (int col = 0; col < GRID_SIZE; ++col) {
            const int expected = static_cast<int>(baseIntensity(clean.pixelAtIndex(col, row), config));
            const int observed = static_cast<int>(noisy.pixelAtIndex(col, row));
            total_error += std::fabs(static_cast<double>(observed - expected));
        }
    }

    return total_error / pixel_count;
}

}  // namespace

TEST(VisualVerificationTest, RepresentativeCleanSnapshotContainsAllRequiredSemanticClasses) {
    const Scene scene = captureScene(42, 0.05f, 60);
    BevRasterizer rasterizer;
    const BevImage image = rasterizer.rasterize(scene);
    const auto counts = countSemanticClasses(image);

    EXPECT_GT(counts[static_cast<std::size_t>(SemanticClass::OBSTACLE)], 0);
    EXPECT_GT(counts[static_cast<std::size_t>(SemanticClass::DRIVABLE)], 0);
    EXPECT_GT(counts[static_cast<std::size_t>(SemanticClass::LANE)], 0);
    EXPECT_GT(counts[static_cast<std::size_t>(SemanticClass::VEHICLE)], 0);
    EXPECT_GT(counts[static_cast<std::size_t>(SemanticClass::PEDESTRIAN)], 0);
}

TEST(VisualVerificationTest, HighNoisePresetIsVisiblyMoreDistortedThanLowNoisePreset) {
    const Scene scene = captureScene(42, 0.05f, 60);
    BevRasterizer rasterizer;
    NoiseInjector injector;
    NoiseConfigLoader loader;

    const BevImage clean = rasterizer.rasterize(scene);
    const NoiseConfig low_config =
        loader.loadFromFile(std::string(SIM_PROJECT_SOURCE_DIR) + "/configs/noise_low.yaml");
    const NoiseConfig high_config =
        loader.loadFromFile(std::string(SIM_PROJECT_SOURCE_DIR) + "/configs/noise_high.yaml");

    const NoisyBevImage low = injector.render(clean, 1234, low_config);
    const NoisyBevImage high = injector.render(clean, 1234, high_config);

    const double low_error = meanAbsoluteErrorFromClean(clean, low, low_config);
    const double high_error = meanAbsoluteErrorFromClean(clean, high, high_config);

    EXPECT_GT(high_error, low_error);
    EXPECT_GT(high_error, 10.0);
}

TEST(VisualVerificationTest, MetadataTickMatchesRenderedSnapshotTick) {
    const int num_ticks = 60;
    const float dt = 0.05f;
    const int snapshot_tick = num_ticks - 1;
    const Scene scene = captureScene(42, dt, num_ticks);
    BevRasterizer rasterizer;
    const BevImage image = rasterizer.rasterize(scene);
    MetadataExporter exporter;
    std::ostringstream json;

    exporter.writeJson(json, scene, 42, snapshot_tick, static_cast<float>(snapshot_tick) * dt);
    const std::string metadata = json.str();
    const auto counts = countSemanticClasses(image);

    EXPECT_NE(metadata.find("\"tick\": 59"), std::string::npos);
    EXPECT_NE(metadata.find("\"actor_count\": 8"), std::string::npos);
    EXPECT_GT(counts[static_cast<std::size_t>(SemanticClass::VEHICLE)], 0);
    EXPECT_GT(counts[static_cast<std::size_t>(SemanticClass::PEDESTRIAN)], 0);
}
