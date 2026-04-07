#include <gtest/gtest.h>

#include "sim/BevRasterizer.h"
#include "sim/NoiseInjector.h"

using namespace sim;

TEST(NoiseInjectorTest, NoisyRasterIs128x128) {
    Scene scene(42);
    BevRasterizer rasterizer;
    NoiseInjector injector;

    const BevImage clean = rasterizer.rasterize(scene);
    const NoisyBevImage noisy = injector.render(clean, 42, NoiseInjector::presetConfig(NoisePreset::LOW));

    EXPECT_EQ(static_cast<int>(noisy.pixels.size()), GRID_SIZE);
    EXPECT_EQ(static_cast<int>(noisy.pixels[0].size()), GRID_SIZE);
}

TEST(NoiseInjectorTest, ZeroNoiseConfigPreservesBaseIntensities) {
    Scene scene(42);
    BevRasterizer rasterizer;
    NoiseInjector injector;

    NoiseConfig config;
    config.obstacle_intensity = 20;
    config.drivable_intensity = 110;
    config.lane_intensity = 190;
    config.vehicle_intensity = 245;
    config.pedestrian_intensity = 225;
    config.speckle_intensity_min = 180;
    config.speckle_intensity_max = 255;
    config.jitter_amplitude = 0;
    config.dropout_probability = 0.0f;
    config.speckle_probability = 0.0f;

    const BevImage clean = rasterizer.rasterize(scene);
    const NoisyBevImage noisy = injector.render(clean, 42, config);

    EXPECT_EQ(noisy.pixelAtWorld(0.0f, 0.0f), 190);
}

TEST(NoiseInjectorTest, SameSeedAndConfigProduceIdenticalNoisyImage) {
    Scene scene(42);
    BevRasterizer rasterizer;
    NoiseInjector injector;

    const BevImage clean = rasterizer.rasterize(scene);
    const NoiseConfig config = NoiseInjector::presetConfig(NoisePreset::LOW);
    const NoisyBevImage first = injector.render(clean, 1234, config);
    const NoisyBevImage second = injector.render(clean, 1234, config);

    EXPECT_EQ(first.pixels, second.pixels);
}

TEST(NoiseInjectorTest, DifferentPresetsProduceDifferentImagesForSameSeed) {
    Scene scene(42);
    BevRasterizer rasterizer;
    NoiseInjector injector;

    const BevImage clean = rasterizer.rasterize(scene);
    const NoisyBevImage low = injector.render(clean, 1234, NoiseInjector::presetConfig(NoisePreset::LOW));
    const NoisyBevImage high = injector.render(clean, 1234, NoiseInjector::presetConfig(NoisePreset::HIGH));

    EXPECT_NE(low.pixels, high.pixels);
}

TEST(NoiseInjectorTest, DifferentSeedsProduceDifferentImages) {
    Scene scene(42);
    BevRasterizer rasterizer;
    NoiseInjector injector;

    const BevImage clean = rasterizer.rasterize(scene);
    const NoiseConfig config = NoiseInjector::presetConfig(NoisePreset::LOW);
    const NoisyBevImage first = injector.render(clean, 1, config);
    const NoisyBevImage second = injector.render(clean, 2, config);

    EXPECT_NE(first.pixels, second.pixels);
}
