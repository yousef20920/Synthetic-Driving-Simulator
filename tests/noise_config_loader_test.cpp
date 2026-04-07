#include <gtest/gtest.h>

#include "sim/NoiseConfigLoader.h"

#include <filesystem>
#include <fstream>
#include <string>

using namespace sim;

namespace {

std::filesystem::path writeTempConfig(const std::string& contents, const std::string& name) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream out(path);
    out << contents;
    out.close();
    return path;
}

}  // namespace

TEST(NoiseConfigLoaderTest, LoadsNoiseValuesFromYamlFile) {
    const auto path = writeTempConfig(
        "obstacle_intensity: 21\n"
        "drivable_intensity: 111\n"
        "lane_intensity: 191\n"
        "vehicle_intensity: 241\n"
        "pedestrian_intensity: 221\n"
        "speckle_intensity_min: 170\n"
        "speckle_intensity_max: 250\n"
        "jitter_amplitude: 9\n"
        "dropout_probability: 0.02\n"
        "speckle_probability: 0.01\n",
        "noise_loader_test.yaml");

    NoiseConfigLoader loader;
    const NoiseConfig config = loader.loadFromFile(path.string());

    EXPECT_EQ(config.obstacle_intensity, 21);
    EXPECT_EQ(config.drivable_intensity, 111);
    EXPECT_EQ(config.lane_intensity, 191);
    EXPECT_EQ(config.vehicle_intensity, 241);
    EXPECT_EQ(config.pedestrian_intensity, 221);
    EXPECT_EQ(config.speckle_intensity_min, 170);
    EXPECT_EQ(config.speckle_intensity_max, 250);
    EXPECT_EQ(config.jitter_amplitude, 9);
    EXPECT_FLOAT_EQ(config.dropout_probability, 0.02f);
    EXPECT_FLOAT_EQ(config.speckle_probability, 0.01f);
}

TEST(NoiseConfigLoaderTest, RejectsUnknownKeys) {
    const auto path = writeTempConfig(
        "drivable_intensity: 111\n"
        "mystery_value: 7\n",
        "noise_loader_invalid.yaml");

    NoiseConfigLoader loader;
    EXPECT_THROW(loader.loadFromFile(path.string()), std::runtime_error);
}

TEST(NoiseConfigLoaderTest, CheckedInLowAndHighPresetFilesDiffer) {
    NoiseConfigLoader loader;
    const auto low = loader.loadFromFile(std::string(SIM_PROJECT_SOURCE_DIR) + "/configs/noise_low.yaml");
    const auto high = loader.loadFromFile(std::string(SIM_PROJECT_SOURCE_DIR) + "/configs/noise_high.yaml");

    EXPECT_NE(low.jitter_amplitude, high.jitter_amplitude);
    EXPECT_NE(low.dropout_probability, high.dropout_probability);
    EXPECT_NE(low.speckle_probability, high.speckle_probability);
}
