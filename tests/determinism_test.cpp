#include <gtest/gtest.h>

#include "sim/CsvLogger.h"
#include "sim/Scene.h"
#include "sim/SimLoop.h"

#include <algorithm>
#include <sstream>
#include <string>

using namespace sim;

static std::string runSimulation(uint32_t seed, float dt, int num_ticks) {
    std::ostringstream oss;
    Scene scene(seed);
    CsvLogger logger(oss);
    SimLoop loop(scene, logger, dt, num_ticks);
    loop.run();
    return oss.str();
}

class DeterminismTest : public ::testing::Test {};

TEST_F(DeterminismTest, SameSeedProducesByteIdenticalOutput_10Ticks) {
    const std::string csv_a = runSimulation(42, 0.05f, 10);
    const std::string csv_b = runSimulation(42, 0.05f, 10);

    ASSERT_FALSE(csv_a.empty());
    ASSERT_FALSE(csv_b.empty());
    EXPECT_EQ(csv_a, csv_b) << "Byte-identical check failed for seed=42, 10 ticks";
}

TEST_F(DeterminismTest, SameSeedProducesByteIdenticalOutput_100Ticks) {
    const std::string csv_a = runSimulation(42, 0.05f, 100);
    const std::string csv_b = runSimulation(42, 0.05f, 100);

    ASSERT_FALSE(csv_a.empty());
    EXPECT_EQ(csv_a, csv_b) << "Byte-identical check failed for seed=42, 100 ticks";
}

TEST_F(DeterminismTest, SameSeedProducesByteIdenticalOutput_DifferentSeed) {
    const std::string csv_a = runSimulation(12345, 0.1f, 50);
    const std::string csv_b = runSimulation(12345, 0.1f, 50);

    ASSERT_FALSE(csv_a.empty());
    EXPECT_EQ(csv_a, csv_b) << "Byte-identical check failed for seed=12345, 50 ticks";
}

TEST_F(DeterminismTest, SameSeedProducesByteIdenticalOutput_SeedZero) {
    const std::string csv_a = runSimulation(0, 0.05f, 10);
    const std::string csv_b = runSimulation(0, 0.05f, 10);

    ASSERT_FALSE(csv_a.empty());
    EXPECT_EQ(csv_a, csv_b) << "Byte-identical check failed for seed=0";
}

TEST_F(DeterminismTest, SameSeedProducesByteIdenticalOutput_MaxSeed) {
    const uint32_t max_seed = 0xFFFFFFFFu;
    const std::string csv_a = runSimulation(max_seed, 0.05f, 10);
    const std::string csv_b = runSimulation(max_seed, 0.05f, 10);

    ASSERT_FALSE(csv_a.empty());
    EXPECT_EQ(csv_a, csv_b) << "Byte-identical check failed for seed=UINT32_MAX";
}

TEST_F(DeterminismTest, DifferentSeedsProduceDifferentOutput) {
    const std::string csv_42 = runSimulation(42, 0.05f, 10);
    const std::string csv_43 = runSimulation(43, 0.05f, 10);
    const std::string csv_9999 = runSimulation(9999, 0.05f, 10);

    EXPECT_NE(csv_42, csv_43);
    EXPECT_NE(csv_42, csv_9999);
    EXPECT_NE(csv_43, csv_9999);
}

TEST_F(DeterminismTest, ActorSpawnPositionsDeterministicAcrossInstances) {
    Scene scene_a(42);
    Scene scene_b(42);

    const auto& actors_a = scene_a.actors();
    const auto& actors_b = scene_b.actors();

    ASSERT_EQ(actors_a.size(), actors_b.size());

    for (std::size_t i = 0; i < actors_a.size(); ++i) {
        EXPECT_EQ(actors_a[i].actor_id, actors_b[i].actor_id);
        EXPECT_EQ(actors_a[i].type, actors_b[i].type);
        EXPECT_FLOAT_EQ(actors_a[i].x, actors_b[i].x);
        EXPECT_FLOAT_EQ(actors_a[i].y, actors_b[i].y);
        EXPECT_FLOAT_EQ(actors_a[i].heading, actors_b[i].heading);
        EXPECT_FLOAT_EQ(actors_a[i].velocity, actors_b[i].velocity);
    }
}

TEST_F(DeterminismTest, ActorSpawnPositionsDifferAcrossSeeds) {
    Scene scene_42(42);
    Scene scene_43(43);

    const auto& a42 = scene_42.actors();
    const auto& a43 = scene_43.actors();

    bool any_differs = false;
    for (std::size_t i = 0; i < std::min(a42.size(), a43.size()); ++i) {
        if (a42[i].x != a43[i].x || a42[i].y != a43[i].y) {
            any_differs = true;
            break;
        }
    }

    EXPECT_TRUE(any_differs);
}

TEST_F(DeterminismTest, CsvHasHeaderAsFirstLine) {
    const std::string csv = runSimulation(42, 0.05f, 1);
    const std::string first_line = csv.substr(0, csv.find('\n'));
    EXPECT_EQ(first_line, "tick,actor_id,type,x,y,heading,velocity");
}

TEST_F(DeterminismTest, CsvRowCountCorrectFor10Ticks) {
    const std::string csv = runSimulation(42, 0.05f, 10);

    int line_count = 0;
    for (char c : csv) {
        if (c == '\n') {
            ++line_count;
        }
    }

    const int expected = 1 + 10 * (DEFAULT_CAR_COUNT + DEFAULT_PED_COUNT);
    EXPECT_EQ(line_count, expected);
}

TEST_F(DeterminismTest, CsvFirstDataRowStartsWithTickZero) {
    const std::string csv = runSimulation(42, 0.05f, 1);
    const std::size_t first_newline = csv.find('\n');
    ASSERT_NE(first_newline, std::string::npos);

    const std::size_t second_newline = csv.find('\n', first_newline + 1);
    const std::string second_line = csv.substr(first_newline + 1, second_newline - first_newline - 1);

    ASSERT_FALSE(second_line.empty());
    EXPECT_EQ(second_line[0], '0');
}
