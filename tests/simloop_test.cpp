#include <gtest/gtest.h>

#include "sim/CsvLogger.h"
#include "sim/Scene.h"
#include "sim/SimLoop.h"

#include <sstream>
#include <string>
#include <vector>

using namespace sim;

static int countLines(const std::string& text) {
    int count = 0;
    for (char c : text) {
        if (c == '\n') {
            ++count;
        }
    }
    return count;
}

static int countOccurrences(const std::string& haystack, const std::string& needle) {
    int count = 0;
    std::size_t pos = 0;
    while ((pos = haystack.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}

class SimLoopTest : public ::testing::Test {
protected:
    std::ostringstream csv_out;
    Scene scene{42};
    CsvLogger logger{csv_out};
};

TEST_F(SimLoopTest, OutputContainsCsvHeader) {
    SimLoop loop(scene, logger, 0.05f, 1);
    loop.run();

    EXPECT_NE(csv_out.str().find("tick,actor_id,type,x,y,heading,velocity"), std::string::npos);
}

TEST_F(SimLoopTest, TenTicksProducesCorrectRowCount) {
    SimLoop loop(scene, logger, 0.05f, 10);
    loop.run();

    const int expected = 1 + 10 * (DEFAULT_CAR_COUNT + DEFAULT_PED_COUNT);
    EXPECT_EQ(countLines(csv_out.str()), expected);
}

TEST_F(SimLoopTest, SingleTickProducesEightDataRows) {
    SimLoop loop(scene, logger, 0.05f, 1);
    loop.run();

    EXPECT_EQ(countLines(csv_out.str()), 1 + DEFAULT_CAR_COUNT + DEFAULT_PED_COUNT);
}

TEST_F(SimLoopTest, ElapsedTimeCorrectAfterRun) {
    constexpr float dt = 0.05f;
    constexpr int ticks = 20;
    SimLoop loop(scene, logger, dt, ticks);
    loop.run();

    EXPECT_NEAR(loop.elapsedTime(), dt * ticks, 1e-4f);
}

TEST_F(SimLoopTest, CsvContainsCarAndPedestrianStrings) {
    SimLoop loop(scene, logger, 0.05f, 1);
    loop.run();

    const std::string out = csv_out.str();
    EXPECT_GT(countOccurrences(out, ",car,"), 0);
    EXPECT_GT(countOccurrences(out, ",pedestrian,"), 0);
}

TEST_F(SimLoopTest, AtLeastOneActorMovesAfterMultipleTicks) {
    std::vector<float> initial_x;
    std::vector<float> initial_y;
    for (const auto& actor : scene.actors()) {
        initial_x.push_back(actor.x);
        initial_y.push_back(actor.y);
    }

    SimLoop loop(scene, logger, 0.05f, 50);
    loop.run();

    bool any_actor_moved = false;
    const auto& actors = scene.actors();
    for (std::size_t i = 0; i < actors.size(); ++i) {
        if (actors[i].x != initial_x[i] || actors[i].y != initial_y[i]) {
            any_actor_moved = true;
            break;
        }
    }

    EXPECT_TRUE(any_actor_moved);
}
