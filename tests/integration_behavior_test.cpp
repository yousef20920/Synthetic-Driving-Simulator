#include <gtest/gtest.h>

#include "sim/CsvLogger.h"
#include "sim/Scene.h"
#include "sim/SimLoop.h"

#include <sstream>
#include <string>

using namespace sim;

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kHeadingWest = kPi;

std::string runSimulation(uint32_t seed, float dt, int num_ticks) {
    std::ostringstream oss;
    Scene scene(seed);
    CsvLogger logger(oss);
    SimLoop loop(scene, logger, dt, num_ticks);
    loop.run();
    return oss.str();
}

void freezeAllActors(Scene& scene) {
    for (auto& actor : scene.actors()) {
        actor.preferred_velocity = 0.0f;
        actor.velocity = 0.0f;
        actor.motion_state = actor.type == ActorType::CAR ? MotionState::CAR_DONE : MotionState::PED_DONE;
    }
}

}  // namespace

TEST(IntegrationBehaviorTest, CarYieldsToCrossingPedestrian) {
    Scene scene(42);
    freezeAllActors(scene);

    auto& car = scene.actors()[0];
    car.type = ActorType::CAR;
    car.route_id = 2;
    car.x = 8.0f;
    car.y = 1.75f;
    car.heading = kHeadingWest;
    car.preferred_velocity = 4.0f;
    car.velocity = 0.0f;
    car.motion_state = MotionState::CAR_WAITING_AT_STOP;
    car.state_tick = 0;

    auto& ped = scene.actors()[4];
    ped.type = ActorType::PEDESTRIAN;
    ped.route_id = 0;
    ped.x = -2.0f;
    ped.y = 0.0f;
    ped.preferred_velocity = 1.0f;
    ped.velocity = 1.0f;
    ped.motion_state = MotionState::PED_CROSSING;

    scene.advance(1.0f, 10);

    EXPECT_EQ(car.motion_state, MotionState::CAR_WAITING_AT_STOP);
    EXPECT_GT(ped.x, -2.0f);
}

TEST(IntegrationBehaviorTest, PhaseTwoBehaviorRemainsDeterministic) {
    const std::string csv_a = runSimulation(42, 0.05f, 120);
    const std::string csv_b = runSimulation(42, 0.05f, 120);

    EXPECT_EQ(csv_a, csv_b);
}
