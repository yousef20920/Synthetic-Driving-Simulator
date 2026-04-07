#include <gtest/gtest.h>

#include "sim/Actor.h"
#include "sim/Scene.h"

using namespace sim;

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kHeadingNorth = kPi / 2.0f;
constexpr float kHeadingSouth = -kPi / 2.0f;
constexpr float kHeadingEast = 0.0f;
constexpr float kHeadingWest = kPi;

void freezeAllActors(Scene& scene) {
    for (auto& actor : scene.actors()) {
        actor.preferred_velocity = 0.0f;
        actor.velocity = 0.0f;
        actor.motion_state = actor.type == ActorType::CAR ? MotionState::CAR_DONE : MotionState::PED_DONE;
    }
}

void configureCar(Actor& actor,
                  int route_id,
                  float longitudinal,
                  float lateral,
                  float preferred_velocity,
                  MotionState state,
                  int state_tick) {
    actor.type = ActorType::CAR;
    actor.route_id = route_id;
    actor.preferred_velocity = preferred_velocity;
    actor.velocity = state == MotionState::CAR_WAITING_AT_STOP ? 0.0f : preferred_velocity;
    actor.motion_state = state;
    actor.state_tick = state_tick;

    switch (route_id) {
        case 0:
            actor.x = lateral;
            actor.y = longitudinal;
            actor.heading = kHeadingSouth;
            break;
        case 1:
            actor.x = lateral;
            actor.y = longitudinal;
            actor.heading = kHeadingNorth;
            break;
        case 2:
            actor.x = longitudinal;
            actor.y = lateral;
            actor.heading = kHeadingWest;
            break;
        case 3:
            actor.x = longitudinal;
            actor.y = lateral;
            actor.heading = kHeadingEast;
            break;
        default:
            break;
    }
}

}  // namespace

TEST(YieldingTest, EarlierArrivalGetsPriorityAtIntersection) {
    Scene scene(42);
    freezeAllActors(scene);

    auto& first = scene.actors()[0];
    auto& second = scene.actors()[1];
    configureCar(first, 0, 8.0f, 1.75f, 4.0f, MotionState::CAR_WAITING_AT_STOP, 5);
    configureCar(second, 3, -8.0f, -1.75f, 4.0f, MotionState::CAR_WAITING_AT_STOP, 7);

    scene.advance(0.5f, 10);

    EXPECT_EQ(first.motion_state, MotionState::CAR_CROSSING);
    EXPECT_EQ(second.motion_state, MotionState::CAR_WAITING_AT_STOP);
}

TEST(YieldingTest, WaitingCarHoldsWhileAnotherCarIsCrossing) {
    Scene scene(42);
    freezeAllActors(scene);

    auto& crossing = scene.actors()[0];
    auto& waiting = scene.actors()[1];
    configureCar(crossing, 0, 4.0f, 1.75f, 4.0f, MotionState::CAR_CROSSING, 2);
    configureCar(waiting, 2, 8.0f, 1.75f, 4.0f, MotionState::CAR_WAITING_AT_STOP, 1);

    scene.advance(0.5f, 10);

    EXPECT_EQ(waiting.motion_state, MotionState::CAR_WAITING_AT_STOP);
    EXPECT_FLOAT_EQ(waiting.x, 8.0f);
}
