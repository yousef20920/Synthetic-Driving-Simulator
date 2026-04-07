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

void freezeOtherActors(Scene& scene, int keep_id) {
    for (auto& actor : scene.actors()) {
        if (actor.actor_id == keep_id) {
            continue;
        }
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
    actor.velocity = preferred_velocity;
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

TEST(CarBehaviorTest, CarsMoveForwardWhileKeepingLaneOffset) {
    Scene scene(42);
    std::vector<float> initial_x;
    std::vector<float> initial_y;

    for (const auto& actor : scene.actors()) {
        if (actor.type == ActorType::CAR) {
            initial_x.push_back(actor.x);
            initial_y.push_back(actor.y);
        }
    }

    for (int tick = 0; tick < 10; ++tick) {
        scene.advance(0.1f, tick);
    }

    int car_index = 0;
    for (const auto& actor : scene.actors()) {
        if (actor.type != ActorType::CAR) {
            continue;
        }

        if (actor.route_id == 0 || actor.route_id == 1) {
            EXPECT_FLOAT_EQ(actor.x, initial_x[car_index]);
            EXPECT_NE(actor.y, initial_y[car_index]);
        } else {
            EXPECT_FLOAT_EQ(actor.y, initial_y[car_index]);
            EXPECT_NE(actor.x, initial_x[car_index]);
        }
        ++car_index;
    }
}

TEST(CarBehaviorTest, CarStopsAtStopLineBeforeIntersection) {
    Scene scene(42);
    auto& target = scene.actors()[0];
    freezeOtherActors(scene, target.actor_id);
    configureCar(target, 2, 8.2f, 1.75f, 5.0f, MotionState::CAR_APPROACH, -1);

    scene.advance(0.5f, 0);

    EXPECT_FLOAT_EQ(target.x, 8.0f);
    EXPECT_FLOAT_EQ(target.y, 1.75f);
    EXPECT_EQ(target.motion_state, MotionState::CAR_WAITING_AT_STOP);
    EXPECT_FLOAT_EQ(target.velocity, 0.0f);
}

TEST(CarBehaviorTest, TrailingCarQueuesBehindStoppedLeaderInSameLane) {
    Scene scene(42);
    freezeOtherActors(scene, -1);

    auto& leader = scene.actors()[0];
    auto& follower = scene.actors()[1];
    configureCar(leader, 2, 8.0f, 1.75f, 5.0f, MotionState::CAR_WAITING_AT_STOP, 0);
    leader.velocity = 0.0f;

    configureCar(follower, 2, 15.0f, 1.75f, 10.0f, MotionState::CAR_APPROACH, -1);

    scene.advance(1.0f, 1);

    EXPECT_EQ(leader.motion_state, MotionState::CAR_WAITING_AT_STOP);
    EXPECT_EQ(follower.motion_state, MotionState::CAR_APPROACH);
    EXPECT_GT(follower.x, 8.0f);
    EXPECT_GE(follower.x - leader.x, 5.0f);
    EXPECT_FLOAT_EQ(follower.y, 1.75f);
}
