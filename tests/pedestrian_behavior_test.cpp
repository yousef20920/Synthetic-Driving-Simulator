#include <gtest/gtest.h>

#include "sim/Actor.h"
#include "sim/Scene.h"

using namespace sim;

namespace {

void freezeAllActors(Scene& scene) {
    for (auto& actor : scene.actors()) {
        actor.preferred_velocity = 0.0f;
        actor.velocity = 0.0f;
        actor.motion_state = actor.type == ActorType::CAR ? MotionState::CAR_DONE : MotionState::PED_DONE;
    }
}

void configurePedestrian(Actor& actor,
                         int route_id,
                         float x,
                         float y,
                         float preferred_velocity,
                         MotionState state) {
    actor.type = ActorType::PEDESTRIAN;
    actor.route_id = route_id;
    actor.x = x;
    actor.y = y;
    actor.preferred_velocity = preferred_velocity;
    actor.velocity = preferred_velocity;
    actor.motion_state = state;
}

}  // namespace

TEST(PedestrianBehaviorTest, PedestrianApproachesCrosswalkEntry) {
    Scene scene(42);
    freezeAllActors(scene);

    auto& ped = scene.actors()[4];
    configurePedestrian(ped, 0, -8.0f, 2.0f, 1.0f, MotionState::PED_APPROACH);

    scene.advance(1.0f, 0);

    EXPECT_FLOAT_EQ(ped.x, -8.0f);
    EXPECT_LT(ped.y, 2.0f);
}

TEST(PedestrianBehaviorTest, PedestrianWaitsWhenSignalIsClosed) {
    Scene scene(42);
    freezeAllActors(scene);

    auto& ped = scene.actors()[4];
    configurePedestrian(ped, 0, -8.0f, 0.0f, 1.0f, MotionState::PED_WAITING_SIGNAL);

    scene.advance(1.0f, 25);

    EXPECT_EQ(ped.motion_state, MotionState::PED_WAITING_SIGNAL);
    EXPECT_FLOAT_EQ(ped.x, -8.0f);
}

TEST(PedestrianBehaviorTest, PedestrianCrossesWhenSignalIsOpen) {
    Scene scene(42);
    freezeAllActors(scene);

    auto& ped = scene.actors()[4];
    configurePedestrian(ped, 0, -8.0f, 0.0f, 1.0f, MotionState::PED_WAITING_SIGNAL);

    scene.advance(1.0f, 5);

    EXPECT_EQ(ped.motion_state, MotionState::PED_CROSSING);
    EXPECT_GT(ped.x, -8.0f);
}
