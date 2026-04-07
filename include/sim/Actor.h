#pragma once

#include <cstdint>

namespace sim {

enum class ActorType : std::uint8_t {
    CAR = 0,
    PEDESTRIAN = 1,
};

enum class MotionState : std::uint8_t {
    STATIC = 0,
    CAR_APPROACH,
    CAR_WAITING_AT_STOP,
    CAR_CROSSING,
    CAR_DONE,
    PED_APPROACH,
    PED_WAITING_SIGNAL,
    PED_CROSSING,
    PED_DONE,
};

struct Actor {
    int actor_id;
    ActorType type;
    float x;
    float y;
    float heading;
    float velocity;
    float preferred_velocity = 0.0f;
    int route_id = -1;
    MotionState motion_state = MotionState::STATIC;
    int state_tick = -1;
};

}  // namespace sim
