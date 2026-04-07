#include "sim/Scene.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace sim {
namespace {

constexpr float kSpawnMin = 15.0f;
constexpr float kSpawnMax = 55.0f;
constexpr float kCarVelocityMin = 5.0f;
constexpr float kCarVelocityMax = 15.0f;
constexpr float kPedPositionMin = -55.0f;
constexpr float kPedPositionMax = 55.0f;
constexpr float kPedVelocityMin = 0.5f;
constexpr float kPedVelocityMax = 2.0f;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kHeadingNorth = kPi / 2.0f;
constexpr float kHeadingSouth = -kPi / 2.0f;
constexpr float kHeadingEast = 0.0f;
constexpr float kHeadingWest = kPi;
constexpr float kSidewalkCenter = ROAD_HALF_WIDTH + (SIDEWALK_WIDTH_M / 2.0f);
constexpr float kStopLine = 8.0f;
constexpr float kBrakeDistance = 6.0f;
constexpr float kCarFollowGap = 5.5f;
constexpr float kLaneTolerance = 0.001f;
constexpr int kMinStopTicks = 2;
constexpr int kSignalCycleTicks = 40;
constexpr int kSignalHalfCycleTicks = kSignalCycleTicks / 2;

bool routeIsVertical(int route_id) {
    return route_id == 0 || route_id == 1;
}

float clampStep(float current, float target, float max_step) {
    if (current < target) {
        return std::min(current + max_step, target);
    }
    return std::max(current - max_step, target);
}

float directionSign(int route_id) {
    switch (route_id) {
        case 0:
        case 2:
            return -1.0f;
        case 1:
        case 3:
            return 1.0f;
        default:
            return 0.0f;
    }
}

float stopLineForRoute(int route_id) {
    switch (route_id) {
        case 0:
        case 2:
            return kStopLine;
        case 1:
        case 3:
            return -kStopLine;
        default:
            return 0.0f;
    }
}

bool passedStopLine(float value, int route_id) {
    return directionSign(route_id) < 0.0f ? value <= -kStopLine : value >= kStopLine;
}

float getLongitudinal(const Actor& actor) {
    return routeIsVertical(actor.route_id) ? actor.y : actor.x;
}

float getLateral(const Actor& actor) {
    return routeIsVertical(actor.route_id) ? actor.x : actor.y;
}

void setLongitudinal(Actor& actor, float value) {
    if (routeIsVertical(actor.route_id)) {
        actor.y = value;
    } else {
        actor.x = value;
    }
}

void setApproachHeading(Actor& actor, float current, float target) {
    if (actor.route_id == 0 || actor.route_id == 1) {
        actor.heading = current <= target ? kHeadingNorth : kHeadingSouth;
    } else {
        actor.heading = current <= target ? kHeadingEast : kHeadingWest;
    }
}

void setCrossingHeading(Actor& actor) {
    switch (actor.route_id) {
        case 0:
            actor.heading = kHeadingEast;
            break;
        case 1:
            actor.heading = kHeadingWest;
            break;
        case 2:
            actor.heading = kHeadingNorth;
            break;
        case 3:
            actor.heading = kHeadingSouth;
            break;
        default:
            break;
    }
}

}  // namespace

Scene::Scene(uint32_t seed)
    : seed_(seed), rng_(seed), world_(), actors_() {
    actors_.reserve(DEFAULT_CAR_COUNT + DEFAULT_PED_COUNT);
    spawnActors();
}

void Scene::advance(float dt, int tick) {
    for (auto& actor : actors_) {
        if (actor.type == ActorType::CAR) {
            advanceCar(actor, dt, tick);
        }
    }

    for (auto& actor : actors_) {
        if (actor.type == ActorType::PEDESTRIAN) {
            advancePedestrian(actor, dt, tick);
        }
    }
}

void Scene::spawnActors() {
    for (int i = 0; i < DEFAULT_CAR_COUNT; ++i) {
        spawnCar(i);
    }

    for (int i = 0; i < DEFAULT_PED_COUNT; ++i) {
        spawnPedestrian(DEFAULT_CAR_COUNT + i);
    }
}

void Scene::spawnCar(int next_id) {
    std::uniform_int_distribution<int> arm_dist(0, 3);
    std::uniform_int_distribution<int> lane_dist(0, 1);
    std::uniform_real_distribution<float> dist_dist(kSpawnMin, kSpawnMax);
    std::uniform_real_distribution<float> vel_dist(kCarVelocityMin, kCarVelocityMax);

    const int arm = arm_dist(rng_);
    const int lane_idx = lane_dist(rng_);
    const float dist = dist_dist(rng_);
    const float velocity = vel_dist(rng_);
    const float lateral_offset = (lane_idx == 0) ? (0.5f * LANE_WIDTH_M) : (1.5f * LANE_WIDTH_M);

    Actor actor{};
    actor.actor_id = next_id;
    actor.type = ActorType::CAR;
    actor.velocity = velocity;
    actor.preferred_velocity = velocity;
    actor.route_id = arm;
    actor.motion_state = MotionState::CAR_APPROACH;

    switch (arm) {
        case 0:
            actor.x = lateral_offset;
            actor.y = dist;
            actor.heading = kHeadingSouth;
            break;
        case 1:
            actor.x = -lateral_offset;
            actor.y = -dist;
            actor.heading = kHeadingNorth;
            break;
        case 2:
            actor.x = dist;
            actor.y = lateral_offset;
            actor.heading = kHeadingWest;
            break;
        case 3:
            actor.x = -dist;
            actor.y = -lateral_offset;
            actor.heading = kHeadingEast;
            break;
        default:
            actor.x = 0.0f;
            actor.y = 0.0f;
            actor.heading = 0.0f;
            break;
    }

    actors_.push_back(actor);
}

void Scene::spawnPedestrian(int next_id) {
    std::uniform_int_distribution<int> side_dist(0, 3);
    std::uniform_real_distribution<float> pos_dist(kPedPositionMin, kPedPositionMax);
    std::uniform_real_distribution<float> vel_dist(kPedVelocityMin, kPedVelocityMax);

    const int side = side_dist(rng_);
    const float pos = pos_dist(rng_);
    const float velocity = vel_dist(rng_);

    Actor actor{};
    actor.actor_id = next_id;
    actor.type = ActorType::PEDESTRIAN;
    actor.velocity = velocity;
    actor.preferred_velocity = velocity;
    actor.route_id = side;
    actor.motion_state = MotionState::PED_APPROACH;

    switch (side) {
        case 0:
            actor.x = -kSidewalkCenter;
            actor.y = pos;
            actor.heading = kHeadingNorth;
            break;
        case 1:
            actor.x = kSidewalkCenter;
            actor.y = pos;
            actor.heading = kHeadingSouth;
            break;
        case 2:
            actor.x = pos;
            actor.y = -kSidewalkCenter;
            actor.heading = kHeadingEast;
            break;
        case 3:
            actor.x = pos;
            actor.y = kSidewalkCenter;
            actor.heading = kHeadingWest;
            break;
        default:
            actor.x = 0.0f;
            actor.y = 0.0f;
            actor.heading = 0.0f;
            break;
    }

    actors_.push_back(actor);
}

bool Scene::sameLane(const Actor& actor, const Actor& other) const {
    return actor.type == ActorType::CAR &&
           other.type == ActorType::CAR &&
           actor.actor_id != other.actor_id &&
           actor.route_id == other.route_id &&
           std::fabs(getLateral(actor) - getLateral(other)) <= kLaneTolerance;
}

const Actor* Scene::leadCarInLane(const Actor& actor) const {
    const float direction = directionSign(actor.route_id);
    const float actor_longitudinal = getLongitudinal(actor);
    const Actor* lead = nullptr;

    for (const auto& other : actors_) {
        if (!sameLane(actor, other)) {
            continue;
        }

        const float other_longitudinal = getLongitudinal(other);
        const bool ahead = direction < 0.0f ? other_longitudinal < actor_longitudinal
                                            : other_longitudinal > actor_longitudinal;
        if (!ahead) {
            continue;
        }

        if (lead == nullptr) {
            lead = &other;
            continue;
        }

        const float lead_longitudinal = getLongitudinal(*lead);
        const bool nearer = direction < 0.0f ? other_longitudinal > lead_longitudinal
                                             : other_longitudinal < lead_longitudinal;
        if (nearer) {
            lead = &other;
        }
    }

    return lead;
}

float Scene::targetLongitudinalForCar(const Actor& actor) const {
    const float direction = directionSign(actor.route_id);
    float target = stopLineForRoute(actor.route_id);

    if (const Actor* lead = leadCarInLane(actor); lead != nullptr) {
        const float queue_target = getLongitudinal(*lead) - (direction * kCarFollowGap);
        target = direction < 0.0f ? std::max(target, queue_target) : std::min(target, queue_target);
    }

    return target;
}

bool Scene::carHasPriority(const Actor& actor) const {
    if (actor.motion_state != MotionState::CAR_WAITING_AT_STOP) {
        return false;
    }

    if (anyOtherCarCrossing(actor.actor_id)) {
        return false;
    }

    for (const auto& other : actors_) {
        if (other.type != ActorType::CAR || other.actor_id == actor.actor_id) {
            continue;
        }
        if (other.motion_state != MotionState::CAR_WAITING_AT_STOP) {
            continue;
        }

        if (other.state_tick < actor.state_tick) {
            return false;
        }
        if (other.state_tick == actor.state_tick && other.actor_id < actor.actor_id) {
            return false;
        }
    }

    return true;
}

bool Scene::anyOtherCarCrossing(int actor_id) const {
    for (const auto& other : actors_) {
        if (other.type == ActorType::CAR &&
            other.actor_id != actor_id &&
            other.motion_state == MotionState::CAR_CROSSING) {
            return true;
        }
    }
    return false;
}

bool Scene::anyPedestrianCrossing() const {
    for (const auto& actor : actors_) {
        if (actor.type == ActorType::PEDESTRIAN &&
            actor.motion_state == MotionState::PED_CROSSING) {
            return true;
        }
    }
    return false;
}

bool Scene::anyCarCrossing() const {
    for (const auto& actor : actors_) {
        if (actor.type == ActorType::CAR &&
            actor.motion_state == MotionState::CAR_CROSSING) {
            return true;
        }
    }
    return false;
}

bool Scene::walkSignalOpen(int route_id, int tick) const {
    const int phase = tick % kSignalCycleTicks;
    const bool horizontal_walk = phase < kSignalHalfCycleTicks;
    const bool horizontal_route = route_id == 0 || route_id == 1;
    return horizontal_route ? horizontal_walk : !horizontal_walk;
}

void Scene::advanceCar(Actor& actor, float dt, int tick) {
    const float current = getLongitudinal(actor);
    const float direction = directionSign(actor.route_id);
    const float stop_line = stopLineForRoute(actor.route_id);

    if (actor.motion_state == MotionState::CAR_APPROACH) {
        const float target = targetLongitudinalForCar(actor);
        const float distance_to_target = std::fabs(current - target);
        const float speed_scale = std::clamp(distance_to_target / kBrakeDistance, 0.25f, 1.0f);
        const float step = actor.preferred_velocity * speed_scale * dt;
        const float next = current + direction * step;

        actor.velocity = actor.preferred_velocity * speed_scale;
        if ((direction < 0.0f && next <= target) || (direction > 0.0f && next >= target)) {
            setLongitudinal(actor, target);
            actor.velocity = 0.0f;
            if (target == stop_line) {
                actor.motion_state = MotionState::CAR_WAITING_AT_STOP;
                actor.state_tick = tick;
            }
            return;
        }

        setLongitudinal(actor, next);
        return;
    }

    if (actor.motion_state == MotionState::CAR_WAITING_AT_STOP) {
        actor.velocity = 0.0f;
        setLongitudinal(actor, stop_line);

        if (tick - actor.state_tick < kMinStopTicks) {
            return;
        }
        if (!carHasPriority(actor) || anyPedestrianCrossing()) {
            return;
        }

        actor.motion_state = MotionState::CAR_CROSSING;
    }

    if (actor.motion_state == MotionState::CAR_CROSSING || actor.motion_state == MotionState::CAR_DONE) {
        const float next = current + direction * actor.preferred_velocity * dt;
        setLongitudinal(actor, next);
        actor.velocity = actor.preferred_velocity;

        if (actor.motion_state == MotionState::CAR_CROSSING && passedStopLine(next, actor.route_id)) {
            actor.motion_state = MotionState::CAR_DONE;
        }
    }
}

void Scene::advancePedestrian(Actor& actor, float dt, int tick) {
    const float step = actor.preferred_velocity * dt;

    if (actor.motion_state == MotionState::PED_APPROACH) {
        if (actor.route_id == 0 || actor.route_id == 1) {
            setApproachHeading(actor, actor.y, 0.0f);
            actor.y = clampStep(actor.y, 0.0f, step);
            actor.x = actor.route_id == 0 ? -kSidewalkCenter : kSidewalkCenter;
            actor.velocity = actor.preferred_velocity;
            if (actor.y == 0.0f) {
                actor.motion_state = MotionState::PED_WAITING_SIGNAL;
                actor.state_tick = tick;
                actor.velocity = 0.0f;
            }
            return;
        }

        setApproachHeading(actor, actor.x, 0.0f);
        actor.x = clampStep(actor.x, 0.0f, step);
        actor.y = actor.route_id == 2 ? -kSidewalkCenter : kSidewalkCenter;
        actor.velocity = actor.preferred_velocity;
        if (actor.x == 0.0f) {
            actor.motion_state = MotionState::PED_WAITING_SIGNAL;
            actor.state_tick = tick;
            actor.velocity = 0.0f;
        }
        return;
    }

    if (actor.motion_state == MotionState::PED_WAITING_SIGNAL) {
        actor.velocity = 0.0f;
        if (!walkSignalOpen(actor.route_id, tick) || anyCarCrossing()) {
            return;
        }
        actor.motion_state = MotionState::PED_CROSSING;
    }

    if (actor.motion_state == MotionState::PED_CROSSING) {
        setCrossingHeading(actor);
        actor.velocity = actor.preferred_velocity;

        switch (actor.route_id) {
            case 0:
                actor.y = 0.0f;
                actor.x = clampStep(actor.x, kSidewalkCenter, step);
                if (actor.x == kSidewalkCenter) {
                    actor.motion_state = MotionState::PED_DONE;
                    actor.velocity = 0.0f;
                }
                break;
            case 1:
                actor.y = 0.0f;
                actor.x = clampStep(actor.x, -kSidewalkCenter, step);
                if (actor.x == -kSidewalkCenter) {
                    actor.motion_state = MotionState::PED_DONE;
                    actor.velocity = 0.0f;
                }
                break;
            case 2:
                actor.x = 0.0f;
                actor.y = clampStep(actor.y, kSidewalkCenter, step);
                if (actor.y == kSidewalkCenter) {
                    actor.motion_state = MotionState::PED_DONE;
                    actor.velocity = 0.0f;
                }
                break;
            case 3:
                actor.x = 0.0f;
                actor.y = clampStep(actor.y, -kSidewalkCenter, step);
                if (actor.y == -kSidewalkCenter) {
                    actor.motion_state = MotionState::PED_DONE;
                    actor.velocity = 0.0f;
                }
                break;
            default:
                break;
        }
        return;
    }

    if (actor.motion_state == MotionState::PED_DONE) {
        actor.velocity = 0.0f;
    }
}

}  // namespace sim
