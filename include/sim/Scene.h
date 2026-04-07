#pragma once

#include "sim/Actor.h"
#include "sim/WorldGrid.h"

#include <cstdint>
#include <random>
#include <vector>

namespace sim {

static constexpr int DEFAULT_CAR_COUNT = 4;
static constexpr int DEFAULT_PED_COUNT = 4;

class Scene {
public:
    explicit Scene(uint32_t seed);

    void advance(float dt, int tick);

    const std::vector<Actor>& actors() const { return actors_; }
    std::vector<Actor>& actors() { return actors_; }

    const WorldGrid& world() const { return world_; }
    WorldGrid& world() { return world_; }

    uint32_t seed() const { return seed_; }
    std::mt19937& rng() { return rng_; }

private:
    uint32_t seed_;
    std::mt19937 rng_;
    WorldGrid world_;
    std::vector<Actor> actors_;

    void spawnActors();
    void spawnCar(int next_id);
    void spawnPedestrian(int next_id);
    void advanceCar(Actor& actor, float dt, int tick);
    void advancePedestrian(Actor& actor, float dt, int tick);
    float targetLongitudinalForCar(const Actor& actor) const;
    const Actor* leadCarInLane(const Actor& actor) const;
    bool sameLane(const Actor& actor, const Actor& other) const;
    bool carHasPriority(const Actor& actor) const;
    bool anyOtherCarCrossing(int actor_id) const;
    bool anyPedestrianCrossing() const;
    bool anyCarCrossing() const;
    bool walkSignalOpen(int route_id, int tick) const;
};

}  // namespace sim
