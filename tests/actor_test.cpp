#include <gtest/gtest.h>

#include "sim/Actor.h"
#include "sim/Scene.h"
#include "sim/WorldGrid.h"

using namespace sim;

class ActorTest : public ::testing::Test {
protected:
    Scene scene{42};
};

TEST_F(ActorTest, SpawnsCorrectNumberOfActors) {
    EXPECT_EQ(static_cast<int>(scene.actors().size()), DEFAULT_CAR_COUNT + DEFAULT_PED_COUNT);
}

TEST_F(ActorTest, SpawnsCorrectNumberOfCars) {
    int car_count = 0;
    for (const auto& actor : scene.actors()) {
        if (actor.type == ActorType::CAR) {
            ++car_count;
        }
    }
    EXPECT_EQ(car_count, DEFAULT_CAR_COUNT);
}

TEST_F(ActorTest, SpawnsCorrectNumberOfPedestrians) {
    int ped_count = 0;
    for (const auto& actor : scene.actors()) {
        if (actor.type == ActorType::PEDESTRIAN) {
            ++ped_count;
        }
    }
    EXPECT_EQ(ped_count, DEFAULT_PED_COUNT);
}

TEST_F(ActorTest, ActorIdsAreSequentialFromZero) {
    const auto& actors = scene.actors();
    for (int i = 0; i < static_cast<int>(actors.size()); ++i) {
        EXPECT_EQ(actors[i].actor_id, i);
    }
}

TEST_F(ActorTest, CarsHavePositiveVelocity) {
    for (const auto& actor : scene.actors()) {
        if (actor.type == ActorType::CAR) {
            EXPECT_GT(actor.velocity, 0.0f);
        }
    }
}

TEST_F(ActorTest, CarsAreWithinWorldBounds) {
    for (const auto& actor : scene.actors()) {
        if (actor.type == ActorType::CAR) {
            EXPECT_GT(actor.x, -WORLD_HALF_EXTENT);
            EXPECT_LT(actor.x, WORLD_HALF_EXTENT);
            EXPECT_GT(actor.y, -WORLD_HALF_EXTENT);
            EXPECT_LT(actor.y, WORLD_HALF_EXTENT);
        }
    }
}

TEST_F(ActorTest, CarsSpawnOnDrivableOrLaneTiles) {
    const WorldGrid& grid = scene.world();
    for (const auto& actor : scene.actors()) {
        if (actor.type == ActorType::CAR) {
            const TileType tile = grid.tileAt(actor.x, actor.y);
            const bool on_road = tile == TileType::DRIVABLE || tile == TileType::LANE;
            EXPECT_TRUE(on_road);
        }
    }
}

TEST_F(ActorTest, PedestriansHavePositiveVelocity) {
    for (const auto& actor : scene.actors()) {
        if (actor.type == ActorType::PEDESTRIAN) {
            EXPECT_GT(actor.velocity, 0.0f);
        }
    }
}

TEST_F(ActorTest, PedestriansAreWithinWorldBounds) {
    for (const auto& actor : scene.actors()) {
        if (actor.type == ActorType::PEDESTRIAN) {
            EXPECT_GT(actor.x, -WORLD_HALF_EXTENT);
            EXPECT_LT(actor.x, WORLD_HALF_EXTENT);
            EXPECT_GT(actor.y, -WORLD_HALF_EXTENT);
            EXPECT_LT(actor.y, WORLD_HALF_EXTENT);
        }
    }
}

TEST_F(ActorTest, PedestriansSpawnOnSidewalk) {
    const WorldGrid& grid = scene.world();
    for (const auto& actor : scene.actors()) {
        if (actor.type == ActorType::PEDESTRIAN) {
            EXPECT_EQ(grid.tileAt(actor.x, actor.y), TileType::SIDEWALK);
        }
    }
}

TEST_F(ActorTest, SceneStoresSeed) {
    EXPECT_EQ(scene.seed(), 42u);
}
