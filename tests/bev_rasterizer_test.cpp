#include <gtest/gtest.h>

#include "sim/BevRasterizer.h"

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
                  float preferred_velocity) {
    actor.type = ActorType::CAR;
    actor.route_id = route_id;
    actor.preferred_velocity = preferred_velocity;
    actor.velocity = preferred_velocity;
    actor.motion_state = MotionState::CAR_DONE;

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

void configurePedestrian(Actor& actor, float x, float y) {
    actor.type = ActorType::PEDESTRIAN;
    actor.x = x;
    actor.y = y;
    actor.heading = kHeadingNorth;
    actor.preferred_velocity = 1.0f;
    actor.velocity = 0.0f;
    actor.motion_state = MotionState::PED_DONE;
}

}  // namespace

TEST(BevRasterizerTest, RasterIs128x128) {
    Scene scene(42);
    BevRasterizer rasterizer;
    const BevImage image = rasterizer.rasterize(scene);

    EXPECT_EQ(static_cast<int>(image.pixels.size()), GRID_SIZE);
    EXPECT_EQ(static_cast<int>(image.pixels[0].size()), GRID_SIZE);
}

TEST(BevRasterizerTest, TerrainTilesMapToSemanticClasses) {
    Scene scene(42);
    freezeAllActors(scene);
    BevRasterizer rasterizer;
    const BevImage image = rasterizer.rasterize(scene);

    EXPECT_EQ(image.pixelAtWorld(0.0f, 0.0f), SemanticClass::LANE);
    EXPECT_EQ(image.pixelAtWorld(2.0f, 2.0f), SemanticClass::DRIVABLE);
    EXPECT_EQ(image.pixelAtWorld(8.0f, 30.0f), SemanticClass::OBSTACLE);
    EXPECT_EQ(image.pixelAtWorld(-30.0f, 30.0f), SemanticClass::OBSTACLE);
}

TEST(BevRasterizerTest, CarsOverlayTerrainAsVehicleClass) {
    Scene scene(42);
    freezeAllActors(scene);
    configureCar(scene.actors()[0], 2, 10.0f, 1.75f, 0.0f);

    BevRasterizer rasterizer;
    const BevImage image = rasterizer.rasterize(scene);

    EXPECT_EQ(image.pixelAtWorld(10.0f, 1.75f), SemanticClass::VEHICLE);
    EXPECT_EQ(image.pixelAtWorld(8.0f, 1.75f), SemanticClass::VEHICLE);
}

TEST(BevRasterizerTest, PedestriansOverlayTerrainAsPedestrianClass) {
    Scene scene(42);
    freezeAllActors(scene);
    configurePedestrian(scene.actors()[4], -8.0f, 0.0f);

    BevRasterizer rasterizer;
    const BevImage image = rasterizer.rasterize(scene);

    EXPECT_EQ(image.pixelAtWorld(-8.0f, 0.0f), SemanticClass::PEDESTRIAN);
}

TEST(BevRasterizerTest, RasterizationIsDeterministicForSameSeedAndTick) {
    Scene first(42);
    Scene second(42);
    for (int tick = 0; tick < 30; ++tick) {
        first.advance(0.05f, tick);
        second.advance(0.05f, tick);
    }

    BevRasterizer rasterizer;
    const BevImage image_a = rasterizer.rasterize(first);
    const BevImage image_b = rasterizer.rasterize(second);

    EXPECT_EQ(image_a.pixels, image_b.pixels);
}
