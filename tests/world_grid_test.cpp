#include <gtest/gtest.h>

#include "sim/TileType.h"
#include "sim/WorldGrid.h"

using namespace sim;

class WorldGridTest : public ::testing::Test {
protected:
    WorldGrid grid;
};

TEST_F(WorldGridTest, GridIs128x128) {
    const auto& t = grid.tiles();
    EXPECT_EQ(static_cast<int>(t.size()), GRID_SIZE);
    EXPECT_EQ(static_cast<int>(t[0].size()), GRID_SIZE);
}

TEST_F(WorldGridTest, OutOfBoundsPositionReturnsOutOfBounds) {
    EXPECT_EQ(grid.tileAt(100.0f, 0.0f), TileType::OUT_OF_BOUNDS);
    EXPECT_EQ(grid.tileAt(0.0f, -100.0f), TileType::OUT_OF_BOUNDS);
    EXPECT_EQ(grid.tileAt(64.0f, 0.0f), TileType::OUT_OF_BOUNDS);
}

TEST_F(WorldGridTest, OutOfBoundsIndexReturnsOutOfBounds) {
    EXPECT_EQ(grid.tileAtIndex(-1, 0), TileType::OUT_OF_BOUNDS);
    EXPECT_EQ(grid.tileAtIndex(0, 128), TileType::OUT_OF_BOUNDS);
    EXPECT_EQ(grid.tileAtIndex(128, 0), TileType::OUT_OF_BOUNDS);
}

TEST_F(WorldGridTest, IntersectionCenterIsLane) {
    EXPECT_EQ(grid.tileAt(0.0f, 0.0f), TileType::LANE);
}

TEST_F(WorldGridTest, OffCenterIntersectionIsDrivable) {
    EXPECT_EQ(grid.tileAt(2.0f, 2.0f), TileType::DRIVABLE);
}

TEST_F(WorldGridTest, NSArmDrivableInsideRoad) {
    EXPECT_EQ(grid.tileAt(2.0f, 30.0f), TileType::DRIVABLE);
}

TEST_F(WorldGridTest, NSArmLaneMarkingAtCenterLine) {
    EXPECT_EQ(grid.tileAt(0.0f, 30.0f), TileType::LANE);
}

TEST_F(WorldGridTest, NSArmLaneMarkingAtInnerBoundary) {
    EXPECT_EQ(grid.tileAt(3.5f, 30.0f), TileType::LANE);
}

TEST_F(WorldGridTest, EWArmDrivableInsideRoad) {
    EXPECT_EQ(grid.tileAt(30.0f, 2.0f), TileType::DRIVABLE);
}

TEST_F(WorldGridTest, EWArmCenterLineIsLane) {
    EXPECT_EQ(grid.tileAt(30.0f, 0.0f), TileType::LANE);
}

TEST_F(WorldGridTest, SidewalkOutsideNSArm) {
    EXPECT_EQ(grid.tileAt(8.0f, 30.0f), TileType::SIDEWALK);
    EXPECT_EQ(grid.tileAt(-8.0f, 30.0f), TileType::SIDEWALK);
}

TEST_F(WorldGridTest, SidewalkOutsideEWArm) {
    EXPECT_EQ(grid.tileAt(30.0f, 8.0f), TileType::SIDEWALK);
    EXPECT_EQ(grid.tileAt(30.0f, -8.0f), TileType::SIDEWALK);
}

TEST_F(WorldGridTest, FarCornerIsObstacle) {
    EXPECT_EQ(grid.tileAt(-30.0f, 30.0f), TileType::OBSTACLE);
}

TEST_F(WorldGridTest, JustOutsideSidewalkIsObstacle) {
    EXPECT_EQ(grid.tileAt(10.0f, 30.0f), TileType::OBSTACLE);
}
