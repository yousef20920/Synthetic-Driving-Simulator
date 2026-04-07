#include <gtest/gtest.h>

#include "sim/MetadataExporter.h"

#include <sstream>
#include <string>

using namespace sim;

namespace {

Scene captureScene(uint32_t seed, float dt, int num_ticks) {
    Scene scene(seed);
    for (int tick = 0; tick + 1 < num_ticks; ++tick) {
        scene.advance(dt, tick);
    }
    return scene;
}

}  // namespace

TEST(MetadataExporterTest, JsonIncludesSeedTickTimestampAndActorArray) {
    const Scene scene = captureScene(42, 0.05f, 20);
    MetadataExporter exporter;
    std::ostringstream oss;

    exporter.writeJson(oss, scene, 42, 19, 0.95f);
    const std::string json = oss.str();

    EXPECT_NE(json.find("\"seed\": 42"), std::string::npos);
    EXPECT_NE(json.find("\"tick\": 19"), std::string::npos);
    EXPECT_NE(json.find("\"time_seconds\": 0.950000"), std::string::npos);
    EXPECT_NE(json.find("\"actors\": ["), std::string::npos);
    EXPECT_NE(json.find("\"type\": \"car\""), std::string::npos);
}

TEST(MetadataExporterTest, CsvIncludesHeaderAndAllActors) {
    const Scene scene = captureScene(42, 0.05f, 10);
    MetadataExporter exporter;
    std::ostringstream oss;

    exporter.writeCsv(oss, scene, 42, 9, 0.45f);
    const std::string csv = oss.str();

    EXPECT_NE(csv.find("seed,tick,time_seconds,actor_id,type,route_id,motion_state,x,y,heading,velocity"),
              std::string::npos);
    EXPECT_NE(csv.find("42,9,0.450000,0,car"), std::string::npos);
    EXPECT_NE(csv.find("42,9,0.450000,4,pedestrian"), std::string::npos);
}

TEST(MetadataExporterTest, JsonAndCsvAreDeterministicForSameSnapshot) {
    const Scene scene = captureScene(7, 0.05f, 15);
    MetadataExporter exporter;
    std::ostringstream json_a;
    std::ostringstream json_b;
    std::ostringstream csv_a;
    std::ostringstream csv_b;

    exporter.writeJson(json_a, scene, 7, 14, 0.70f);
    exporter.writeJson(json_b, scene, 7, 14, 0.70f);
    exporter.writeCsv(csv_a, scene, 7, 14, 0.70f);
    exporter.writeCsv(csv_b, scene, 7, 14, 0.70f);

    EXPECT_EQ(json_a.str(), json_b.str());
    EXPECT_EQ(csv_a.str(), csv_b.str());
}
