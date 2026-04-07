#include "sim/MetadataExporter.h"

#include <iomanip>

namespace sim {

const char* MetadataExporter::actorTypeLabel(ActorType type) {
    return type == ActorType::CAR ? "car" : "pedestrian";
}

const char* MetadataExporter::motionStateLabel(MotionState state) {
    switch (state) {
        case MotionState::STATIC:
            return "static";
        case MotionState::CAR_APPROACH:
            return "car_approach";
        case MotionState::CAR_WAITING_AT_STOP:
            return "car_waiting_at_stop";
        case MotionState::CAR_CROSSING:
            return "car_crossing";
        case MotionState::CAR_DONE:
            return "car_done";
        case MotionState::PED_APPROACH:
            return "ped_approach";
        case MotionState::PED_WAITING_SIGNAL:
            return "ped_waiting_signal";
        case MotionState::PED_CROSSING:
            return "ped_crossing";
        case MotionState::PED_DONE:
            return "ped_done";
    }

    return "unknown";
}

void MetadataExporter::writeJson(
    std::ostream& out,
    const Scene& scene,
    uint32_t seed,
    int tick,
    float time_seconds) const {
    out << std::fixed << std::setprecision(6);
    out << "{\n"
        << "  \"seed\": " << seed << ",\n"
        << "  \"tick\": " << tick << ",\n"
        << "  \"time_seconds\": " << time_seconds << ",\n"
        << "  \"actor_count\": " << scene.actors().size() << ",\n"
        << "  \"actors\": [\n";

    for (std::size_t index = 0; index < scene.actors().size(); ++index) {
        const auto& actor = scene.actors()[index];
        out << "    {\n"
            << "      \"actor_id\": " << actor.actor_id << ",\n"
            << "      \"type\": \"" << actorTypeLabel(actor.type) << "\",\n"
            << "      \"route_id\": " << actor.route_id << ",\n"
            << "      \"motion_state\": \"" << motionStateLabel(actor.motion_state) << "\",\n"
            << "      \"x\": " << actor.x << ",\n"
            << "      \"y\": " << actor.y << ",\n"
            << "      \"heading\": " << actor.heading << ",\n"
            << "      \"velocity\": " << actor.velocity << "\n"
            << "    }" << (index + 1 == scene.actors().size() ? "\n" : ",\n");
    }

    out << "  ]\n"
        << "}\n";
}

void MetadataExporter::writeCsv(
    std::ostream& out,
    const Scene& scene,
    uint32_t seed,
    int tick,
    float time_seconds) const {
    out << std::fixed << std::setprecision(6);
    out << "seed,tick,time_seconds,actor_id,type,route_id,motion_state,x,y,heading,velocity\n";
    for (const auto& actor : scene.actors()) {
        out << seed
            << ',' << tick
            << ',' << time_seconds
            << ',' << actor.actor_id
            << ',' << actorTypeLabel(actor.type)
            << ',' << actor.route_id
            << ',' << motionStateLabel(actor.motion_state)
            << ',' << actor.x
            << ',' << actor.y
            << ',' << actor.heading
            << ',' << actor.velocity
            << '\n';
    }
}

}  // namespace sim
