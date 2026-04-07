#pragma once

#include "sim/Scene.h"

#include <ostream>

namespace sim {

class MetadataExporter {
public:
    void writeJson(std::ostream& out, const Scene& scene, uint32_t seed, int tick, float time_seconds) const;
    void writeCsv(std::ostream& out, const Scene& scene, uint32_t seed, int tick, float time_seconds) const;

private:
    static const char* actorTypeLabel(ActorType type);
    static const char* motionStateLabel(MotionState state);
};

}  // namespace sim
