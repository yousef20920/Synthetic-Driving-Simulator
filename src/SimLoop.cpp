#include "sim/SimLoop.h"

namespace sim {

SimLoop::SimLoop(Scene& scene, CsvLogger& logger, float dt, int num_ticks)
    : scene_(scene), logger_(logger), dt_(dt), num_ticks_(num_ticks), elapsed_time_(0.0f) {}

void SimLoop::run() {
    logger_.writeHeader();

    for (int tick = 0; tick < num_ticks_; ++tick) {
        logger_.writeActors(tick, scene_.actors());
        scene_.advance(dt_, tick);
        elapsed_time_ += dt_;
    }
}

}  // namespace sim
