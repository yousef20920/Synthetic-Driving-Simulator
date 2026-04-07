#pragma once

#include "sim/CsvLogger.h"
#include "sim/Scene.h"

namespace sim {

class SimLoop {
public:
    SimLoop(Scene& scene, CsvLogger& logger, float dt, int num_ticks);

    void run();

    float elapsedTime() const { return elapsed_time_; }

private:
    Scene& scene_;
    CsvLogger& logger_;
    float dt_;
    int num_ticks_;
    float elapsed_time_ = 0.0f;
};

}  // namespace sim
