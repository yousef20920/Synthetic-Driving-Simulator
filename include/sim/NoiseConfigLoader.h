#pragma once

#include "sim/NoiseInjector.h"

#include <string>

namespace sim {

class NoiseConfigLoader {
public:
    NoiseConfig loadFromFile(const std::string& path) const;
};

}  // namespace sim
