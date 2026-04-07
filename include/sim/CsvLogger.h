#pragma once

#include "sim/Actor.h"

#include <ostream>
#include <vector>

namespace sim {

class CsvLogger {
public:
    explicit CsvLogger(std::ostream& out);

    void writeHeader();
    void writeActors(int tick, const std::vector<Actor>& actors);

private:
    std::ostream& out_;
};

}  // namespace sim
