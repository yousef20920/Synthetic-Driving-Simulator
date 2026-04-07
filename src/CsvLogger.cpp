#include "sim/CsvLogger.h"

#include <iomanip>

namespace sim {

CsvLogger::CsvLogger(std::ostream& out)
    : out_(out) {}

void CsvLogger::writeHeader() {
    out_ << "tick,actor_id,type,x,y,heading,velocity\n";
}

void CsvLogger::writeActors(int tick, const std::vector<Actor>& actors) {
    out_ << std::fixed << std::setprecision(6);
    for (const auto& actor : actors) {
        const char* type_str = actor.type == ActorType::CAR ? "car" : "pedestrian";
        out_ << tick
             << ',' << actor.actor_id
             << ',' << type_str
             << ',' << actor.x
             << ',' << actor.y
             << ',' << actor.heading
             << ',' << actor.velocity
             << '\n';
    }
}

}  // namespace sim
