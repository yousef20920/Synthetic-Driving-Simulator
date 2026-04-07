#include "sim/NoiseConfigLoader.h"

#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>

namespace sim {
namespace {

std::string trim(const std::string& value) {
    std::size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
        ++begin;
    }

    std::size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
        --end;
    }

    return value.substr(begin, end - begin);
}

void assignConfigValue(NoiseConfig& config, const std::string& key, const std::string& value) {
    if (key == "obstacle_intensity") {
        config.obstacle_intensity = static_cast<std::uint8_t>(std::stoi(value));
        return;
    }
    if (key == "drivable_intensity") {
        config.drivable_intensity = static_cast<std::uint8_t>(std::stoi(value));
        return;
    }
    if (key == "lane_intensity") {
        config.lane_intensity = static_cast<std::uint8_t>(std::stoi(value));
        return;
    }
    if (key == "vehicle_intensity") {
        config.vehicle_intensity = static_cast<std::uint8_t>(std::stoi(value));
        return;
    }
    if (key == "pedestrian_intensity") {
        config.pedestrian_intensity = static_cast<std::uint8_t>(std::stoi(value));
        return;
    }
    if (key == "speckle_intensity_min") {
        config.speckle_intensity_min = static_cast<std::uint8_t>(std::stoi(value));
        return;
    }
    if (key == "speckle_intensity_max") {
        config.speckle_intensity_max = static_cast<std::uint8_t>(std::stoi(value));
        return;
    }
    if (key == "jitter_amplitude") {
        config.jitter_amplitude = static_cast<std::uint8_t>(std::stoi(value));
        return;
    }
    if (key == "dropout_probability") {
        config.dropout_probability = std::stof(value);
        return;
    }
    if (key == "speckle_probability") {
        config.speckle_probability = std::stof(value);
        return;
    }

    throw std::runtime_error("Unknown noise config key: " + key);
}

}  // namespace

NoiseConfig NoiseConfigLoader::loadFromFile(const std::string& path) const {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("Could not open noise config: " + path);
    }

    NoiseConfig config;
    std::string line;
    int line_number = 0;

    while (std::getline(input, line)) {
        ++line_number;
        const std::size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line.erase(comment_pos);
        }

        const std::string trimmed = trim(line);
        if (trimmed.empty()) {
            continue;
        }

        const std::size_t colon_pos = trimmed.find(':');
        if (colon_pos == std::string::npos) {
            throw std::runtime_error(
                "Invalid noise config line " + std::to_string(line_number) + " in " + path);
        }

        const std::string key = trim(trimmed.substr(0, colon_pos));
        const std::string value = trim(trimmed.substr(colon_pos + 1));
        if (key.empty() || value.empty()) {
            throw std::runtime_error(
                "Invalid noise config line " + std::to_string(line_number) + " in " + path);
        }

        assignConfigValue(config, key, value);
    }

    return config;
}

}  // namespace sim
