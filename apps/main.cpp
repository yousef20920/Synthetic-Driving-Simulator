// sim_runner: CLI entry point for the Synthetic Driving Simulator.
#include "sim/BevRasterizer.h"
#include "sim/CsvLogger.h"
#include "sim/MetadataExporter.h"
#include "sim/NoiseConfigLoader.h"
#include "sim/NoiseInjector.h"
#include "sim/Scene.h"
#include "sim/SimLoop.h"
#include "sim/WorldGrid.h"

#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

char tileGlyph(sim::TileType tile) {
    switch (tile) {
        case sim::TileType::OUT_OF_BOUNDS:
            return ' ';
        case sim::TileType::DRIVABLE:
            return '.';
        case sim::TileType::LANE:
            return '=';
        case sim::TileType::SIDEWALK:
            return 's';
        case sim::TileType::OBSTACLE:
            return '#';
    }

    return '?';
}

const char* semanticLabel(sim::SemanticClass semantic_class) {
    switch (semantic_class) {
        case sim::SemanticClass::DRIVABLE:
            return "drivable";
        case sim::SemanticClass::LANE:
            return "lane";
        case sim::SemanticClass::VEHICLE:
            return "vehicle";
        case sim::SemanticClass::PEDESTRIAN:
            return "pedestrian";
        case sim::SemanticClass::OBSTACLE:
            return "obstacle";
    }

    return "unknown";
}

const char* noisePresetLabel(sim::NoisePreset preset) {
    switch (preset) {
        case sim::NoisePreset::LOW:
            return "low";
        case sim::NoisePreset::HIGH:
            return "high";
    }

    return "low";
}

std::string defaultNoiseConfigPath(sim::NoisePreset preset) {
    const char* file_name = preset == sim::NoisePreset::LOW ? "noise_low.yaml" : "noise_high.yaml";
#ifdef SIM_PROJECT_SOURCE_DIR
    return std::string(SIM_PROJECT_SOURCE_DIR) + "/configs/" + file_name;
#else
    return std::string("configs/") + file_name;
#endif
}

bool parseNoisePreset(const std::string& value, sim::NoisePreset& preset) {
    if (value == "low") {
        preset = sim::NoisePreset::LOW;
        return true;
    }
    if (value == "high") {
        preset = sim::NoisePreset::HIGH;
        return true;
    }
    return false;
}

void printUsage(const char* prog) {
    std::fprintf(
        stderr,
        "Usage: %s [--seed N] [--dt SECONDS] [--ticks N] [--out PATH]\n"
        "       %s --dump-grid\n"
        "       %s [--seed N] [--dt SECONDS] [--ticks N] --dump-grid-html [PATH]\n"
        "       %s [--seed N] [--dt SECONDS] [--ticks N] --dump-bev-ppm [PATH]\n"
        "       %s [--seed N] [--dt SECONDS] [--ticks N] [--noise PRESET] [--noise-config PATH] --dump-noisy-bev-pgm [PATH]\n"
        "       %s [--seed N] [--dt SECONDS] [--ticks N] --dump-metadata-json [PATH]\n"
        "       %s [--seed N] [--dt SECONDS] [--ticks N] --dump-metadata-csv [PATH]\n"
        "  --seed N              Random seed (default: 42)\n"
        "  --dt SECONDS          Fixed timestep in seconds (default: 0.05)\n"
        "  --ticks N             Number of ticks (default: 100)\n"
        "  --noise PRESET        Noise preset name backed by YAML: low or high (default: low)\n"
        "  --noise-config PATH   Explicit YAML noise config path for noisy BEV export\n"
        "  --out PATH            Output CSV file path; '-' or omit for stdout\n"
        "  --dump-grid           Print the 128x128 world grid as ASCII\n"
        "  --dump-bev-ppm        Write a colorized clean semantic BEV snapshot (PPM)\n"
        "  --dump-noisy-bev-pgm  Write a noisy single-channel BEV input snapshot (PGM)\n"
        "  --dump-metadata-json  Write frame metadata JSON for the exported snapshot tick\n"
        "  --dump-metadata-csv   Write frame metadata CSV for the exported snapshot tick\n"
        "  --dump-grid-html      Write a self-contained animated HTML scene viewer\n",
        prog,
        prog,
        prog,
        prog,
        prog,
        prog,
        prog);
}

void dumpGrid() {
    const sim::WorldGrid grid;

    std::printf("Legend: # obstacle, s sidewalk, . drivable, = lane\n");
    std::printf("Orientation: top = +Y (north), left = -X (west)\n");

    for (int row = sim::GRID_SIZE - 1; row >= 0; --row) {
        for (int col = 0; col < sim::GRID_SIZE; ++col) {
            std::putchar(tileGlyph(grid.tileAtIndex(col, row)));
        }
        std::putchar('\n');
    }
}

std::vector<std::vector<sim::Actor>> captureFrames(uint32_t seed, float dt, int num_ticks) {
    sim::Scene scene(seed);
    std::vector<std::vector<sim::Actor>> frames;
    frames.reserve(static_cast<std::size_t>(num_ticks));

    for (int tick = 0; tick < num_ticks; ++tick) {
        frames.push_back(scene.actors());
        scene.advance(dt, tick);
    }

    return frames;
}

sim::Scene captureSceneAtLoggedTick(uint32_t seed, float dt, int num_ticks) {
    sim::Scene scene(seed);
    for (int tick = 0; tick + 1 < num_ticks; ++tick) {
        scene.advance(dt, tick);
    }
    return scene;
}

void writeFrameData(std::FILE* file, const std::vector<std::vector<sim::Actor>>& frames) {
    std::fputs("    const frames = [\n", file);
    for (std::size_t frame_index = 0; frame_index < frames.size(); ++frame_index) {
        std::fputs("      [", file);
        const auto& frame = frames[frame_index];
        for (std::size_t actor_index = 0; actor_index < frame.size(); ++actor_index) {
            const auto& actor = frame[actor_index];
            std::fprintf(
                file,
                "[%d,%d,%.6f,%.6f,%.6f,%.6f]%s",
                actor.actor_id,
                static_cast<int>(actor.type),
                actor.x,
                actor.y,
                actor.heading,
                actor.velocity,
                actor_index + 1 == frame.size() ? "" : ",");
        }
        std::fputs(frame_index + 1 == frames.size() ? "]\n" : "],\n", file);
    }
    std::fputs("    ];\n", file);
}

void writeTileData(std::FILE* file, const sim::WorldGrid& grid) {
    std::fputs("    const tiles = [\n", file);
    for (int row = sim::GRID_SIZE - 1; row >= 0; --row) {
        std::fputs("      [", file);
        for (int col = 0; col < sim::GRID_SIZE; ++col) {
            const auto tile = static_cast<int>(grid.tileAtIndex(col, row));
            std::fprintf(file, "%d%s", tile, col + 1 == sim::GRID_SIZE ? "" : ",");
        }
        std::fputs(row == 0 ? "]\n" : "],\n", file);
    }
    std::fputs("    ];\n", file);
}

bool writeGridHtml(const char* output_path, uint32_t seed, float dt, int num_ticks) {
    const sim::WorldGrid grid;
    const auto frames = captureFrames(seed, dt, num_ticks);

    std::FILE* file = std::fopen(output_path, "w");
    if (file == nullptr) {
        std::perror("failed to open output file");
        return false;
    }

    std::fputs(
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"UTF-8\" />\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />\n"
        "  <title>Synthetic Driving Simulator Scene Viewer</title>\n"
        "  <style>\n"
        "    :root {\n"
        "      --bg: #efe5d0;\n"
        "      --panel: rgba(255, 250, 240, 0.9);\n"
        "      --ink: #1e1812;\n"
        "      --muted: #6f6254;\n"
        "      --obstacle: #2b2117;\n"
        "      --sidewalk: #d9c4a6;\n"
        "      --drivable: #5f6358;\n"
        "      --lane: #f5d061;\n"
        "      --car: #be4a2f;\n"
        "      --car-top: #ffd7c7;\n"
        "      --ped: #0f7c73;\n"
        "      --trail-car: rgba(190, 74, 47, 0.18);\n"
        "      --trail-ped: rgba(15, 124, 115, 0.2);\n"
        "      --accent: #23413b;\n"
        "      --grid: rgba(30, 24, 18, 0.08);\n"
        "    }\n"
        "    * { box-sizing: border-box; }\n"
        "    body {\n"
        "      margin: 0;\n"
        "      min-height: 100vh;\n"
        "      font-family: Georgia, 'Times New Roman', serif;\n"
        "      color: var(--ink);\n"
        "      background:\n"
        "        radial-gradient(circle at top left, rgba(255,255,255,0.65), transparent 35%),\n"
        "        linear-gradient(180deg, #f7f0e2 0%, var(--bg) 100%);\n"
        "    }\n"
        "    .layout {\n"
        "      display: grid;\n"
        "      grid-template-columns: minmax(300px, 380px) 1fr;\n"
        "      gap: 24px;\n"
        "      padding: 24px;\n"
        "      align-items: start;\n"
        "    }\n"
        "    .panel {\n"
        "      background: var(--panel);\n"
        "      backdrop-filter: blur(10px);\n"
        "      border: 1px solid rgba(30, 24, 18, 0.1);\n"
        "      border-radius: 24px;\n"
        "      padding: 24px;\n"
        "      box-shadow: 0 18px 50px rgba(80, 58, 32, 0.12);\n"
        "    }\n"
        "    h1 {\n"
        "      margin: 0 0 12px;\n"
        "      font-size: clamp(2rem, 4vw, 3.5rem);\n"
        "      line-height: 0.95;\n"
        "      letter-spacing: -0.04em;\n"
        "    }\n"
        "    .lede {\n"
        "      margin: 0 0 18px;\n"
        "      font-size: 1rem;\n"
        "      line-height: 1.5;\n"
        "      color: var(--muted);\n"
        "    }\n"
        "    .legend {\n"
        "      display: grid;\n"
        "      gap: 10px;\n"
        "      margin: 18px 0 22px;\n"
        "    }\n"
        "    .legend-item {\n"
        "      display: grid;\n"
        "      grid-template-columns: 18px 1fr;\n"
        "      gap: 10px;\n"
        "      align-items: center;\n"
        "      font-size: 0.98rem;\n"
        "    }\n"
        "    .swatch {\n"
        "      width: 18px;\n"
        "      height: 18px;\n"
        "      border-radius: 5px;\n"
        "      border: 1px solid rgba(30, 24, 18, 0.16);\n"
        "    }\n"
        "    .meta {\n"
        "      display: grid;\n"
        "      gap: 8px;\n"
        "      padding-top: 18px;\n"
        "      border-top: 1px solid rgba(30, 24, 18, 0.1);\n"
        "      font-size: 0.92rem;\n"
        "      color: var(--muted);\n"
        "    }\n"
        "    .viewer {\n"
        "      display: grid;\n"
        "      gap: 16px;\n"
        "    }\n"
        "    .controls {\n"
        "      display: grid;\n"
        "      gap: 16px;\n"
        "    }\n"
        "    .toolbar {\n"
        "      display: flex;\n"
        "      flex-wrap: wrap;\n"
        "      gap: 12px;\n"
        "      align-items: center;\n"
        "    }\n"
        "    button, select {\n"
        "      font: inherit;\n"
        "      color: var(--ink);\n"
        "      background: rgba(255, 255, 255, 0.88);\n"
        "      border: 1px solid rgba(30, 24, 18, 0.14);\n"
        "      border-radius: 999px;\n"
        "      padding: 10px 16px;\n"
        "      cursor: pointer;\n"
        "    }\n"
        "    button.primary {\n"
        "      background: var(--accent);\n"
        "      color: #f8f3ea;\n"
        "      border-color: transparent;\n"
        "    }\n"
        "    .timeline {\n"
        "      display: grid;\n"
        "      gap: 8px;\n"
        "    }\n"
        "    .timeline label {\n"
        "      font-size: 0.85rem;\n"
        "      color: var(--muted);\n"
        "      text-transform: uppercase;\n"
        "      letter-spacing: 0.08em;\n"
        "    }\n"
        "    input[type='range'] {\n"
        "      width: 100%;\n"
        "      accent-color: var(--accent);\n"
        "    }\n"
        "    .mini-grid {\n"
        "      display: grid;\n"
        "      grid-template-columns: repeat(2, minmax(0, 1fr));\n"
        "      gap: 10px;\n"
        "      margin-top: 6px;\n"
        "    }\n"
        "    .stat {\n"
        "      padding: 12px 14px;\n"
        "      border-radius: 16px;\n"
        "      background: rgba(255,255,255,0.56);\n"
        "      border: 1px solid rgba(30, 24, 18, 0.08);\n"
        "    }\n"
        "    .stat-label {\n"
        "      display: block;\n"
        "      font-size: 0.8rem;\n"
        "      letter-spacing: 0.08em;\n"
        "      text-transform: uppercase;\n"
        "      color: var(--muted);\n"
        "      margin-bottom: 4px;\n"
        "    }\n"
        "    .stat-value {\n"
        "      font-size: 1.1rem;\n"
        "      font-weight: 700;\n"
        "    }\n"
        "    .viewport {\n"
        "      background: rgba(255,255,255,0.72);\n"
        "      border-radius: 28px;\n"
        "      padding: 20px;\n"
        "      border: 1px solid rgba(30, 24, 18, 0.08);\n"
        "      box-shadow: inset 0 1px 0 rgba(255,255,255,0.8);\n"
        "    }\n"
        "    canvas {\n"
        "      display: block;\n"
        "      width: min(80vw, 820px);\n"
        "      height: auto;\n"
        "      max-width: 100%;\n"
        "      image-rendering: pixelated;\n"
        "      border-radius: 18px;\n"
        "      background: #ede6d6;\n"
        "      border: 1px solid rgba(30, 24, 18, 0.1);\n"
        "    }\n"
        "    .status {\n"
        "      display: flex;\n"
        "      flex-wrap: wrap;\n"
        "      gap: 12px 18px;\n"
        "      font-size: 0.95rem;\n"
        "      color: var(--muted);\n"
        "    }\n"
        "    .status strong { color: var(--ink); }\n"
        "    @media (max-width: 980px) {\n"
        "      .layout { grid-template-columns: 1fr; }\n"
        "      canvas { width: 100%; }\n"
        "      .mini-grid { grid-template-columns: 1fr; }\n"
        "    }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <main class=\"layout\">\n"
        "    <section class=\"panel\">\n"
        "      <p style=\"margin:0 0 10px; text-transform: uppercase; letter-spacing: 0.14em; font-size: 0.78rem; color: var(--muted);\">Synthetic Driving Simulator</p>\n"
        "      <h1>Animated Intersection Run</h1>\n"
        "      <p class=\"lede\">A browser playback of the current 128x128 world terrain plus the Phase 2 actor simulation. Cars stay in-lane, stop before the intersection, yield deterministically, and pedestrians wait for a signal before crossing.</p>\n"
        "      <div class=\"legend\">\n"
        "        <div class=\"legend-item\"><span class=\"swatch\" style=\"background: var(--obstacle)\"></span><span><strong>Obstacle</strong> Off-road terrain</span></div>\n"
        "        <div class=\"legend-item\"><span class=\"swatch\" style=\"background: var(--sidewalk)\"></span><span><strong>Sidewalk</strong> Pedestrian walkable edge</span></div>\n"
        "        <div class=\"legend-item\"><span class=\"swatch\" style=\"background: var(--drivable)\"></span><span><strong>Drivable</strong> Road surface</span></div>\n"
        "        <div class=\"legend-item\"><span class=\"swatch\" style=\"background: var(--lane)\"></span><span><strong>Lane</strong> Painted lane marking</span></div>\n"
        "        <div class=\"legend-item\"><span class=\"swatch\" style=\"background: var(--car)\"></span><span><strong>Car</strong> Rule-based vehicle actor</span></div>\n"
        "        <div class=\"legend-item\"><span class=\"swatch\" style=\"background: var(--ped)\"></span><span><strong>Pedestrian</strong> Signal-aware walker</span></div>\n"
        "      </div>\n"
        "      <div class=\"meta\">\n"
        "        <div><strong>Grid:</strong> 128 x 128 tiles at 1m per tile</div>\n"
        "        <div><strong>World extents:</strong> x,y in [-64m, +64m)</div>\n"
        "        <div><strong>Orientation:</strong> top = north (+Y), left = west (-X)</div>\n"
        "        <div><strong>Playback:</strong> use the scrubber, speed selector, and loop toggle to inspect trajectories</div>\n"
        "      </div>\n"
        "    </section>\n"
        "    <section class=\"viewer\">\n"
        "      <div class=\"viewport panel\">\n"
        "        <canvas id=\"grid\" width=\"768\" height=\"768\" aria-label=\"Animated world viewer\"></canvas>\n"
        "      </div>\n"
        "      <div class=\"panel controls\">\n"
        "        <div class=\"toolbar\">\n"
        "          <button id=\"playPause\" class=\"primary\" type=\"button\">Pause</button>\n"
        "          <button id=\"restart\" type=\"button\">Restart</button>\n"
        "          <label>Speed <select id=\"speed\">\n"
        "            <option value=\"0.5\">0.5x</option>\n"
        "            <option value=\"1\" selected>1x</option>\n"
        "            <option value=\"2\">2x</option>\n"
        "            <option value=\"4\">4x</option>\n"
        "          </select></label>\n"
        "          <label><input id=\"loop\" type=\"checkbox\" checked /> Loop</label>\n"
        "        </div>\n"
        "        <div class=\"timeline\">\n"
        "          <label for=\"scrubber\">Frame</label>\n"
        "          <input id=\"scrubber\" type=\"range\" min=\"0\" max=\"0\" value=\"0\" step=\"1\" />\n"
        "        </div>\n"
        "        <div class=\"mini-grid\">\n"
        "          <div class=\"stat\"><span class=\"stat-label\">Seed</span><span class=\"stat-value\" id=\"seedValue\"></span></div>\n"
        "          <div class=\"stat\"><span class=\"stat-label\">Timestep</span><span class=\"stat-value\" id=\"dtValue\"></span></div>\n"
        "          <div class=\"stat\"><span class=\"stat-label\">Tick</span><span class=\"stat-value\" id=\"tickValue\"></span></div>\n"
        "          <div class=\"stat\"><span class=\"stat-label\">Sim Time</span><span class=\"stat-value\" id=\"timeValue\"></span></div>\n"
        "          <div class=\"stat\"><span class=\"stat-label\">Cars</span><span class=\"stat-value\" id=\"carCount\"></span></div>\n"
        "          <div class=\"stat\"><span class=\"stat-label\">Pedestrians</span><span class=\"stat-value\" id=\"pedCount\"></span></div>\n"
        "        </div>\n"
        "      </div>\n"
        "      <div class=\"panel status\">\n"
        "        <div><strong>Trails</strong> Recent positions fade behind each actor so you can read the motion without frame-by-frame stepping</div>\n"
        "        <div><strong>Current source</strong> The viewer is generated directly from the same deterministic scene run used by the CLI CSV mode</div>\n"
        "      </div>\n"
        "    </section>\n"
        "  </main>\n"
        "  <script>\n"
        "    const TILE = { OUT_OF_BOUNDS: 0, DRIVABLE: 1, LANE: 2, SIDEWALK: 3, OBSTACLE: 4 };\n"
        "    const ACTOR = { CAR: 0, PEDESTRIAN: 1 };\n"
        "    const COLORS = {\n"
        "      [TILE.OUT_OF_BOUNDS]: '#000000',\n"
        "      [TILE.DRIVABLE]: getComputedStyle(document.documentElement).getPropertyValue('--drivable').trim(),\n"
        "      [TILE.LANE]: getComputedStyle(document.documentElement).getPropertyValue('--lane').trim(),\n"
        "      [TILE.SIDEWALK]: getComputedStyle(document.documentElement).getPropertyValue('--sidewalk').trim(),\n"
        "      [TILE.OBSTACLE]: getComputedStyle(document.documentElement).getPropertyValue('--obstacle').trim(),\n"
        "    };\n"
        "    const gridSize = 128;\n"
        "    const worldHalfExtent = 64;\n"
        "    const seed = ",
        file);

    std::fprintf(file, "%u", seed);

    std::fputs(";\n    const dt = ", file);
    std::fprintf(file, "%.6f", dt);
    std::fputs(";\n", file);

    writeFrameData(file, frames);
    writeTileData(file, grid);

    std::fputs(
        "    const canvas = document.getElementById('grid');\n"
        "    const ctx = canvas.getContext('2d');\n"
        "    const terrainCanvas = document.createElement('canvas');\n"
        "    terrainCanvas.width = canvas.width;\n"
        "    terrainCanvas.height = canvas.height;\n"
        "    const terrainCtx = terrainCanvas.getContext('2d');\n"
        "    const tilePx = canvas.width / gridSize;\n"
        "    const scrubber = document.getElementById('scrubber');\n"
        "    const playPause = document.getElementById('playPause');\n"
        "    const restart = document.getElementById('restart');\n"
        "    const speed = document.getElementById('speed');\n"
        "    const loop = document.getElementById('loop');\n"
        "    const seedValue = document.getElementById('seedValue');\n"
        "    const dtValue = document.getElementById('dtValue');\n"
        "    const tickValue = document.getElementById('tickValue');\n"
        "    const timeValue = document.getElementById('timeValue');\n"
        "    const carCount = document.getElementById('carCount');\n"
        "    const pedCount = document.getElementById('pedCount');\n"
        "    seedValue.textContent = String(seed);\n"
        "    dtValue.textContent = `${dt.toFixed(2)} s`;\n"
        "    scrubber.max = String(Math.max(frames.length - 1, 0));\n"
        "    const carColor = getComputedStyle(document.documentElement).getPropertyValue('--car').trim();\n"
        "    const carTopColor = getComputedStyle(document.documentElement).getPropertyValue('--car-top').trim();\n"
        "    const pedColor = getComputedStyle(document.documentElement).getPropertyValue('--ped').trim();\n"
        "    const trailCar = getComputedStyle(document.documentElement).getPropertyValue('--trail-car').trim();\n"
        "    const trailPed = getComputedStyle(document.documentElement).getPropertyValue('--trail-ped').trim();\n"
        "    const trailLength = 18;\n"
        "    let currentFrame = 0;\n"
        "    let playing = true;\n"
        "    let speedMultiplier = Number(speed.value);\n"
        "    let previousTimestamp = null;\n"
        "    let frameAccumulator = 0;\n"
        "    function drawTerrain() {\n"
        "      for (let row = 0; row < gridSize; row += 1) {\n"
        "        for (let col = 0; col < gridSize; col += 1) {\n"
        "          terrainCtx.fillStyle = COLORS[tiles[row][col]];\n"
        "          terrainCtx.fillRect(col * tilePx, row * tilePx, tilePx, tilePx);\n"
        "        }\n"
        "      }\n"
        "      terrainCtx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--grid').trim();\n"
        "      terrainCtx.lineWidth = 1;\n"
        "      for (let i = 0; i <= gridSize; i += 16) {\n"
        "        const p = i * tilePx;\n"
        "        terrainCtx.beginPath();\n"
        "        terrainCtx.moveTo(p, 0);\n"
        "        terrainCtx.lineTo(p, canvas.height);\n"
        "        terrainCtx.stroke();\n"
        "        terrainCtx.beginPath();\n"
        "        terrainCtx.moveTo(0, p);\n"
        "        terrainCtx.lineTo(canvas.width, p);\n"
        "        terrainCtx.stroke();\n"
        "      }\n"
        "      terrainCtx.strokeStyle = 'rgba(30, 24, 18, 0.35)';\n"
        "      terrainCtx.lineWidth = 2;\n"
        "      terrainCtx.strokeRect(0, 0, canvas.width, canvas.height);\n"
        "    }\n"
        "    function worldToCanvas(x, y) {\n"
        "      return {\n"
        "        x: ((x + worldHalfExtent) / gridSize) * canvas.width,\n"
        "        y: canvas.height - (((y + worldHalfExtent) / gridSize) * canvas.height),\n"
        "      };\n"
        "    }\n"
        "    function actorAt(frameIndex, actorId) {\n"
        "      return (frames[frameIndex] || []).find((actor) => actor[0] === actorId);\n"
        "    }\n"
        "    function drawTrail(frameIndex) {\n"
        "      const actors = frames[frameIndex] || [];\n"
        "      const start = Math.max(0, frameIndex - trailLength);\n"
        "      actors.forEach((currentActor) => {\n"
        "        let previousPoint = null;\n"
        "        for (let index = start; index <= frameIndex; index += 1) {\n"
        "          const actor = actorAt(index, currentActor[0]);\n"
        "          if (!actor) {\n"
        "            continue;\n"
        "          }\n"
        "          const point = worldToCanvas(actor[2], actor[3]);\n"
        "          if (previousPoint) {\n"
        "            const progress = (index - start + 1) / Math.max(frameIndex - start + 1, 1);\n"
        "            ctx.strokeStyle = actor[1] === ACTOR.CAR ? trailCar : trailPed;\n"
        "            ctx.globalAlpha = 0.2 + progress * 0.5;\n"
        "            ctx.lineWidth = actor[1] === ACTOR.CAR ? tilePx * 0.48 : tilePx * 0.22;\n"
        "            ctx.beginPath();\n"
        "            ctx.moveTo(previousPoint.x, previousPoint.y);\n"
        "            ctx.lineTo(point.x, point.y);\n"
        "            ctx.stroke();\n"
        "          }\n"
        "          previousPoint = point;\n"
        "        }\n"
        "      });\n"
        "      ctx.globalAlpha = 1;\n"
        "    }\n"
        "    function drawActor(actor) {\n"
        "      const point = worldToCanvas(actor[2], actor[3]);\n"
        "      const heading = -actor[4];\n"
        "      ctx.save();\n"
        "      ctx.translate(point.x, point.y);\n"
        "      ctx.rotate(heading);\n"
        "      if (actor[1] === ACTOR.CAR) {\n"
        "        const length = tilePx * 4.2;\n"
        "        const width = tilePx * 2.1;\n"
        "        ctx.fillStyle = carColor;\n"
        "        ctx.fillRect(-length / 2, -width / 2, length, width);\n"
        "        ctx.fillStyle = carTopColor;\n"
        "        ctx.fillRect(length * 0.1, -width * 0.3, length * 0.24, width * 0.6);\n"
        "      } else {\n"
        "        ctx.fillStyle = pedColor;\n"
        "        ctx.beginPath();\n"
        "        ctx.arc(0, 0, tilePx * 0.62, 0, Math.PI * 2);\n"
        "        ctx.fill();\n"
        "      }\n"
        "      ctx.restore();\n"
        "    }\n"
        "    function render(frameIndex) {\n"
        "      const actors = frames[frameIndex] || [];\n"
        "      ctx.clearRect(0, 0, canvas.width, canvas.height);\n"
        "      ctx.drawImage(terrainCanvas, 0, 0);\n"
        "      drawTrail(frameIndex);\n"
        "      actors.forEach(drawActor);\n"
        "      const cars = actors.filter((actor) => actor[1] === ACTOR.CAR).length;\n"
        "      const pedestrians = actors.length - cars;\n"
        "      tickValue.textContent = `${frameIndex + 1} / ${frames.length}`;\n"
        "      timeValue.textContent = `${(frameIndex * dt).toFixed(2)} s`;\n"
        "      carCount.textContent = String(cars);\n"
        "      pedCount.textContent = String(pedestrians);\n"
        "      scrubber.value = String(frameIndex);\n"
        "    }\n"
        "    function setPlaying(next) {\n"
        "      playing = next;\n"
        "      playPause.textContent = playing ? 'Pause' : 'Play';\n"
        "    }\n"
        "    function step(timestamp) {\n"
        "      if (previousTimestamp === null) {\n"
        "        previousTimestamp = timestamp;\n"
        "      }\n"
        "      const delta = timestamp - previousTimestamp;\n"
        "      previousTimestamp = timestamp;\n"
        "      if (playing && frames.length > 1) {\n"
        "        frameAccumulator += delta;\n"
        "        const frameDuration = (dt * 1000) / speedMultiplier;\n"
        "        while (frameAccumulator >= frameDuration) {\n"
        "          frameAccumulator -= frameDuration;\n"
        "          if (currentFrame + 1 < frames.length) {\n"
        "            currentFrame += 1;\n"
        "          } else if (loop.checked) {\n"
        "            currentFrame = 0;\n"
        "          } else {\n"
        "            setPlaying(false);\n"
        "            frameAccumulator = 0;\n"
        "            break;\n"
        "          }\n"
        "        }\n"
        "      }\n"
        "      render(currentFrame);\n"
        "      window.requestAnimationFrame(step);\n"
        "    }\n"
        "    playPause.addEventListener('click', () => setPlaying(!playing));\n"
        "    restart.addEventListener('click', () => {\n"
        "      currentFrame = 0;\n"
        "      frameAccumulator = 0;\n"
        "      render(currentFrame);\n"
        "      setPlaying(true);\n"
        "    });\n"
        "    scrubber.addEventListener('input', (event) => {\n"
        "      currentFrame = Number(event.target.value);\n"
        "      frameAccumulator = 0;\n"
        "      render(currentFrame);\n"
        "    });\n"
        "    speed.addEventListener('change', (event) => {\n"
        "      speedMultiplier = Number(event.target.value);\n"
        "      frameAccumulator = 0;\n"
        "    });\n"
        "    drawTerrain();\n"
        "    render(currentFrame);\n"
        "    window.requestAnimationFrame(step);\n"
        "  </script>\n"
        "</body>\n"
        "</html>\n",
        file);

    std::fclose(file);
    std::printf("Wrote animated HTML viewer to %s\n", output_path);
    return true;
}

bool writeBevPpm(const char* output_path, uint32_t seed, float dt, int num_ticks) {
    const sim::Scene scene = captureSceneAtLoggedTick(seed, dt, num_ticks);
    const sim::BevRasterizer rasterizer;
    const sim::BevImage image = rasterizer.rasterize(scene);

    std::FILE* file = std::fopen(output_path, "w");
    if (file == nullptr) {
        std::perror("failed to open output file");
        return false;
    }

    std::fprintf(file, "P3\n%d %d\n255\n", sim::GRID_SIZE, sim::GRID_SIZE);
    for (int row = 0; row < sim::GRID_SIZE; ++row) {
        for (int col = 0; col < sim::GRID_SIZE; ++col) {
            int red = 43;
            int green = 33;
            int blue = 23;

            switch (image.pixelAtIndex(col, row)) {
                case sim::SemanticClass::DRIVABLE:
                    red = 96;
                    green = 99;
                    blue = 88;
                    break;
                case sim::SemanticClass::LANE:
                    red = 245;
                    green = 208;
                    blue = 97;
                    break;
                case sim::SemanticClass::VEHICLE:
                    red = 190;
                    green = 74;
                    blue = 47;
                    break;
                case sim::SemanticClass::PEDESTRIAN:
                    red = 15;
                    green = 124;
                    blue = 115;
                    break;
                case sim::SemanticClass::OBSTACLE:
                    break;
            }

            std::fprintf(file, "%d %d %d%s", red, green, blue, col + 1 == sim::GRID_SIZE ? "" : " ");
        }
        std::fputc('\n', file);
    }

    std::fclose(file);
    std::printf(
        "Wrote clean semantic BEV snapshot to %s (classes: %s, %s, %s, %s, %s)\n",
        output_path,
        semanticLabel(sim::SemanticClass::DRIVABLE),
        semanticLabel(sim::SemanticClass::LANE),
        semanticLabel(sim::SemanticClass::VEHICLE),
        semanticLabel(sim::SemanticClass::PEDESTRIAN),
        semanticLabel(sim::SemanticClass::OBSTACLE));
    return true;
}

bool writeNoisyBevPgm(const char* output_path,
                      uint32_t seed,
                      float dt,
                      int num_ticks,
                      const sim::NoiseConfig& config,
                      const std::string& config_label) {
    const sim::Scene scene = captureSceneAtLoggedTick(seed, dt, num_ticks);
    const sim::BevRasterizer rasterizer;
    const sim::NoiseInjector injector;
    const sim::BevImage clean = rasterizer.rasterize(scene);
    const sim::NoisyBevImage noisy = injector.render(clean, seed, config);

    std::FILE* file = std::fopen(output_path, "w");
    if (file == nullptr) {
        std::perror("failed to open output file");
        return false;
    }

    std::fprintf(file, "P2\n%d %d\n255\n", sim::GRID_SIZE, sim::GRID_SIZE);
    for (int row = 0; row < sim::GRID_SIZE; ++row) {
        for (int col = 0; col < sim::GRID_SIZE; ++col) {
            std::fprintf(
                file,
                "%u%s",
                static_cast<unsigned int>(noisy.pixelAtIndex(col, row)),
                col + 1 == sim::GRID_SIZE ? "" : " ");
        }
        std::fputc('\n', file);
    }

    std::fclose(file);
    std::printf(
        "Wrote noisy BEV input snapshot to %s using %s\n",
        output_path,
        config_label.c_str());
    return true;
}

bool writeMetadataJson(const char* output_path, uint32_t seed, float dt, int num_ticks) {
    const sim::Scene scene = captureSceneAtLoggedTick(seed, dt, num_ticks);
    const int tick = num_ticks - 1;
    const float time_seconds = static_cast<float>(tick) * dt;
    std::ofstream file(output_path);
    if (!file.is_open()) {
        std::fprintf(stderr, "Error: cannot open output file: %s\n", output_path);
        return false;
    }

    sim::MetadataExporter exporter;
    exporter.writeJson(file, scene, seed, tick, time_seconds);
    std::printf("Wrote frame metadata JSON to %s\n", output_path);
    return true;
}

bool writeMetadataCsv(const char* output_path, uint32_t seed, float dt, int num_ticks) {
    const sim::Scene scene = captureSceneAtLoggedTick(seed, dt, num_ticks);
    const int tick = num_ticks - 1;
    const float time_seconds = static_cast<float>(tick) * dt;
    std::ofstream file(output_path);
    if (!file.is_open()) {
        std::fprintf(stderr, "Error: cannot open output file: %s\n", output_path);
        return false;
    }

    sim::MetadataExporter exporter;
    exporter.writeCsv(file, scene, seed, tick, time_seconds);
    std::printf("Wrote frame metadata CSV to %s\n", output_path);
    return true;
}

int runSimulation(uint32_t seed, float dt, int num_ticks, const std::string& out_path) {
    std::ofstream file_out;
    std::ostream* out_ptr = &std::cout;

    if (out_path != "-" && !out_path.empty()) {
        file_out.open(out_path);
        if (!file_out.is_open()) {
            std::fprintf(stderr, "Error: cannot open output file: %s\n", out_path.c_str());
            return 1;
        }
        out_ptr = &file_out;
    }

    sim::Scene scene(seed);
    sim::CsvLogger logger(*out_ptr);
    sim::SimLoop loop(scene, logger, dt, num_ticks);
    loop.run();
    return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
    uint32_t seed = 42;
    float dt = 0.05f;
    int num_ticks = 100;
    std::string out_path = "-";
    bool dump_grid = false;
    bool dump_grid_html = false;
    bool dump_bev_ppm = false;
    bool dump_noisy_bev_pgm = false;
    bool dump_metadata_json = false;
    bool dump_metadata_csv = false;
    std::string html_out_path = "grid_viewer.html";
    std::string bev_out_path = "clean_bev.ppm";
    std::string noisy_bev_out_path = "noisy_bev.pgm";
    std::string metadata_json_out_path = "frame_metadata.json";
    std::string metadata_csv_out_path = "frame_metadata.csv";
    sim::NoisePreset noise_preset = sim::NoisePreset::LOW;
    std::string noise_config_path;

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
        if (arg == "--dump-grid") {
            dump_grid = true;
            continue;
        }
        if (arg == "--dump-grid-html") {
            dump_grid_html = true;
            if (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) {
                html_out_path = argv[++i];
            }
            continue;
        }
        if (arg == "--dump-bev-ppm") {
            dump_bev_ppm = true;
            if (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) {
                bev_out_path = argv[++i];
            }
            continue;
        }
        if (arg == "--dump-noisy-bev-pgm") {
            dump_noisy_bev_pgm = true;
            if (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) {
                noisy_bev_out_path = argv[++i];
            }
            continue;
        }
        if (arg == "--dump-metadata-json") {
            dump_metadata_json = true;
            if (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) {
                metadata_json_out_path = argv[++i];
            }
            continue;
        }
        if (arg == "--dump-metadata-csv") {
            dump_metadata_csv = true;
            if (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) {
                metadata_csv_out_path = argv[++i];
            }
            continue;
        }
        if (arg == "--seed" && i + 1 < argc) {
            seed = static_cast<uint32_t>(std::stoul(argv[++i]));
            continue;
        }
        if (arg == "--dt" && i + 1 < argc) {
            dt = std::stof(argv[++i]);
            continue;
        }
        if (arg == "--ticks" && i + 1 < argc) {
            num_ticks = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "--noise" && i + 1 < argc) {
            if (!parseNoisePreset(argv[++i], noise_preset)) {
                std::fprintf(stderr, "Error: --noise must be 'low' or 'high'\n");
                return 1;
            }
            continue;
        }
        if (arg == "--noise-config" && i + 1 < argc) {
            noise_config_path = argv[++i];
            continue;
        }
        if (arg == "--out" && i + 1 < argc) {
            out_path = argv[++i];
            continue;
        }

        std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
        printUsage(argv[0]);
        return 1;
    }

    const int dump_mode_count =
        static_cast<int>(dump_grid) +
        static_cast<int>(dump_grid_html) +
        static_cast<int>(dump_bev_ppm) +
        static_cast<int>(dump_noisy_bev_pgm) +
        static_cast<int>(dump_metadata_json) +
        static_cast<int>(dump_metadata_csv);
    if (dump_mode_count > 1) {
        std::fprintf(stderr, "Error: choose only one dump mode at a time\n");
        return 1;
    }
    if (dt <= 0.0f) {
        std::fprintf(stderr, "Error: --dt must be > 0\n");
        return 1;
    }
    if (num_ticks <= 0) {
        std::fprintf(stderr, "Error: --ticks must be > 0\n");
        return 1;
    }
    if (dump_grid) {
        dumpGrid();
        return 0;
    }
    if (dump_grid_html) {
        return writeGridHtml(html_out_path.c_str(), seed, dt, num_ticks) ? 0 : 1;
    }
    if (dump_bev_ppm) {
        return writeBevPpm(bev_out_path.c_str(), seed, dt, num_ticks) ? 0 : 1;
    }
    if (dump_noisy_bev_pgm) {
        const std::string resolved_noise_config_path =
            noise_config_path.empty() ? defaultNoiseConfigPath(noise_preset) : noise_config_path;
        try {
            sim::NoiseConfigLoader loader;
            const sim::NoiseConfig config = loader.loadFromFile(resolved_noise_config_path);
            const std::string config_label = noise_config_path.empty()
                ? std::string("YAML preset '") + noisePresetLabel(noise_preset) + "'"
                : std::string("YAML config '") + resolved_noise_config_path + "'";
            return writeNoisyBevPgm(
                noisy_bev_out_path.c_str(),
                seed,
                dt,
                num_ticks,
                config,
                config_label) ? 0 : 1;
        } catch (const std::exception& error) {
            std::fprintf(stderr, "Error: %s\n", error.what());
            return 1;
        }
    }
    if (dump_metadata_json) {
        return writeMetadataJson(metadata_json_out_path.c_str(), seed, dt, num_ticks) ? 0 : 1;
    }
    if (dump_metadata_csv) {
        return writeMetadataCsv(metadata_csv_out_path.c_str(), seed, dt, num_ticks) ? 0 : 1;
    }

    return runSimulation(seed, dt, num_ticks, out_path);
}
