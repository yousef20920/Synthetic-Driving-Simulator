#!/usr/bin/env python3
"""Export a self-contained HTML viewer for U-Net predictions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import torch

from models import SEMANTIC_CLASS_NAMES
from perception import (
    extract_semantic_state,
    forecast_from_semantic_maps,
    plan_ego_route,
    rollout_ego_control,
)
from training import (
    SEMANTIC_CLASS_COLORS,
    SplitBevDataset,
    load_checkpoint,
    load_noisy_input_tensor,
    resolve_device,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIM_RUNNER = REPO_ROOT / "build" / "bin" / "sim_runner"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run checkpoint inference on a dataset split and write an HTML prediction viewer.",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Dataset root directory")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint to visualize")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test", help="Dataset split")
    parser.add_argument("--num-samples", type=int, default=6, help="Number of samples to render")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, mps")
    parser.add_argument("--sim-runner", type=Path, default=DEFAULT_SIM_RUNNER, help="sim_runner path for forecasting")
    parser.add_argument("--forecast-horizon", type=int, default=5, help="Number of future steps to forecast")
    parser.add_argument("--control-horizon", type=int, default=5, help="Number of ego-control rollout steps")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("prediction_viewer.html"),
        help="HTML file to write",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset_dir.is_dir():
        raise SystemExit(f"dataset directory not found: {args.dataset_dir}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be greater than zero")
    if args.forecast_horizon <= 0:
        raise SystemExit("--forecast-horizon must be greater than zero")
    if args.control_horizon <= 0:
        raise SystemExit("--control-horizon must be greater than zero")


def flatten_2d(values: torch.Tensor) -> list[int]:
    return [int(value) for value in values.reshape(-1).tolist()]


def parse_metadata(metadata_path: Path) -> dict[str, object]:
    if metadata_path.suffix != ".json":
        return {"path": str(metadata_path)}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"path": str(metadata_path)}
    payload["path"] = str(metadata_path)
    return payload


def dataset_config(dataset_dir: Path) -> dict[str, object]:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return manifest.get("config", {})


def run_command(command: list[str], label: str) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        return

    detail = [f"{label} failed with exit code {result.returncode}"]
    if result.stdout.strip():
        detail.extend(["stdout:", result.stdout.strip()])
    if result.stderr.strip():
        detail.extend(["stderr:", result.stderr.strip()])
    raise RuntimeError("\n".join(detail))


def predict_semantic_map(model: torch.nn.Module, device: torch.device, input_tensor: torch.Tensor) -> torch.Tensor:
    logits = model(input_tensor.unsqueeze(0).to(device))
    return torch.argmax(logits, dim=1).squeeze(0).cpu()


def compact_blob(blob: dict[str, object]) -> dict[str, object]:
    return {
        "class_name": blob["class_name"],
        "area": blob["area"],
        "centroid_row": blob["centroid_row"],
        "centroid_col": blob["centroid_col"],
        "bbox": blob["bbox"],
    }


def compact_state(state: object) -> dict[str, object]:
    payload = state.to_dict()
    return {
        "height": payload["height"],
        "width": payload["width"],
        "free_space": {
            "area": payload["free_space"]["area"],
            "bbox": payload["free_space"]["bbox"],
        },
        "lane_regions": [compact_blob(blob) for blob in payload["lane_regions"]],
        "lane_tracks": [
            {
                "orientation": track["orientation"],
                "region": compact_blob(track["region"]),
                "centerline": track["centerline"],
            }
            for track in payload["lane_tracks"]
        ],
        "vehicle_blobs": [compact_blob(blob) for blob in payload["vehicle_blobs"]],
        "pedestrian_blobs": [compact_blob(blob) for blob in payload["pedestrian_blobs"]],
        "obstacle_regions": [compact_blob(blob) for blob in payload["obstacle_regions"]],
    }


def compact_plan(plan: object) -> dict[str, object]:
    return plan.to_dict()


def compact_control(control: object) -> dict[str, object]:
    return control.to_dict()


def try_forecast(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
    current_prediction: torch.Tensor,
    metadata: dict[str, object],
    dataset_cfg: dict[str, object],
) -> object | None:
    if not args.sim_runner.is_file():
        return None
    if "seed" not in metadata or "tick" not in metadata:
        return None
    if not isinstance(metadata["tick"], int) or metadata["tick"] <= 0:
        return None
    if "dt" not in dataset_cfg or "noise" not in dataset_cfg:
        return None

    seed = int(metadata["seed"])
    previous_tick = int(metadata["tick"]) - 1
    dt = float(dataset_cfg["dt"])
    noise = str(dataset_cfg["noise"])
    if previous_tick < 0:
        return None

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        previous_input_path = temp_path / "previous_input.pgm"
        command = [
            str(args.sim_runner),
            "--seed",
            str(seed),
            "--dt",
            str(dt),
            "--ticks",
            str(previous_tick + 1),
            "--noise",
            noise,
            "--dump-noisy-bev-pgm",
            str(previous_input_path),
        ]
        run_command(command, "previous-frame noisy BEV export")
        previous_input = load_noisy_input_tensor(previous_input_path)
        previous_prediction = predict_semantic_map(model, device, previous_input)
        forecast = forecast_from_semantic_maps(
            previous_prediction,
            current_prediction,
            dt=dt,
            horizon_steps=args.forecast_horizon,
        )
        return forecast


def export_samples(args: argparse.Namespace) -> dict[str, object]:
    validate_args(args)
    dataset = SplitBevDataset(args.dataset_dir, args.split)
    if len(dataset) == 0:
        raise SystemExit(f"{args.split} split is empty")
    manifest_config = dataset_config(args.dataset_dir)

    device = resolve_device(args.device)
    model, checkpoint_payload = load_checkpoint(args.checkpoint, map_location=device)
    model = model.to(device)
    model.eval()

    colors = {name: list(rgb) for name, rgb in SEMANTIC_CLASS_COLORS.items()}
    selected_count = min(args.num_samples, len(dataset))
    samples = []

    with torch.no_grad():
        for index in range(selected_count):
            sample = dataset[index]
            input_tensor = sample["input"].unsqueeze(0).to(device)
            logits = model(input_tensor)
            prediction = torch.argmax(logits, dim=1).squeeze(0).cpu()
            target = sample["target"].cpu()
            input_image = sample["input"].squeeze(0).cpu()
            extracted_state_full = extract_semantic_state(prediction)
            extracted_state = compact_state(extracted_state_full)

            pixel_accuracy = float((prediction == target).float().mean().item())
            predicted_classes = sorted({SEMANTIC_CLASS_NAMES[class_id] for class_id in prediction.unique().tolist()})

            metadata_path = Path(str(sample["metadata_path"]))
            metadata_payload = parse_metadata(metadata_path)
            control_dt = float(manifest_config.get("dt", 0.05))
            forecast = try_forecast(
                args=args,
                model=model,
                device=device,
                current_prediction=prediction,
                metadata=metadata_payload,
                dataset_cfg=manifest_config,
            )
            ego_plan_full = plan_ego_route(
                prediction,
                extracted_state=extracted_state_full,
                forecast=forecast,
            )
            ego_plan = compact_plan(ego_plan_full)
            ego_control = compact_control(
                rollout_ego_control(
                    ego_plan_full,
                    forecast=forecast,
                    dt=control_dt,
                    horizon_steps=args.control_horizon,
                )
            )
            samples.append(
                {
                    "sample_id": sample["sample_id"],
                    "split": sample["split"],
                    "width": int(target.shape[1]),
                    "height": int(target.shape[0]),
                    "input": flatten_2d(torch.round(input_image * 255.0).to(torch.uint8)),
                    "label": flatten_2d(target),
                    "prediction": flatten_2d(prediction),
                    "pixel_accuracy": pixel_accuracy,
                    "predicted_classes": predicted_classes,
                    "metadata": metadata_payload,
                    "extracted_state": extracted_state,
                    "forecast": None if forecast is None else forecast.to_dict(),
                    "ego_plan": ego_plan,
                    "ego_control": ego_control,
                }
            )

    return {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "device": str(device),
        "checkpoint_epoch": checkpoint_payload["epoch"],
        "forecast_horizon": args.forecast_horizon,
        "control_horizon": args.control_horizon,
        "class_names": list(SEMANTIC_CLASS_NAMES),
        "class_colors": colors,
        "samples": samples,
    }


def render_html(payload: dict[str, object]) -> str:
    data_json = json.dumps(payload)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Prediction Viewer</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --panel: rgba(255, 251, 245, 0.92);
      --ink: #1f1a17;
      --muted: #6a5f57;
      --accent: #173d57;
      --grid: rgba(31, 26, 23, 0.08);
      --line: rgba(31, 26, 23, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: Georgia, 'Times New Roman', serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(255,255,255,0.72), transparent 28%),
        linear-gradient(180deg, #fbf7ef 0%%, var(--bg) 100%%);
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(280px, 360px) 1fr;
      gap: 24px;
      padding: 24px;
      align-items: start;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 22px;
      box-shadow: 0 18px 48px rgba(73, 55, 35, 0.1);
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: clamp(2rem, 4vw, 3rem);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }}
    .lede {{
      margin: 0 0 18px;
      line-height: 1.5;
      color: var(--muted);
    }}
    .meta {{
      display: grid;
      gap: 6px;
      font-size: 0.95rem;
      color: var(--muted);
      padding-bottom: 18px;
      border-bottom: 1px solid var(--line);
      margin-bottom: 18px;
    }}
    .control-block {{
      display: grid;
      gap: 10px;
      margin-bottom: 18px;
    }}
    .toolbar {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      margin-bottom: 18px;
    }}
    .control-block label {{
      font-size: 0.9rem;
      color: var(--muted);
    }}
    select, input[type="range"] {{
      width: 100%;
    }}
    select, button {{
      font: inherit;
      padding: 10px 12px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: white;
    }}
    button {{
      cursor: pointer;
    }}
    .stat-list {{
      display: grid;
      gap: 8px;
      margin: 0;
      padding: 0;
      list-style: none;
    }}
    .legend {{
      display: grid;
      gap: 8px;
      margin-top: 16px;
    }}
    .legend-item {{
      display: grid;
      grid-template-columns: 16px 1fr;
      gap: 10px;
      align-items: center;
      font-size: 0.95rem;
    }}
    .swatch {{
      width: 16px;
      height: 16px;
      border-radius: 5px;
      border: 1px solid rgba(0,0,0,0.14);
    }}
    .viewer {{
      display: grid;
      gap: 18px;
    }}
    .canvas-grid {{
      display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 18px;
    }}
    .tile {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 18px;
      box-shadow: 0 14px 38px rgba(73, 55, 35, 0.08);
    }}
    .tile h2 {{
      margin: 0 0 12px;
      font-size: 1.1rem;
      letter-spacing: -0.02em;
    }}
    canvas {{
      width: 100%;
      aspect-ratio: 1 / 1;
      image-rendering: pixelated;
      border-radius: 18px;
      background: white;
      border: 1px solid var(--line);
    }}
    .caption {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 0.92rem;
      line-height: 1.4;
    }}
    .metadata {{
      white-space: pre-wrap;
      background: rgba(255,255,255,0.7);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      font-size: 0.88rem;
      line-height: 1.45;
      color: var(--muted);
    }}
    @media (max-width: 980px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .canvas-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="panel">
      <h1>Model Viewer</h1>
      <p class="lede">Side-by-side view of the noisy BEV input, ground-truth semantic label, and the U-Net prediction from a saved checkpoint.</p>
      <div class="meta">
        <div><strong>Dataset:</strong> <span id="datasetDir"></span></div>
        <div><strong>Checkpoint:</strong> <span id="checkpointPath"></span></div>
        <div><strong>Split:</strong> <span id="splitName"></span></div>
        <div><strong>Device:</strong> <span id="deviceName"></span></div>
        <div><strong>Checkpoint Epoch:</strong> <span id="checkpointEpoch"></span></div>
        <div><strong>Forecast Horizon:</strong> <span id="forecastHorizon"></span></div>
        <div><strong>Control Horizon:</strong> <span id="controlHorizon"></span></div>
      </div>
      <div class="control-block">
        <label for="sampleSelect">Sample</label>
        <select id="sampleSelect"></select>
      </div>
      <div class="control-block">
        <label for="sampleScrubber">Browse Samples</label>
        <input id="sampleScrubber" type="range" min="0" max="0" value="0" />
      </div>
      <div class="toolbar">
        <button id="forecastToggle" type="button">Play Rollout</button>
        <button id="forecastRestart" type="button">Restart</button>
      </div>
      <div class="control-block">
        <label for="forecastScrubber">Scenario Step</label>
        <input id="forecastScrubber" type="range" min="0" max="0" value="0" />
      </div>
      <ul class="stat-list" id="sampleStats"></ul>
      <div class="legend" id="legend"></div>
    </aside>
    <main class="viewer">
      <div class="canvas-grid">
        <section class="tile">
          <h2>Noisy Input</h2>
          <canvas id="inputCanvas" width="256" height="256"></canvas>
          <div class="caption">Single-channel noisy BEV raster fed into the U-Net.</div>
        </section>
        <section class="tile">
          <h2>Ground Truth</h2>
          <canvas id="labelCanvas" width="256" height="256"></canvas>
          <div class="caption">Semantic target exported by the simulator.</div>
        </section>
        <section class="tile">
          <h2>Prediction</h2>
          <canvas id="predictionCanvas" width="256" height="256"></canvas>
          <div class="caption">Argmax class map predicted by the checkpoint.</div>
        </section>
        <section class="tile">
          <h2>Extracted State</h2>
          <canvas id="stateCanvas" width="256" height="256"></canvas>
          <div class="caption">Planner-facing overlay with lane centerlines, detected blobs, forecasted future points, and the ego plan drawn over the prediction.</div>
        </section>
      </div>
      <div class="canvas-grid">
        <section class="tile">
          <h2>Metadata</h2>
          <div id="metadataView" class="metadata"></div>
        </section>
        <section class="tile">
          <h2>Extracted State Summary</h2>
          <div id="stateSummaryView" class="metadata"></div>
        </section>
        <section class="tile">
          <h2>Ego Plan Summary</h2>
          <div id="planSummaryView" class="metadata"></div>
        </section>
        <section class="tile">
          <h2>Ego Control Summary</h2>
          <div id="controlSummaryView" class="metadata"></div>
        </section>
      </div>
    </main>
  </div>
  <script>
    const payload = {data_json};
    const classNames = payload.class_names;
    const classColors = payload.class_colors;
    const samples = payload.samples;

    const datasetDir = document.getElementById('datasetDir');
    const checkpointPath = document.getElementById('checkpointPath');
    const splitName = document.getElementById('splitName');
    const deviceName = document.getElementById('deviceName');
    const checkpointEpoch = document.getElementById('checkpointEpoch');
    const forecastHorizon = document.getElementById('forecastHorizon');
    const controlHorizon = document.getElementById('controlHorizon');
    const sampleSelect = document.getElementById('sampleSelect');
    const sampleScrubber = document.getElementById('sampleScrubber');
    const forecastToggle = document.getElementById('forecastToggle');
    const forecastRestart = document.getElementById('forecastRestart');
    const forecastScrubber = document.getElementById('forecastScrubber');
    const sampleStats = document.getElementById('sampleStats');
    const metadataView = document.getElementById('metadataView');
    const legend = document.getElementById('legend');

    const inputCanvas = document.getElementById('inputCanvas');
    const labelCanvas = document.getElementById('labelCanvas');
    const predictionCanvas = document.getElementById('predictionCanvas');
    const stateCanvas = document.getElementById('stateCanvas');
    const stateSummaryView = document.getElementById('stateSummaryView');
    const planSummaryView = document.getElementById('planSummaryView');
    const controlSummaryView = document.getElementById('controlSummaryView');
    let currentScenarioStep = 0;
    let forecastPlaying = false;
    let forecastTimer = null;

    datasetDir.textContent = payload.dataset_dir;
    checkpointPath.textContent = payload.checkpoint;
    splitName.textContent = payload.split;
    deviceName.textContent = payload.device;
    checkpointEpoch.textContent = String(payload.checkpoint_epoch);
    forecastHorizon.textContent = String(payload.forecast_horizon) + ' steps';
    controlHorizon.textContent = String(payload.control_horizon) + ' steps';

    function renderLegend() {{
      classNames.forEach((className) => {{
        const item = document.createElement('div');
        item.className = 'legend-item';
        const swatch = document.createElement('span');
        swatch.className = 'swatch';
        const [r, g, b] = classColors[className];
        swatch.style.background = `rgb(${{r}}, ${{g}}, ${{b}})`;
        const label = document.createElement('span');
        label.textContent = className;
        item.appendChild(swatch);
        item.appendChild(label);
        legend.appendChild(item);
      }});
    }}

    function fillSelect() {{
      samples.forEach((sample, index) => {{
        const option = document.createElement('option');
        option.value = String(index);
        option.textContent = `${{sample.sample_id}}`;
        sampleSelect.appendChild(option);
      }});
      sampleScrubber.max = String(Math.max(samples.length - 1, 0));
    }}

    function forecastStepCount(sample) {{
      if (!sample.forecast) {{
        return 0;
      }}
      const vehicleSteps = (sample.forecast.vehicles || []).reduce(
        (count, actor) => Math.max(count, actor.trajectory ? actor.trajectory.length : 0),
        0,
      );
      const pedestrianSteps = (sample.forecast.pedestrians || []).reduce(
        (count, actor) => Math.max(count, actor.trajectory ? actor.trajectory.length : 0),
        0,
      );
      return Math.max(vehicleSteps, pedestrianSteps);
    }}

    function controlStepCount(sample) {{
      if (!sample.ego_control || !sample.ego_control.states) {{
        return 0;
      }}
      return Math.max(sample.ego_control.states.length - 1, 0);
    }}

    function scenarioStepCount(sample) {{
      return Math.max(forecastStepCount(sample), controlStepCount(sample));
    }}

    function stopForecastPlayback() {{
      forecastPlaying = false;
      forecastToggle.textContent = 'Play Rollout';
      if (forecastTimer !== null) {{
        window.clearInterval(forecastTimer);
        forecastTimer = null;
      }}
    }}

    function syncForecastControls(sample) {{
      const maxStep = scenarioStepCount(sample);
      currentScenarioStep = Math.min(currentScenarioStep, maxStep);
      forecastScrubber.max = String(maxStep);
      forecastScrubber.value = String(currentScenarioStep);
      forecastToggle.disabled = maxStep === 0;
      forecastRestart.disabled = maxStep === 0;
    }}

    function drawInput(canvas, sample) {{
      const ctx = canvas.getContext('2d');
      const imageData = ctx.createImageData(sample.width, sample.height);
      for (let i = 0; i < sample.input.length; i += 1) {{
        const value = sample.input[i];
        const base = i * 4;
        imageData.data[base + 0] = value;
        imageData.data[base + 1] = value;
        imageData.data[base + 2] = value;
        imageData.data[base + 3] = 255;
      }}
      const offscreen = document.createElement('canvas');
      offscreen.width = sample.width;
      offscreen.height = sample.height;
      offscreen.getContext('2d').putImageData(imageData, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);
    }}

    function drawSemantic(canvas, sample, key) {{
      const ctx = canvas.getContext('2d');
      const imageData = ctx.createImageData(sample.width, sample.height);
      const values = sample[key];
      for (let i = 0; i < values.length; i += 1) {{
        const className = classNames[values[i]];
        const [r, g, b] = classColors[className];
        const base = i * 4;
        imageData.data[base + 0] = r;
        imageData.data[base + 1] = g;
        imageData.data[base + 2] = b;
        imageData.data[base + 3] = 255;
      }}
      const offscreen = document.createElement('canvas');
      offscreen.width = sample.width;
      offscreen.height = sample.height;
      offscreen.getContext('2d').putImageData(imageData, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);
    }}

    function drawBlobBoxes(ctx, blobs, sample, strokeStyle) {{
      const scaleX = stateCanvas.width / sample.width;
      const scaleY = stateCanvas.height / sample.height;
      ctx.strokeStyle = strokeStyle;
      ctx.lineWidth = 2;
      blobs.forEach((blob) => {{
        const bbox = blob.bbox;
        ctx.strokeRect(
          bbox.min_col * scaleX,
          bbox.min_row * scaleY,
          (bbox.max_col - bbox.min_col + 1) * scaleX,
          (bbox.max_row - bbox.min_row + 1) * scaleY,
        );
      }});
    }}

    function drawLaneTracks(ctx, laneTracks, sample) {{
      const scaleX = stateCanvas.width / sample.width;
      const scaleY = stateCanvas.height / sample.height;
      ctx.strokeStyle = '#ffffff';
      ctx.fillStyle = '#173d57';
      ctx.lineWidth = 3;
      laneTracks.forEach((track) => {{
        const points = track.centerline || [];
        if (!points.length) {{
          return;
        }}
        ctx.beginPath();
        points.forEach((point, index) => {{
          const x = (point.col + 0.5) * scaleX;
          const y = (point.row + 0.5) * scaleY;
          if (index === 0) {{
            ctx.moveTo(x, y);
          }} else {{
            ctx.lineTo(x, y);
          }}
        }});
        ctx.stroke();
        points.forEach((point) => {{
          const x = (point.col + 0.5) * scaleX;
          const y = (point.row + 0.5) * scaleY;
          ctx.beginPath();
          ctx.arc(x, y, 2.6, 0, Math.PI * 2);
          ctx.fill();
        }});
      }});
    }}

    function drawForecast(ctx, forecast, sample, activeStep) {{
      if (!forecast) {{
        return;
      }}
      const scaleX = stateCanvas.width / sample.width;
      const scaleY = stateCanvas.height / sample.height;

      function drawActorPath(actor, color) {{
        if (!actor.trajectory || !actor.trajectory.length) {{
          return;
        }}
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        actor.trajectory.forEach((point, index) => {{
          const x = (point.col + 0.5) * scaleX;
          const y = (point.row + 0.5) * scaleY;
          if (index === 0) {{
            ctx.moveTo(x, y);
          }} else {{
            ctx.lineTo(x, y);
          }}
        }});
        ctx.stroke();
        actor.trajectory.forEach((point) => {{
          const x = (point.col + 0.5) * scaleX;
          const y = (point.row + 0.5) * scaleY;
          ctx.beginPath();
          ctx.globalAlpha = 0.35;
          ctx.arc(x, y, 3, 0, Math.PI * 2);
          ctx.fill();
          ctx.globalAlpha = 1.0;
        }});
        if (activeStep > 0) {{
          const activePoint = actor.trajectory[Math.min(activeStep, actor.trajectory.length) - 1];
          if (activePoint) {{
            const x = (activePoint.col + 0.5) * scaleX;
            const y = (activePoint.row + 0.5) * scaleY;
            ctx.beginPath();
            ctx.fillStyle = color;
            ctx.arc(x, y, 6, 0, Math.PI * 2);
            ctx.fill();
            ctx.beginPath();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.arc(x, y, 8.5, 0, Math.PI * 2);
            ctx.stroke();
          }}
        }}
      }}

      (forecast.vehicles || []).forEach((actor) => drawActorPath(actor, 'rgba(190, 74, 47, 0.92)'));
      (forecast.pedestrians || []).forEach((actor) => drawActorPath(actor, 'rgba(15, 124, 115, 0.92)'));
    }}

    function drawEgoPlan(ctx, plan, sample) {{
      if (!plan || !plan.path || !plan.path.length) {{
        return;
      }}
      const scaleX = stateCanvas.width / sample.width;
      const scaleY = stateCanvas.height / sample.height;
      ctx.strokeStyle = 'rgba(23, 61, 87, 0.96)';
      ctx.lineWidth = 4;
      ctx.beginPath();
      plan.path.forEach((point, index) => {{
        const x = (point.col + 0.5) * scaleX;
        const y = (point.row + 0.5) * scaleY;
        if (index === 0) {{
          ctx.moveTo(x, y);
        }} else {{
          ctx.lineTo(x, y);
        }}
      }});
      ctx.stroke();

      const startX = (plan.start.col + 0.5) * scaleX;
      const startY = (plan.start.row + 0.5) * scaleY;
      const goalX = (plan.goal.col + 0.5) * scaleX;
      const goalY = (plan.goal.row + 0.5) * scaleY;

      ctx.fillStyle = '#0f7c73';
      ctx.beginPath();
      ctx.arc(startX, startY, 6, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = '#173d57';
      ctx.beginPath();
      ctx.arc(goalX, goalY, 6, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(startX, startY, 8.5, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(goalX, goalY, 8.5, 0, Math.PI * 2);
      ctx.stroke();
    }}

    function drawEgoControl(ctx, control, sample, activeStep) {{
      if (!control || !control.states || !control.states.length) {{
        return;
      }}
      const scaleX = stateCanvas.width / sample.width;
      const scaleY = stateCanvas.height / sample.height;

      ctx.strokeStyle = 'rgba(233, 179, 46, 0.95)';
      ctx.lineWidth = 3;
      ctx.setLineDash([8, 5]);
      ctx.beginPath();
      control.states.forEach((state, index) => {{
        const x = (state.col + 0.5) * scaleX;
        const y = (state.row + 0.5) * scaleY;
        if (index === 0) {{
          ctx.moveTo(x, y);
        }} else {{
          ctx.lineTo(x, y);
        }}
      }});
      ctx.stroke();
      ctx.setLineDash([]);

      const activeIndex = Math.min(activeStep, control.states.length - 1);
      const activeState = control.states[activeIndex];
      if (!activeState) {{
        return;
      }}

      const x = (activeState.col + 0.5) * scaleX;
      const y = (activeState.row + 0.5) * scaleY;
      ctx.fillStyle = '#e9b32e';
      ctx.beginPath();
      ctx.arc(x, y, 6.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, 9.5, 0, Math.PI * 2);
      ctx.stroke();
    }}

    function drawStateOverlay(sample) {{
      drawSemantic(stateCanvas, sample, 'prediction');
      const ctx = stateCanvas.getContext('2d');
      drawBlobBoxes(ctx, sample.extracted_state.obstacle_regions, sample, '#2b2117');
      drawBlobBoxes(ctx, sample.extracted_state.vehicle_blobs, sample, '#be4a2f');
      drawBlobBoxes(ctx, sample.extracted_state.pedestrian_blobs, sample, '#0f7c73');
      drawLaneTracks(ctx, sample.extracted_state.lane_tracks, sample);
      drawForecast(ctx, sample.forecast, sample, currentScenarioStep);
      drawEgoPlan(ctx, sample.ego_plan, sample);
      drawEgoControl(ctx, sample.ego_control, sample, currentScenarioStep);
    }}

    function renderStats(sample) {{
      sampleStats.innerHTML = '';
      const entries = [
        `Sample: ${{sample.sample_id}}`,
        `Pixel Accuracy: ${{sample.pixel_accuracy.toFixed(4)}}`,
        `Predicted Classes: ${{sample.predicted_classes.join(', ') || 'none'}}`,
        `Resolution: ${{sample.width}}x${{sample.height}}`,
        `Lane Tracks: ${{sample.extracted_state.lane_tracks.length}}`,
        `Vehicles: ${{sample.extracted_state.vehicle_blobs.length}}`,
        `Pedestrians: ${{sample.extracted_state.pedestrian_blobs.length}}`,
        `Obstacle Regions: ${{sample.extracted_state.obstacle_regions.length}}`,
        `Forecast Vehicles: ${{sample.forecast ? sample.forecast.vehicles.length : 0}}`,
        `Forecast Pedestrians: ${{sample.forecast ? sample.forecast.pedestrians.length : 0}}`,
        `Scenario Step: ${{currentScenarioStep}} / ${{scenarioStepCount(sample)}}`,
        `Plan Strategy: ${{sample.ego_plan.strategy}}`,
        `Plan Path Cells: ${{sample.ego_plan.path.length}}`,
        `Plan Fallback: ${{sample.ego_plan.used_fallback ? 'yes' : 'no'}}`,
        `Control Commands: ${{sample.ego_control.commands.length}}`,
        `Goal Reached: ${{sample.ego_control.goal_reached ? 'yes' : 'no'}}`,
      ];
      entries.forEach((text) => {{
        const item = document.createElement('li');
        item.textContent = text;
        sampleStats.appendChild(item);
      }});
    }}

    function renderMetadata(sample) {{
      metadataView.textContent = JSON.stringify(sample.metadata, null, 2);
    }}

    function renderStateSummary(sample) {{
      stateSummaryView.textContent = JSON.stringify(
        {{
          extracted_state: sample.extracted_state,
          forecast: sample.forecast,
        }},
        null,
        2,
      );
    }}

    function renderPlanSummary(sample) {{
      planSummaryView.textContent = JSON.stringify(sample.ego_plan, null, 2);
    }}

    function renderControlSummary(sample) {{
      controlSummaryView.textContent = JSON.stringify(sample.ego_control, null, 2);
    }}

    function renderSample(index) {{
      const sample = samples[index];
      sampleSelect.value = String(index);
      sampleScrubber.value = String(index);
      syncForecastControls(sample);
      drawInput(inputCanvas, sample);
      drawSemantic(labelCanvas, sample, 'label');
      drawSemantic(predictionCanvas, sample, 'prediction');
      drawStateOverlay(sample);
      renderStats(sample);
      renderMetadata(sample);
      renderStateSummary(sample);
      renderPlanSummary(sample);
      renderControlSummary(sample);
    }}

    sampleSelect.addEventListener('change', (event) => {{
      stopForecastPlayback();
      renderSample(Number(event.target.value));
    }});

    sampleScrubber.addEventListener('input', (event) => {{
      stopForecastPlayback();
      renderSample(Number(event.target.value));
    }});

    forecastScrubber.addEventListener('input', (event) => {{
      currentScenarioStep = Number(event.target.value);
      renderSample(Number(sampleSelect.value));
    }});

    forecastToggle.addEventListener('click', () => {{
      const sample = samples[Number(sampleSelect.value)];
      const maxStep = scenarioStepCount(sample);
      if (maxStep === 0) {{
        return;
      }}
      if (forecastPlaying) {{
        stopForecastPlayback();
        return;
      }}
      forecastPlaying = true;
      forecastToggle.textContent = 'Pause Rollout';
      forecastTimer = window.setInterval(() => {{
        if (currentScenarioStep >= maxStep) {{
          currentScenarioStep = 0;
        }} else {{
          currentScenarioStep += 1;
        }}
        forecastScrubber.value = String(currentScenarioStep);
        renderSample(Number(sampleSelect.value));
      }}, 450);
    }});

    forecastRestart.addEventListener('click', () => {{
      stopForecastPlayback();
      currentScenarioStep = 0;
      forecastScrubber.value = '0';
      renderSample(Number(sampleSelect.value));
    }});

    renderLegend();
    fillSelect();
    renderSample(0);
  </script>
</body>
</html>
"""


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    payload = export_samples(args)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(render_html(payload), encoding="utf-8")
    print(f"wrote viewer to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
