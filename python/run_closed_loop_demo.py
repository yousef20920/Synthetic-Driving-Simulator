#!/usr/bin/env python3
"""Run a closed-loop ego-driving demo on top of simulator-generated BEV observations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import torch

from models import SEMANTIC_CLASS_NAMES
from perception import ClosedLoopFrame, run_closed_loop_episode, world_to_plan_point
from training import SEMANTIC_CLASS_COLORS, load_checkpoint, load_noisy_input_tensor, resolve_device


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIM_RUNNER = REPO_ROOT / "build" / "bin" / "sim_runner"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a closed-loop ego-driving demo using the trained perception stack.",
    )
    parser.add_argument("--sim-runner", type=Path, default=DEFAULT_SIM_RUNNER, help="Path to sim_runner")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained model checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Simulation seed")
    parser.add_argument("--ticks", type=int, default=40, help="Number of ticks to run")
    parser.add_argument("--dt", type=float, default=0.05, help="Simulation timestep in seconds")
    parser.add_argument("--noise", choices=("low", "high"), default="low", help="Noisy BEV preset")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, mps")
    parser.add_argument("--control-horizon", type=int, default=5, help="Closed-loop control horizon")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/closed_loop_demo"),
        help="Output directory for the demo artifacts",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not args.sim_runner.is_file():
        raise SystemExit(f"sim_runner not found: {args.sim_runner}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    if args.ticks <= 0:
        raise SystemExit("--ticks must be greater than zero")
    if args.dt <= 0.0:
        raise SystemExit("--dt must be greater than zero")
    if args.control_horizon <= 0:
        raise SystemExit("--control-horizon must be greater than zero")


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


def parse_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_2d(values: torch.Tensor) -> list[int]:
    return [int(value) for value in values.reshape(-1).tolist()]


def predict_semantic_map(model: torch.nn.Module, device: torch.device, input_tensor: torch.Tensor) -> torch.Tensor:
    logits = model(input_tensor.unsqueeze(0).to(device))
    return torch.argmax(logits, dim=1).squeeze(0).cpu()


def collect_frames(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[list[ClosedLoopFrame], list[dict[str, Any]]]:
    frames: list[ClosedLoopFrame] = []
    visuals: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        with torch.no_grad():
            for tick in range(args.ticks):
                input_path = temp_path / f"tick_{tick:04d}_input.pgm"
                metadata_path = temp_path / f"tick_{tick:04d}_metadata.json"
                common = [
                    str(args.sim_runner),
                    "--seed",
                    str(args.seed),
                    "--dt",
                    str(args.dt),
                    "--ticks",
                    str(tick + 1),
                ]
                run_command(
                    [*common, "--noise", args.noise, "--dump-noisy-bev-pgm", str(input_path)],
                    f"tick {tick} noisy BEV export",
                )
                run_command(
                    [*common, "--dump-metadata-json", str(metadata_path)],
                    f"tick {tick} metadata export",
                )

                input_tensor = load_noisy_input_tensor(input_path)
                prediction = predict_semantic_map(model, device, input_tensor)
                metadata = parse_json(metadata_path)
                frames.append(
                    ClosedLoopFrame(
                        tick=int(metadata["tick"]),
                        time_seconds=float(metadata["time_seconds"]),
                        semantic_prediction=prediction,
                        metadata=metadata,
                    )
                )
                visuals.append(
                    {
                        "width": int(prediction.shape[1]),
                        "height": int(prediction.shape[0]),
                        "input": flatten_2d(torch.round(input_tensor.squeeze(0) * 255.0).to(torch.uint8)),
                        "prediction": flatten_2d(prediction),
                        "metadata": metadata,
                    }
                )

    return frames, visuals


def actor_overlay_payload(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for actor in metadata.get("actors", []):
        if not isinstance(actor, dict):
            continue
        center = world_to_plan_point(float(actor.get("x", 0.0)), float(actor.get("y", 0.0)))
        payload.append(
            {
                "type": actor.get("type", "unknown"),
                "motion_state": actor.get("motion_state", "unknown"),
                "x": actor.get("x", 0.0),
                "y": actor.get("y", 0.0),
                "heading": actor.get("heading", 0.0),
                "center": None if center is None else {"row": center.row, "col": center.col},
            }
        )
    return payload


def build_payload(
    *,
    args: argparse.Namespace,
    episode: dict[str, Any],
    visuals: list[dict[str, Any]],
    checkpoint_epoch: int,
    device: torch.device,
) -> dict[str, Any]:
    steps = []
    for visual, step in zip(visuals, episode["steps"]):
        steps.append(
            {
                "tick": step["tick"],
                "time_seconds": step["time_seconds"],
                "width": visual["width"],
                "height": visual["height"],
                "input": visual["input"],
                "prediction": visual["prediction"],
                "metadata": visual["metadata"],
                "actor_overlay": actor_overlay_payload(visual["metadata"]),
                "ego_position": step["ego_position"],
                "goal": step["goal"],
                "collision": step["collision"],
                "goal_reached": step["goal_reached"],
                "predicted_classes": step["predicted_classes"],
                "forecast": step["forecast"],
                "extracted_state": step["extracted_state"],
                "ego_plan": step["ego_plan"],
                "ego_control": step["ego_control"],
                "command": step["command"],
            }
        )

    return {
        "seed": args.seed,
        "dt": args.dt,
        "noise": args.noise,
        "ticks_requested": args.ticks,
        "device": str(device),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint_epoch,
        "control_horizon": args.control_horizon,
        "class_names": list(SEMANTIC_CLASS_NAMES),
        "class_colors": {name: list(rgb) for name, rgb in SEMANTIC_CLASS_COLORS.items()},
        "episode": {
            key: value
            for key, value in episode.items()
            if key not in {"steps"}
        },
        "steps": steps,
    }


def render_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Closed-Loop Autonomy Viewer</title>
  <style>
    :root {{
      --bg: #efe7da;
      --panel: rgba(255, 251, 245, 0.92);
      --ink: #1f1a17;
      --muted: #695f56;
      --line: rgba(31, 26, 23, 0.12);
      --accent: #173d57;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: Georgia, 'Times New Roman', serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(255,255,255,0.7), transparent 30%),
        linear-gradient(180deg, #f7f1e8 0%, var(--bg) 100%);
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(300px, 380px) 1fr;
      gap: 24px;
      padding: 24px;
      align-items: start;
    }}
    .panel, .tile {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 20px;
      box-shadow: 0 16px 42px rgba(73, 55, 35, 0.08);
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: clamp(2rem, 4vw, 3.2rem);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }}
    .lede {{
      margin: 0 0 18px;
      color: var(--muted);
      line-height: 1.5;
    }}
    .meta, .stat-list {{
      display: grid;
      gap: 8px;
      color: var(--muted);
    }}
    .control-block {{
      display: grid;
      gap: 10px;
      margin-top: 18px;
    }}
    .toolbar {{
      display: flex;
      gap: 10px;
      margin-top: 18px;
      flex-wrap: wrap;
    }}
    select, input[type="range"], button {{
      width: 100%;
      font: inherit;
    }}
    button, select {{
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: white;
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
    canvas {{
      width: 100%;
      aspect-ratio: 1 / 1;
      image-rendering: pixelated;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: white;
    }}
    .metadata {{
      white-space: pre-wrap;
      background: rgba(255,255,255,0.75);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      font-size: 0.88rem;
      line-height: 1.45;
      color: var(--muted);
    }}
    .caption {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 0.92rem;
      line-height: 1.4;
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
      <h1>Closed-Loop Demo</h1>
      <p class="lede">Per-tick replay of the noisy BEV input, predicted semantic scene, and the ego agent replanning and stepping through the intersection.</p>
      <div class="meta">
        <div><strong>Seed:</strong> <span id="seedValue"></span></div>
        <div><strong>Noise:</strong> <span id="noiseValue"></span></div>
        <div><strong>Ticks Requested:</strong> <span id="ticksRequested"></span></div>
        <div><strong>Ticks Completed:</strong> <span id="ticksCompleted"></span></div>
        <div><strong>Checkpoint Epoch:</strong> <span id="checkpointEpoch"></span></div>
        <div><strong>Device:</strong> <span id="deviceName"></span></div>
        <div><strong>Success:</strong> <span id="successValue"></span></div>
        <div><strong>Collision:</strong> <span id="collisionValue"></span></div>
      </div>
      <div class="control-block">
        <label for="stepSelect">Tick</label>
        <select id="stepSelect"></select>
      </div>
      <div class="control-block">
        <label for="stepScrubber">Replay Tick</label>
        <input id="stepScrubber" type="range" min="0" max="0" value="0" />
      </div>
      <div class="toolbar">
        <button id="playToggle" type="button">Play Replay</button>
        <button id="restartButton" type="button">Restart</button>
      </div>
      <ul class="stat-list" id="stepStats"></ul>
    </aside>
    <main class="viewer">
      <div class="canvas-grid">
        <section class="tile">
          <h2>Noisy Input</h2>
          <canvas id="inputCanvas" width="256" height="256"></canvas>
          <div class="caption">Observed noisy BEV fed into the perception model at the current tick.</div>
        </section>
        <section class="tile">
          <h2>Closed-Loop Scene</h2>
          <canvas id="sceneCanvas" width="256" height="256"></canvas>
          <div class="caption">Predicted semantic scene with ground-truth actor markers, ego trail, forecast, plan, and current ego state.</div>
        </section>
      </div>
      <div class="canvas-grid">
        <section class="tile">
          <h2>Step Metadata</h2>
          <div id="metadataView" class="metadata"></div>
        </section>
        <section class="tile">
          <h2>Command Summary</h2>
          <div id="commandView" class="metadata"></div>
        </section>
      </div>
    </main>
  </div>
  <script>
    const payload = {data_json};
    const classNames = payload.class_names;
    const classColors = payload.class_colors;
    const steps = payload.steps;

    const seedValue = document.getElementById('seedValue');
    const noiseValue = document.getElementById('noiseValue');
    const ticksRequested = document.getElementById('ticksRequested');
    const ticksCompleted = document.getElementById('ticksCompleted');
    const checkpointEpoch = document.getElementById('checkpointEpoch');
    const deviceName = document.getElementById('deviceName');
    const successValue = document.getElementById('successValue');
    const collisionValue = document.getElementById('collisionValue');
    const stepSelect = document.getElementById('stepSelect');
    const stepScrubber = document.getElementById('stepScrubber');
    const playToggle = document.getElementById('playToggle');
    const restartButton = document.getElementById('restartButton');
    const stepStats = document.getElementById('stepStats');
    const metadataView = document.getElementById('metadataView');
    const commandView = document.getElementById('commandView');
    const inputCanvas = document.getElementById('inputCanvas');
    const sceneCanvas = document.getElementById('sceneCanvas');

    let currentStep = 0;
    let playing = false;
    let timer = null;

    seedValue.textContent = String(payload.seed);
    noiseValue.textContent = payload.noise;
    ticksRequested.textContent = String(payload.ticks_requested);
    ticksCompleted.textContent = String(payload.episode.ticks_completed);
    checkpointEpoch.textContent = String(payload.checkpoint_epoch);
    deviceName.textContent = payload.device;
    successValue.textContent = payload.episode.success ? 'yes' : 'no';
    collisionValue.textContent = payload.episode.collision ? 'yes' : 'no';

    function fillControls() {{
      steps.forEach((step, index) => {{
        const option = document.createElement('option');
        option.value = String(index);
        option.textContent = `tick ${{step.tick}}`;
        stepSelect.appendChild(option);
      }});
      stepScrubber.max = String(Math.max(steps.length - 1, 0));
    }}

    function stopPlayback() {{
      playing = false;
      playToggle.textContent = 'Play Replay';
      if (timer !== null) {{
        window.clearInterval(timer);
        timer = null;
      }}
    }}

    function drawInput(step) {{
      const ctx = inputCanvas.getContext('2d');
      const imageData = ctx.createImageData(step.width, step.height);
      for (let i = 0; i < step.input.length; i += 1) {{
        const value = step.input[i];
        const base = i * 4;
        imageData.data[base + 0] = value;
        imageData.data[base + 1] = value;
        imageData.data[base + 2] = value;
        imageData.data[base + 3] = 255;
      }}
      const offscreen = document.createElement('canvas');
      offscreen.width = step.width;
      offscreen.height = step.height;
      offscreen.getContext('2d').putImageData(imageData, 0, 0);
      ctx.clearRect(0, 0, inputCanvas.width, inputCanvas.height);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(offscreen, 0, 0, inputCanvas.width, inputCanvas.height);
    }}

    function drawPredictionBase(step) {{
      const ctx = sceneCanvas.getContext('2d');
      const imageData = ctx.createImageData(step.width, step.height);
      for (let i = 0; i < step.prediction.length; i += 1) {{
        const className = classNames[step.prediction[i]];
        const [r, g, b] = classColors[className];
        const base = i * 4;
        imageData.data[base + 0] = r;
        imageData.data[base + 1] = g;
        imageData.data[base + 2] = b;
        imageData.data[base + 3] = 255;
      }}
      const offscreen = document.createElement('canvas');
      offscreen.width = step.width;
      offscreen.height = step.height;
      offscreen.getContext('2d').putImageData(imageData, 0, 0);
      ctx.clearRect(0, 0, sceneCanvas.width, sceneCanvas.height);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(offscreen, 0, 0, sceneCanvas.width, sceneCanvas.height);
      return ctx;
    }}

    function drawLaneTracks(ctx, step) {{
      const scaleX = sceneCanvas.width / step.width;
      const scaleY = sceneCanvas.height / step.height;
      ctx.strokeStyle = '#ffffff';
      ctx.fillStyle = '#173d57';
      ctx.lineWidth = 3;
      (step.extracted_state.lane_tracks || []).forEach((track) => {{
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
      }});
    }}

    function drawForecast(ctx, step) {{
      if (!step.forecast) {{
        return;
      }}
      const scaleX = sceneCanvas.width / step.width;
      const scaleY = sceneCanvas.height / step.height;
      function drawPath(actor, color) {{
        if (!actor.trajectory || !actor.trajectory.length) {{
          return;
        }}
        ctx.strokeStyle = color;
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
      }}
      (step.forecast.vehicles || []).forEach((actor) => drawPath(actor, 'rgba(190, 74, 47, 0.85)'));
      (step.forecast.pedestrians || []).forEach((actor) => drawPath(actor, 'rgba(15, 124, 115, 0.85)'));
    }}

    function drawPlan(ctx, step) {{
      const plan = step.ego_plan;
      const scaleX = sceneCanvas.width / step.width;
      const scaleY = sceneCanvas.height / step.height;
      if (!plan || !plan.path || !plan.path.length) {{
        return;
      }}
      ctx.strokeStyle = 'rgba(23, 61, 87, 0.98)';
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
    }}

    function drawActorOverlay(ctx, step) {{
      const scaleX = sceneCanvas.width / step.width;
      const scaleY = sceneCanvas.height / step.height;
      (step.actor_overlay || []).forEach((actor) => {{
        if (!actor.center) {{
          return;
        }}
        const x = (actor.center.col + 0.5) * scaleX;
        const y = (actor.center.row + 0.5) * scaleY;
        if (actor.type === 'car') {{
          ctx.strokeStyle = '#2b2117';
          ctx.lineWidth = 2;
          ctx.strokeRect(x - 7, y - 4, 14, 8);
        }} else {{
          ctx.fillStyle = '#2b2117';
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, Math.PI * 2);
          ctx.fill();
        }}
      }});
    }}

    function drawEgoTrail(ctx, uptoIndex, step) {{
      const scaleX = sceneCanvas.width / step.width;
      const scaleY = sceneCanvas.height / step.height;
      ctx.strokeStyle = 'rgba(233, 179, 46, 0.95)';
      ctx.lineWidth = 3;
      ctx.setLineDash([8, 5]);
      ctx.beginPath();
      steps.slice(0, uptoIndex + 1).forEach((traceStep, index) => {{
        const x = (traceStep.ego_position.col + 0.5) * scaleX;
        const y = (traceStep.ego_position.row + 0.5) * scaleY;
        if (index === 0) {{
          ctx.moveTo(x, y);
        }} else {{
          ctx.lineTo(x, y);
        }}
      }});
      ctx.stroke();
      ctx.setLineDash([]);
    }}

    function drawEgo(ctx, step) {{
      const scaleX = sceneCanvas.width / step.width;
      const scaleY = sceneCanvas.height / step.height;
      const x = (step.ego_position.col + 0.5) * scaleX;
      const y = (step.ego_position.row + 0.5) * scaleY;
      ctx.fillStyle = step.collision ? '#be4a2f' : '#e9b32e';
      ctx.beginPath();
      ctx.arc(x, y, 6.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, 9.5, 0, Math.PI * 2);
      ctx.stroke();
    }}

    function renderStats(step, index) {{
      stepStats.innerHTML = '';
      const items = [
        `Tick: ${{step.tick}}`,
        `Replay Step: ${{index + 1}} / ${{steps.length}}`,
        `Time: ${{step.time_seconds.toFixed(2)}} s`,
        `Predicted Classes: ${{step.predicted_classes.join(', ') || 'none'}}`,
        `Ego Position: (${{step.ego_position.row}}, ${{step.ego_position.col}})`,
        `Goal: (${{step.goal.row}}, ${{step.goal.col}})`,
        `Collision: ${{step.collision ? 'yes' : 'no'}}`,
        `Goal Reached: ${{step.goal_reached ? 'yes' : 'no'}}`,
        `Command: ${{step.command ? step.command.action : 'none'}}`,
      ];
      items.forEach((text) => {{
        const li = document.createElement('li');
        li.textContent = text;
        stepStats.appendChild(li);
      }});
    }}

    function renderMetadata(step) {{
      metadataView.textContent = JSON.stringify(step.metadata, null, 2);
      commandView.textContent = JSON.stringify({{
        command: step.command,
        ego_plan: step.ego_plan,
        ego_control: step.ego_control,
      }}, null, 2);
    }}

    function renderStep(index) {{
      currentStep = index;
      const step = steps[index];
      stepSelect.value = String(index);
      stepScrubber.value = String(index);
      drawInput(step);
      const ctx = drawPredictionBase(step);
      drawLaneTracks(ctx, step);
      drawForecast(ctx, step);
      drawPlan(ctx, step);
      drawActorOverlay(ctx, step);
      drawEgoTrail(ctx, index, step);
      drawEgo(ctx, step);
      renderStats(step, index);
      renderMetadata(step);
    }}

    stepSelect.addEventListener('change', (event) => {{
      stopPlayback();
      renderStep(Number(event.target.value));
    }});
    stepScrubber.addEventListener('input', (event) => {{
      stopPlayback();
      renderStep(Number(event.target.value));
    }});
    playToggle.addEventListener('click', () => {{
      if (playing) {{
        stopPlayback();
        return;
      }}
      playing = true;
      playToggle.textContent = 'Pause Replay';
      timer = window.setInterval(() => {{
        if (currentStep >= steps.length - 1) {{
          currentStep = 0;
        }} else {{
          currentStep += 1;
        }}
        renderStep(currentStep);
      }}, 450);
    }});
    restartButton.addEventListener('click', () => {{
      stopPlayback();
      renderStep(0);
    }});

    fillControls();
    renderStep(0);
  </script>
</body>
</html>
"""


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    validate_args(args)

    device = resolve_device(args.device)
    model, checkpoint_payload = load_checkpoint(args.checkpoint, map_location=device)
    model = model.to(device)
    model.eval()

    frames, visuals = collect_frames(args=args, model=model, device=device)
    episode = run_closed_loop_episode(
        frames,
        dt=args.dt,
        control_horizon=args.control_horizon,
    ).to_dict()

    payload = build_payload(
        args=args,
        episode=episode,
        visuals=visuals,
        checkpoint_epoch=int(checkpoint_payload["epoch"]),
        device=device,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "closed_loop_run.json"
    html_path = args.output_dir / "closed_loop_viewer.html"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    html_path.write_text(render_html(payload), encoding="utf-8")

    print(f"wrote closed-loop summary to {summary_path}")
    print(f"wrote closed-loop viewer to {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
