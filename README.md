# Synthetic Driving Simulator

A lightweight **C++ + Python** project for generating synthetic driving scenes, exporting BEV training data, and training a small ML model to test how simulator choices affect downstream performance.

## Goal

Build a pure-software simulator that:

* generates simple traffic scenes
* renders a BEV sensor/output
* exports labels automatically
* trains a small **U-Net** on the synthetic data
* measures **speed, determinism, and ML performance**

## Why this project

This is meant to look like **simulation engineering first, ML second**.

It shows:

* **C++ systems work**
* **synthetic data generation**
* **reproducibility + benchmarking**
* **ML evaluation on generated data**

## Tech Stack

* **C++17/20** — simulator core
* **CMake** — builds
* **Python 3.11+** — dataset generation, training, analysis
* **PyTorch** — U-Net model
* **NumPy / Pandas / Matplotlib** — data + plots
* **GoogleTest** — C++ tests
* **YAML** — configs

## V1 Scope

Keep the first version small:

* one **4-way intersection**
* actor types: **cars + pedestrians**
* one output: **128x128 BEV semantic map**
* one ML model: **small U-Net**
* one experiment: **low-noise vs high-noise**

## ML Task

**Input:** noisy BEV occupancy / sensor raster
**Target:** clean semantic BEV map

Use a **small U-Net** to predict:

* drivable area
* lanes
* vehicles
* pedestrians
* obstacles

## Repo Structure

```text
synthetic-driving-sim/
├── README.md
├── CMakeLists.txt
├── configs/
├── cpp/
│   ├── include/
│   ├── src/
│   └── tests/
├── python/
│   ├── generate_dataset.py
│   ├── train_unet.py
│   ├── eval_model.py
│   └── run_benchmarks.py
├── data/
├── outputs/
└── docs/
```

## Implementation Plan

### 1. Simulator Core

Build in C++:

* world grid / map
* actors
* fixed timestep loop
* seeded determinism
* simple motion rules

### 2. Renderer + Labels

Export:

* BEV input image
* BEV semantic label image
* metadata JSON/CSV

### 3. Dataset Pipeline

Write Python scripts to:

* generate many scenes
* split train/val/test
* save samples + metadata

### 4. ML Baseline

Train a **small U-Net** in PyTorch and report:

* IoU
* pixel accuracy
* per-class IoU

### 5. Benchmarks

Measure:

* generation time per sample
* scenes/sec
* determinism under fixed seed
* model performance under different noise levels

## Success Criteria

V1 is done when you can:

* generate synthetic BEV scenes
* export labels automatically
* train U-Net end-to-end
* compare low-noise vs high-noise results
* show a small benchmark table

## Example Commands

```bash
cmake -S . -B build
cmake --build build
python python/generate_dataset.py --config configs/base.yaml
python python/train_unet.py --config configs/train.yaml
python python/eval_model.py --checkpoint outputs/models/unet.pt
python python/run_benchmarks.py --config configs/base.yaml
```

## Resume Pitch

Built a C++ driving simulator that generates synthetic BEV training data, then trained a PyTorch U-Net to evaluate how simulator noise and scene complexity affect downstream perception performance.
