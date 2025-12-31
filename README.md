# MLP-LiDAR Formal Verification with αβ-CROWN

[![CI](https://github.com/francescacraievich/mola-pointnet-verification/actions/workflows/ci.yml/badge.svg)](https://github.com/francescacraievich/mola-pointnet-verification/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/francescacraievich/mola-pointnet-verification/graph/badge.svg)](https://codecov.io/gh/francescacraievich/mola-pointnet-verification)

Formal verification of an MLP classifier for LiDAR point clouds using αβ-CROWN, with comparison to adversarial attacks from NSGA-III optimization.

## Overview

This project verifies the robustness of a neural network classifier for LiDAR point clouds. The goal is to demonstrate that formal verification can predict real-world system failures by comparing:

1. **Formal Verification**: Using αβ-CROWN to prove robustness properties
2. **Empirical Attacks**: NSGA-III adversarial perturbations on MOLA SLAM (from [mola-adversarial-nsga3](https://github.com/francescacraievich/mola-adversarial-nsga3))

### Key Hypothesis

If the critical ε value (where verification rate drops below 50%) correlates with the perturbation magnitude that causes SLAM failure (~1.5-2cm), we demonstrate that formal verification predicts real-world system vulnerabilities.

## Properties Verified

### Property 1: Local Robustness (L∞)
For every correctly classified point x₀:
```
∀x' with ||x' - x₀||_∞ ≤ ε : f(x') = f(x₀)
```
"Classification remains invariant under perturbation"

### Property 2: Safety Property
For every point x₀ classified as OBSTACLE:
```
∀x' with ||x' - x₀||_∞ ≤ ε : f(x') ≠ GROUND
```
"An obstacle is never misclassified as drivable ground"

Both properties are tested for ε ∈ {0.01, 0.02, 0.03, 0.05, 0.10} meters.

## NSGA-III Results (Baseline)

From the adversarial attack project:
| Perturbation | ATE (SLAM Error) | Status |
|--------------|------------------|--------|
| 0 cm | 23 cm | Baseline |
| 1.5 cm | 32 cm | SLAM degrades |
| 3.5 cm | 65 cm | Significant degradation |
| 4.6 cm | 85 cm | SLAM fails |

**Critical threshold**: ~1.5-2cm perturbation causes unacceptable SLAM degradation.

## Installation

```bash
# Clone the repository
git clone https://github.com/francescacraievich/mola-mlp-verification.git
cd mola-mlp-verification

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For verification, install αβ-CROWN separately:
# https://github.com/Verified-Intelligence/alpha-beta-CROWN
```

## Usage

### 1. Data Preparation

Process raw LiDAR point clouds and generate heuristic labels:

```bash
python src/data_preparation.py
```

This will:
- Load 113 frames from `data/raw/frame_sequence.npy`
- Apply geometric heuristics to label points (GROUND/OBSTACLE/OTHER)
- Subsample to ~6000 points per frame (balanced classes)
- Normalize data and split 80/20 train/test
- Save to `data/processed/`

### 2. Model Training

```bash
python src/train.py
```

Trains the MLP classifier (~100K parameters) to achieve >80% accuracy.

### 3. Export to ONNX

```bash
python src/export_onnx.py
```

Exports the trained model to ONNX format for αβ-CROWN.

### 4. Verification

```bash
python src/verify.py
```

Runs formal verification with αβ-CROWN for all ε values.

### 5. Analysis

```bash
python src/analyze_results.py
```

Generates comparison plots and identifies critical ε values.

## Project Structure

```
mola-mlp-verification/
├── data/
│   ├── raw/                    # Original point clouds
│   │   └── frame_sequence.npy
│   └── processed/              # Processed dataset
│       ├── train_points.npy
│       ├── train_labels.npy
│       ├── test_points.npy
│       ├── test_labels.npy
│       └── normalization_params.npy
├── src/
│   ├── data_preparation.py     # Data loading + heuristic labeling
│   ├── model.py                # MLP architecture definition
│   ├── train.py                # Training script
│   ├── export_onnx.py          # ONNX export
│   ├── verify.py               # αβ-CROWN verification
│   └── analyze_results.py      # Results analysis and plots
├── configs/
│   ├── train_config.yaml       # Training hyperparameters
│   └── verification_config.yaml # αβ-CROWN configuration
├── models/
│   ├── mlp_lidar.pth           # PyTorch weights
│   └── mlp_lidar.onnx          # ONNX model
├── results/
│   ├── training_log.json       # Training metrics
│   ├── verification_results.json
│   └── figures/
│       ├── verified_vs_epsilon.png
│       ├── comparison_nsga3.png
│       └── certified_accuracy.png
├── notebooks/                  # Optional exploration
├── requirements.txt
└── README.md
```

## Network Architecture

MLP designed for efficient verification (~100K parameters):

```
Input (3) → Linear(256) → ReLU → Linear(256) → ReLU → Linear(128) → ReLU → Linear(3)
```

- **Input**: 3 features (x, y, z coordinates)
- **Output**: 3 classes (GROUND, OBSTACLE, OTHER)
- **No BatchNorm/Dropout**: Simplified for verification

## Labeling Heuristics

Based on the Isaac Sim environment geometry:

| Class | Rule | Description |
|-------|------|-------------|
| GROUND (0) | z < -0.3m | Below robot level |
| OBSTACLE (1) | 0.2m < z < 2.5m AND dist_xy < 15m | Walls, objects |
| OTHER (2) | Everything else | Sky, far points, noise |

## Expected Results

1. **Training**: >80% accuracy on test set
2. **Verification**: Decreasing verified % as ε increases
3. **Critical ε**: Value where verified rate drops below 50%
4. **Correlation**: Critical ε should be near 1.5-2cm (NSGA-III threshold)

## Related Work

- [mola-adversarial-nsga3](https://github.com/francescacraievich/mola-adversarial-nsga3): NSGA-III adversarial attacks on MOLA SLAM
- [αβ-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN): Neural network verifier

## License

MIT License

## Author

Francesca Craievich - Safe and Verified AI Course Project
