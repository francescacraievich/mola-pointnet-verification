# PointNet Formal Verification for MOLA SLAM

[![CI](https://github.com/francescacraievich/mola-pointnet-verification/actions/workflows/ci.yml/badge.svg)](https://github.com/francescacraievich/mola-pointnet-verification/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/francescacraievich/mola-pointnet-verification/branch/main/graph/badge.svg)](https://codecov.io/gh/francescacraievich/mola-pointnet-verification)

Formal verification of a PointNet classifier for LiDAR point clouds using **ERAN** and **α,β-CROWN** verifiers, with dynamic criticality weights derived from NSGA-III adversarial attack optimization.

## Overview

This project connects **adversarial attack optimization** with **formal neural network verification** to predict real-world SLAM system vulnerabilities:

1. **NSGA-III Attack Optimization** ([mola-adversarial-nsga3](https://github.com/francescacraievich/mola-adversarial-nsga3)): Multi-objective genetic algorithm finds Pareto-optimal adversarial attacks on MOLA SLAM
2. **Dynamic Vulnerability Labeling**: Extract criticality weights from NSGA-III Pareto set to label point cloud regions as CRITICAL or NON_CRITICAL
3. **Formal Verification**: Use ERAN and α,β-CROWN to verify PointNet robustness under L∞ perturbations

### Key Hypothesis

If the critical ε value (where verification rate drops significantly) correlates with the perturbation magnitude that causes SLAM failure (~1.5-2cm), we demonstrate that **formal verification can predict real-world system vulnerabilities**.

## Dataset

- **Source**: MOLA LiDAR SLAM system (14.4M raw points across 113 frames)
- **Preprocessing**: K-NN grouping (k=1024) extracts local regions around each point
- **Samples**: 4,881 training + 1,000 test samples
- **Subsampling**: 1024 → 64 points per region (for verification tractability)
- **Labels**: Binary classification (CRITICAL=0, NON_CRITICAL=1) based on NSGA-III weights

## Verification Setup

### Input Representation

Each sample is a **local region** of 64 LiDAR points:
- Input shape: `(64, 3)` → 64 points × 3 coordinates (x, y, z)
- Total input dimensions: 192 values

### Perturbation Model (L∞)

For perturbation budget ε, each coordinate can vary independently:
```
Original point: (x, y, z)
Perturbed box:  [x-ε, x+ε] × [y-ε, y+ε] × [z-ε, z+ε]
```

The verifier checks **all** possible perturbations of **all 64 points simultaneously** (an infinite set in a 192-dimensional hypercube).

### Property Verified: Local Robustness

For every correctly classified point cloud region x₀:
```
∀x' with ||x' - x₀||_∞ ≤ ε : f(x') = f(x₀)
```
"Classification remains invariant under coordinate perturbations"

## Verification Methods

### ERAN (ETH Zurich)

- **Domain**: DeepZono (zonotope-based abstract interpretation)
- **Model**: 3DCertify PointNet architecture
  - 64 input points, 512 max features
  - Cascading MaxPool (improved_max)
  - BatchNorm layers
- **Type**: Incomplete verifier (sound but may return "unknown")
- **Speed**: Fast (~1-2 seconds per sample)

### α,β-CROWN (VNN-COMP Winner)

- **Method**: Linear bound propagation with branch-and-bound
- **Model**: Simplified PointNet architecture
  - 64 input points, 512 max features
  - MeanPool (more stable for bound propagation)
  - No BatchNorm (compatibility with auto_LiRPA)
- **Type**: Complete verifier (can prove or find counterexample)
- **Speed**: Slower, especially for large ε (timeout at ε ≥ 0.1)

## Verification Results

### Configuration

| Parameter | ERAN | α,β-CROWN |
|-----------|------|-----------|
| Input points | 64 | 64 |
| Max features | 512 | 512 |
| Pooling | MaxPool | MeanPool |
| BatchNorm | Yes | No |
| Samples | 100 | 100 |
| Selection | Random (correctly classified) | Random (correctly classified) |

### ERAN Results (DeepZono)

| ε | Verified | Total | Rate |
|---|----------|-------|------|
| 0.001 | 99 | 100 | **99%** |
| 0.003 | 96 | 100 | **96%** |
| 0.005 | 94 | 100 | **94%** |
| 0.007 | 78 | 100 | **78%** |
| 0.01 | 53 | 100 | **53%** |
| 0.02 | 2 | 100 | **2%** |

### α,β-CROWN Results

| ε | Verified | Unsafe | Timeout | Total | Rate |
|---|----------|--------|---------|-------|------|
| 0.001 | 100 | 0 | 0 | 100 | **100%** |
| 0.003 | 100 | 0 | 0 | 100 | **100%** |
| 0.005 | 99 | 1 | 0 | 100 | **99%** |
| 0.007 | 98 | 1 | 1 | 99 | **99%** |
| 0.01 | 98 | 2 | 0 | 100 | **98%** |
| 0.02 | 97 | 3 | 0 | 100 | **97%** |

**Note**: Both verifiers use 512 max features for computational tractability. α,β-CROWN achieves higher verification rates due to its complete verification approach, maintaining >95% robustness even at ε=0.02 where ERAN drops to 2%.

## Installation

### Basic Setup

```bash
git clone https://github.com/francescacraievich/mola-pointnet-verification.git
cd mola-pointnet-verification

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### External Dependencies

```bash
# 3DCertify (for ERAN)
git clone https://github.com/eth-sri/3dcertify.git 3dcertify

# ERAN (ETH Zurich)
git clone https://github.com/eth-sri/eran.git ERAN
cd ERAN && git checkout 61e3667a4d59efefd195e3623bb1ba483d41332c && cd ..

# α,β-CROWN
git clone https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
```

## Usage

### 1. Data Preparation

```bash
python src/scripts/data_preparation_pointnet.py \
    --source /path/to/mola-data \
    --output data/pointnet \
    --n-points 64
```

### 2. Model Training

**For ERAN** (3DCertify architecture):
```bash
python src/train/train_3dcertify_64p.py
```
Output: `saved_models/pointnet_3dcertify_64p.pth` (accuracy: ~74%)

**For α,β-CROWN** (simplified architecture):
```bash
python src/train/train_pointnet_autolirpa.py
```
Output: `saved_models/pointnet_autolirpa_512.pth` (accuracy: ~72%)

### 3. Verification

**ERAN verification**:
```bash
python src/verification/verify_eran_python_api.py
```

**α,β-CROWN verification**:
```bash
python src/verification/verify_abcrown.py \
    --n-samples 100 \
    --epsilon 0.001 0.003 0.005 0.007 0.01 0.02 0.03
```

Options:
- `--n-samples`: Number of samples to verify
- `--epsilon`: Perturbation budgets (L∞ norm, in meters)
- `--timeout`: Timeout per sample (default 300s)

### 4. Visualization

```bash
# Plot verification comparison
python src/plots/plot_verification_results.py

# Animate point cloud regions
python src/plots/animate_pointcloud.py
```

## Project Structure

```
mola-pointnet-verification/
├── src/
│   ├── model/
│   │   └── pointnet_autolirpa_compatible.py  # PointNet for α,β-CROWN
│   ├── plots/
│   │   ├── visualize_pointcloud.py           # Static visualization
│   │   ├── animate_pointcloud.py             # Animated visualization
│   │   └── plot_verification_results.py      # Results comparison plots
│   ├── scripts/
│   │   ├── data_preparation_pointnet.py      # Data preprocessing
│   │   └── nsga3_integration.py              # NSGA-III weight extraction
│   ├── train/
│   │   ├── train_3dcertify_64p.py            # Train for ERAN
│   │   └── train_pointnet_autolirpa.py       # Train for α,β-CROWN
│   ├── verification/
│   │   ├── verify_eran_python_api.py         # ERAN verification
│   │   └── verify_abcrown.py                 # α,β-CROWN verification
│   └── tests/                                # Unit tests
├── data/configs/
│   └── abcrown_pointnet_complete.yaml        # α,β-CROWN configuration
├── data/pointnet/                            # Processed dataset
│   ├── train_groups.npy                      # Training samples (4881, 1024, 7)
│   ├── train_labels.npy                      # Training labels
│   ├── test_groups.npy                       # Test samples (1000, 1024, 7)
│   └── test_labels.npy                       # Test labels
├── saved_models/
│   ├── pointnet_3dcertify_64p.pth            # For ERAN (1024 features)
│   └── pointnet_autolirpa_512.pth            # For α,β-CROWN (512 features)
├── results/                                  # Verification results
│   ├── eran_verification_*.json
│   ├── abcrown_verification_*.json
│   ├── verification_comparison.png
│   └── verification_comparison_linear.png
├── 3dcertify/                                # External (not in git)
├── ERAN/                                     # External (not in git)
└── alpha-beta-CROWN/                         # External (not in git)
```

## Models Comparison

| Aspect | ERAN Model | α,β-CROWN Model |
|--------|------------|-----------------|
| Architecture | 3DCertify PointNet | Simplified PointNet |
| Input points | 64 | 64 |
| Max features | 512 | 512 |
| Pooling | Cascading MaxPool | MeanPool |
| BatchNorm | Yes | No |
| Dropout | Yes (0.3) | No |
| Test accuracy | ~74% | ~72% |
| Verification speed | Fast | Slow for large ε |

Both models are trained on MOLA LiDAR data with NSGA-III dynamic criticality labels.

## Key Findings

1. **NSGA-III Integration**: Successfully derived criticality weights from adversarial attack Pareto analysis
2. **Verification Scalability**: 64 points is a practical trade-off between geometric fidelity and verification tractability
3. **Tool Comparison**:
   - ERAN (DeepZono) is faster but incomplete (may report "unknown")
   - α,β-CROWN is complete but computationally expensive for large perturbations
4. **Critical Epsilon**: Verification rate drops below 50% around ε=0.01 (1cm), which aligns with perturbation magnitudes that affect SLAM performance

## Limitations

- **Subsampling**: 64 points is sparse compared to standard benchmarks (ModelNet40 uses 1024)
- **α,β-CROWN Timeout**: Complete verification becomes intractable for ε ≥ 0.1
- **Different Models**: ERAN and α,β-CROWN use different architectures, so results are not directly comparable
- **Incomplete Verification**: ERAN may fail to verify samples that are actually robust

## Related Projects

- [mola-adversarial-nsga3](https://github.com/francescacraievich/mola-adversarial-nsga3): NSGA-III adversarial attacks on MOLA SLAM
- [3DCertify](https://github.com/eth-sri/3dcertify): ETH Zurich PointNet verification
- [ERAN](https://github.com/eth-sri/eran): ETH Robustness Analyzer for Neural Networks
- [α,β-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN): VNN-COMP winning neural network verifier

## References

- Fischer et al., "Scalable Certified Segmentation via Randomized Smoothing", ICML 2021
- Wang et al., "Beta-CROWN: Efficient Bound Propagation with Per-neuron Split Constraints", NeurIPS 2021
- Qi et al., "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation", CVPR 2017

## License

MIT License

## Author

Francesca Craievich
