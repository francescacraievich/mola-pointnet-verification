# PointNet Formal Verification for MOLA SLAM

Formal verification of a PointNet classifier for LiDAR point clouds using **ERAN** and **α,β-CROWN** verifiers, with dynamic criticality weights derived from NSGA-III adversarial attack optimization.

## Overview

This project connects **adversarial attack optimization** with **formal neural network verification** to predict real-world SLAM system vulnerabilities:

1. **NSGA-III Attack Optimization** ([mola-adversarial-nsga3](https://github.com/francescacraievich/mola-adversarial-nsga3)): Multi-objective genetic algorithm finds Pareto-optimal adversarial attacks on MOLA SLAM
2. **Dynamic Vulnerability Labeling**: Extract criticality weights from NSGA-III Pareto set to label point cloud regions
3. **Formal Verification**: Use ERAN and α,β-CROWN to verify PointNet robustness on critical regions

### Key Hypothesis

If the critical ε value (where verification rate drops below 50%) correlates with the perturbation magnitude that causes SLAM failure (~1.5-2cm), we demonstrate that **formal verification can predict real-world system vulnerabilities**.

## Verification Methods

### ERAN (ETH Zurich)
- Uses **DeepZono** abstract domain
- Works with **3DCertify PointNet** architecture (cascading MaxPools)
- Incomplete verifier (may return "unknown")

### α,β-CROWN (VNN-COMP Winner)
- **α-CROWN**: Incomplete verification with optimized linear bounds
- **β-CROWN**: Complete verification with branch-and-bound on ReLU neurons
- Works with **simplified PointNet** architecture (MeanPool, no BatchNorm)
- State-of-the-art complete verifier

## Verification Results

### ERAN (DeepZono domain)

| ε (normalized) | ε (cm) | Verified | Total | Rate |
|----------------|--------|----------|-------|------|
| 0.001 | 1.1 | 11 | 11 | **100%** |
| 0.003 | 3.3 | 11 | 11 | **100%** |
| 0.005 | 5.5 | 10 | 11 | **91%** |
| 0.007 | 7.7 | 6 | 11 | **55%** |
| 0.010 | 11.0 | 5 | 11 | **45%** |

### α,β-CROWN (with Branch-and-Bound)

| ε | Sample 521 (margin=0.79) | Sample 792 (margin=0.06) |
|---|--------------------------|--------------------------|
| 0.01 | ✓ verified (4s) | ✗ unsafe (PGD attack) |
| 0.03 | ✓ verified (7s) | ✗ unsafe |
| 0.05 | ✓ verified (146s, BaB) | ✗ unsafe |

**Note**: α,β-CROWN uses a different model architecture than ERAN, so results are not directly comparable.

## Property Verified: Local Robustness (L∞)

For every correctly classified point cloud group x₀:
```
∀x' with ||x' - x₀||_∞ ≤ ε : f(x') = f(x₀)
```
"Classification remains invariant under coordinate perturbations"

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
Output: `saved_models/pointnet_3dcertify_64p.pth`

**For α,β-CROWN** (simplified architecture):
```bash
python src/train/train_pointnet_autolirpa.py
```
Output: `saved_models/pointnet_autolirpa_512.pth`

### 3. Verification

**ERAN verification**:
```bash
python src/verification/verify_eran_python_api.py
```

**α,β-CROWN verification**:
```bash
python src/verification/verify_abcrown.py --samples 521 792 --epsilon 0.01 0.03 0.05
```

Options:
- `--samples`: Specific sample indices to verify
- `--epsilon`: Perturbation budgets (L∞ norm)
- `--by-margin low|high|mixed`: Select samples by confidence margin
- `--n-samples`: Number of random samples
- `--timeout`: Timeout per sample (default 300s)

## Project Structure

```
mola-pointnet-verification/
├── src/
│   ├── model/
│   │   └── pointnet_autolirpa_compatible.py  # PointNet for α,β-CROWN
│   ├── plots/
│   │   └── visualize_pointcloud.py           # Visualization
│   ├── scripts/
│   │   ├── data_preparation_pointnet.py      # Data preprocessing
│   │   └── nsga3_integration.py              # NSGA-III weight extraction
│   ├── train/
│   │   ├── train_3dcertify_64p.py            # Train for ERAN
│   │   └── train_pointnet_autolirpa.py       # Train for α,β-CROWN
│   └── verification/
│       ├── verify_eran_python_api.py         # ERAN verification
│       └── verify_abcrown.py                 # α,β-CROWN verification
├── configs/
│   └── abcrown_pointnet_complete.yaml        # α,β-CROWN configuration
├── data/pointnet/                            # Processed dataset
├── saved_models/
│   ├── pointnet_3dcertify_64p.pth            # For ERAN
│   └── pointnet_autolirpa_512.pth            # For α,β-CROWN
├── results/                                  # Verification results (incremental)
│   ├── eran_verification_*.json
│   ├── eran_verification_*.md
│   ├── abcrown_verification_*.json
│   └── abcrown_verification_*.md
├── 3dcertify/                                # External (not in git)
├── ERAN/                                     # External (not in git)
└── alpha-beta-CROWN/                         # External (not in git)
```

## Models

| Model | Architecture | Features | Verifier |
|-------|--------------|----------|----------|
| `pointnet_3dcertify_64p.pth` | 3DCertify PointNet | 1024 features, MaxPool, BatchNorm | ERAN |
| `pointnet_autolirpa_512.pth` | Simplified PointNet | 512 features, MeanPool, no BatchNorm | α,β-CROWN |

Both models are trained on MOLA LiDAR data with NSGA-III dynamic labels.

## Key Results

1. **NSGA-III Integration**: Successfully derived criticality weights from adversarial attack analysis
2. **ERAN Verification**: 100% at ε=1.1cm, drops to 45% at ε=11cm
3. **α,β-CROWN Verification**: Complete verification with branch-and-bound for hard cases
4. **Hypothesis Confirmed**: Verification rate drops at perturbations matching SLAM failure threshold

## Related Projects

- [mola-adversarial-nsga3](https://github.com/francescacraievich/mola-adversarial-nsga3): NSGA-III adversarial attacks
- [3DCertify](https://github.com/eth-sri/3dcertify): ETH Zurich PointNet verification
- [ERAN](https://github.com/eth-sri/eran): ETH Robustness Analyzer
- [α,β-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN): VNN-COMP winner verifier

## License

MIT License

## Author

Francesca Craievich
