# NSGA3-Based Verification

This document explains how the verification connects to MOLA adversarial attack optimization.

## Overview

Traditional verification uses static labels (CRITICAL vs NON_CRITICAL) defined manually.
With NSGA3 integration, we compute **dynamic labels** based on vulnerability to optimal attacks.

## The Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                  MOLA Adversarial Attack Optimization            │
│                     (mola-adversarial-nsga3)                     │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           │ NSGA3 Multi-Objective Optimization
                           │ • Maximize ATE (attack effectiveness)
                           │ • Minimize perturbation magnitude
                           │
                           ▼
                    ┌──────────────┐
                    │ Pareto Set   │
                    │ 26 solutions │  Each genome: 17 attack parameters
                    │ (genomes)    │  (noise, distortion, targeting, etc.)
                    └──────┬───────┘
                           │
                           │ Load Pareto-optimal genomes
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│              Dynamic Vulnerability Assessment                    │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           │ For each point cloud region:
                           │ 1. Apply each Pareto genome
                           │ 2. Compute vulnerability score
                           │ 3. max_vuln = max(all genome scores)
                           │
                           ▼
                    ┌──────────────┐
                    │   Labeling   │
                    │ threshold=0.5│
                    └──────┬───────┘
                           │
              ┌────────────┴────────────┐
              │                         │
    max_vuln >= 0.5         max_vuln < 0.5
              │                         │
              ▼                         ▼
        CRITICAL (0)            NON_CRITICAL (1)
    High vulnerability       Low vulnerability
              │                         │
              └────────────┬────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│              Neural Network Verification                         │
│                   (α,β-CROWN)                                    │
└──────────────────────────────────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
    Property 1: Robustness   Property 2: Safety
    pred(x+δ) == pred(x)     CRITICAL never → NON_CRITICAL
```

## Results Interpretation

**Run 10 Statistics** (from mola-adversarial-nsga3):
- **26 Pareto-optimal solutions** balancing ATE vs perturbation
- **ATE range**: 25.2 - 105.9 cm (attack effectiveness)
- **Perturbation range**: 1.38 - 4.86 cm
- **Baseline ATE**: 23.0 cm (no attack)

**Dynamic Labels** (from PointNet test set):
- **CRITICAL (0)**: 151 samples (7.6%)
  - Regions where ≥1 Pareto-optimal attack achieves high vulnerability
  - Examples: Sharp edges, high-curvature surfaces, low-density areas
- **NON_CRITICAL (1)**: 1838 samples (92.4%)
  - Regions robust to all Pareto-optimal attacks

**Verification with ε=0.005 (~0.8cm)**:
- Tests whether PointNet predictions are robust within 0.8cm perturbations
- Connects to NSGA3: ε << min_perturbation (1.38cm)
- This verifies robustness BEFORE the attack threshold

## Usage

### Basic Verification (Static Labels)
```bash
python scripts/verify_with_abcrown_nsga3.py \
  --n-samples 10 \
  --epsilon 0.005 \
  --property robustness
```

### NSGA3-Based Verification (Dynamic Labels)
```bash
python scripts/verify_with_abcrown_nsga3.py \
  --use-nsga3 \
  --n-samples 10 \
  --epsilon 0.005 \
  --property robustness
```

### Advanced Options
```bash
python scripts/verify_with_abcrown_nsga3.py \
  --use-nsga3 \
  --nsga3-dir /path/to/nsga3/results/runs \
  --nsga3-run-id 10 \
  --vulnerability-threshold 0.5 \
  --n-samples 20 \
  --epsilon 0.005 0.01 0.02 \
  --property both
```

### Parameters

- `--use-nsga3`: Enable dynamic labeling from NSGA3 Pareto set
- `--nsga3-dir`: Path to NSGA3 results (default: `/home/francesca/mola-adversarial-nsga3/src/results/runs`)
- `--nsga3-run-id`: Which NSGA3 run to load (default: 10)
- `--vulnerability-threshold`: Threshold for CRITICAL vs NON_CRITICAL (default: 0.5)
- `--epsilon`: Perturbation bounds to test (default: [0.005, 0.01, 0.02])
- `--property`: `robustness`, `safety`, or `both`

## Key Insight

**Why this matters**: Traditional verification uses arbitrary labels. With NSGA3, labels represent **actual vulnerability to optimal attacks**.

- If a region is labeled CRITICAL by NSGA3, it means Pareto-optimal attacks exploit it
- If verification proves robustness for CRITICAL regions, we know the model is robust even against optimal attacks
- This creates a **provable safety margin** between ε and the attack perturbation threshold

## Example Output

```
Loading NSGA3 Pareto set...
  Directory: /home/francesca/mola-adversarial-nsga3/src/results/runs
  Run ID: 10
  ✓ Loaded Pareto set: (26, 17)
  Pareto solutions: 26
  ATE range: 25.2 - 105.9 cm
  Perturbation range: 1.38 - 4.86 cm
  Vulnerability threshold: 0.5

Computing dynamic labels from NSGA3 Pareto set...
  ✓ Computed labels: 1989 samples
     CRITICAL (0): 151 (7.6%)
     NON_CRITICAL (1): 1838 (92.4%)

======================================================================
Epsilon = 0.005 (~0.8 cm in MOLA coordinates)
======================================================================

  [1/10] Sample 843 (NON_CRITICAL): ✓ VERIFIED
  [2/10] Sample 1265 (NON_CRITICAL): ✓ VERIFIED
  ...
```

## File Structure

```
mola-pointnet-verification/
├── scripts/
│   └── verify_with_abcrown_nsga3.py    # Main verification script
├── src/
│   └── nsga3_integration.py            # NSGA3 loading & vulnerability computation
└── results/
    └── abcrown_verification/
        ├── verification_results_robustness.json
        └── verification_results_safety.json

mola-adversarial-nsga3/
└── src/results/runs/
    ├── optimized_genome10.pareto_set.npy   # 26 Pareto-optimal genomes
    ├── optimized_genome10.pareto_front.npy # ATE & perturbation values
    └── optimized_genome10.valid_genomes.npy # All evaluated genomes
```

## References

1. **NSGA3 Optimization**: `mola-adversarial-nsga3` project
2. **α,β-CROWN Verification**: [alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN)
3. **Integration Module**: [src/nsga3_integration.py](../src/nsga3_integration.py)
