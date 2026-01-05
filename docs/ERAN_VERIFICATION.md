# ERAN Verification Guide

This guide explains how to set up and run formal verification of the PointNet classifier using ERAN/ELINA with 3DCertify's improved relaxations.

## Overview

**ERAN (ETH Robustness Analyzer)** is a formal verification tool that can prove neural network robustness against L∞ perturbations. We use it with **3DCertify**, which provides specialized relaxations for MaxPool operations in point cloud networks.

**Verification Goal**: Prove that for a given point cloud sample, if any point's XYZ coordinates change by at most ε (epsilon), the model's prediction remains correct.

## Architecture

The verification uses a modified PointNet architecture compatible with ERAN:

- **Input**: (batch, 3, 1024, 1) - 4D tensor with only XYZ coordinates
- **MLPs**: Conv1d layers (3→64→64, 64→64→128→1024)
- **Pooling**: Cascaded MaxPool1d with kernel_size=8 (improved_max)
- **Classifier**: Linear layers (1024→512→256→2) without Dropout
- **No T-Net**: Removed for verification compatibility

Key differences from training model:
- Uses only XYZ coordinates (not all 7 features)
- Uses `nn.MaxPool1d` instead of `torch.max()` for ERAN compatibility
- No Dropout layers (deterministic behavior required)
- Transposed input format: (batch, 3, n_points) instead of (batch, n_points, 3)

## Setup Instructions

### 1. Clone and Compile 3DCertify + ERAN

```bash
cd /home/francesca/mola-pointnet-verification
bash scripts/setup_3dcertify.sh
```

This script will:
- Clone 3DCertify repository with ERAN submodule
- Compile ELINA library (abstract domain backend)
- Install ELINA system-wide
- Configure LD_LIBRARY_PATH
- Test ELINA Python interface

**Requirements**:
- `sudo` access (for ELINA installation)
- C compiler (gcc)
- Make
- Git

**Time**: ~10-15 minutes (mostly compilation)

### 2. Apply Compatibility Patches

```bash
source ~/.bashrc  # Reload environment
python scripts/patch_3dcertify.py
```

This patches 3DCertify to handle PyTorch 2.x ONNX exports:
- `util/translate_onnx.py` - Handle `auto_pad`, `storage_order`, `dilations` attributes
- `ERAN/tf_verify/deeppoly_nodes.py` - Add missing `predecessors` arguments

### 3. Install Python Dependencies

```bash
pip install -r requirements_eran.txt
```

Installs:
- ONNX 1.5.0 (compatible with ERAN)
- pycddlib, mpmath (required by ERAN)
- pyyaml (for configuration)

### 4. Export Model to ONNX

```bash
python scripts/export_for_eran.py \
    --model models/pointnet.pth \
    --output models/pointnet_eran.onnx \
    --num_points 1024 \
    --num_classes 2 \
    --pool_function improved_max
```

This script:
1. Loads your trained PointNet model
2. Creates a verification-compatible architecture
3. Transfers weights (only matching layers)
4. Exports to ONNX using 3DCertify's converter
5. Validates the exported model

**Output**: `models/pointnet_eran.onnx` (4D input format for ERAN)

## Running Verification

### Basic Usage

```bash
python scripts/verify_eran_local.py --config configs/eran_verification.yaml
```

### Quick Test (10 samples)

```bash
python scripts/verify_eran_local.py \
    --config configs/eran_verification.yaml \
    --num-samples 10 \
    --epsilon 0.001
```

### Custom Epsilon Values

```bash
python scripts/verify_eran_local.py \
    --config configs/eran_verification.yaml \
    --epsilon 0.001 0.005 0.01
```

## Configuration

Edit [`configs/eran_verification.yaml`](../configs/eran_verification.yaml) to customize:

### Model Settings

```yaml
model:
  path: models/pointnet.pth
  onnx_path: models/pointnet_eran.onnx
  num_points: 1024
  num_classes: 2
  pool_function: improved_max  # or 'max'
```

### Verification Settings

```yaml
verification:
  domain: deeppoly  # or 'refinepoly', 'neurify'
  epsilons: [0.001, 0.003, 0.005, 0.007, 0.01]
  num_samples: null  # null = all samples
  timeout: 120  # seconds per sample
```

### Data Paths

```yaml
data:
  test_groups: data/pointnet/test_groups.npy
  test_labels: data/pointnet/test_labels.npy
```

## Understanding Results

### Verification Output

For each epsilon, the script reports:

- **Verified**: Samples proven robust (prediction guaranteed correct under perturbation)
- **Unknown**: Could not prove or disprove (timeout or imprecise bounds)
- **Misclassified**: Sample incorrectly classified (skipped)
- **Timeout**: Verification exceeded time limit
- **Error**: Verification failed with error

### Example Output

```
Results for ε=0.001:
  Verified:        45/50 (90.0%)
  Unknown:         5
  Misclassified:   10
  Timeout:         0
  Error:           0
  Avg time:        8.23s
```

**Interpretation**:
- Out of 60 total samples, 50 were correctly classified
- Of those 50, ERAN proved 45 (90%) are robust to ε=0.001 perturbations
- 5 samples could not be verified (but may still be robust)
- 10 samples were misclassified by the model (excluded from verification)

### Saved Results

Results are saved to `results/eran/eran_verification_YYYYMMDD_HHMMSS.json`:

```json
{
  "model_accuracy": 0.833,
  "results": {
    "epsilon_0.001": {
      "verified": 45,
      "unknown": 5,
      "misclassified": 10,
      "samples": [
        {
          "sample_id": 0,
          "true_label": 1,
          "pred_label": 1,
          "verified": true,
          "status": "verified",
          "verify_time": 7.82
        },
        ...
      ]
    }
  }
}
```

## Abstract Domains

ERAN supports multiple abstract domains with different precision/speed tradeoffs:

| Domain | Precision | Speed | Use Case |
|--------|-----------|-------|----------|
| `deeppoly` | Medium | Fast | Quick testing, large ε |
| `refinepoly` | High | Slow (~10x) | Final results, small ε |
| `neurify` | Medium | Medium | Alternative to deeppoly |

**Recommendation**: Start with `deeppoly` for quick iteration, then use `refinepoly` for final published results.

## Pooling Functions

| Function | Description | Verification Tightness |
|----------|-------------|------------------------|
| `improved_max` | Cascaded MaxPool1d (kernel=8) | Tighter bounds |
| `max` | Single MaxPool1d (kernel=1024) | Looser bounds |

**3DCertify's contribution** is the `improved_max` relaxation, which provides tighter bounds by decomposing the global max operation into cascaded smaller max operations.

## Comparing with α,β-CROWN

Your project already has α,β-CROWN verification ([`scripts/verify_abcrown_direct.py`](../scripts/verify_abcrown_direct.py)). Here's how ERAN compares:

| Aspect | ERAN | α,β-CROWN |
|--------|------|-----------|
| **Method** | Abstract interpretation | CROWN bounds + LP |
| **Precision** | Medium (DeepPoly) | High (branch-and-bound) |
| **Speed** | Fast (~10s/sample) | Slower (~60s/sample) |
| **Completeness** | Sound but incomplete | Complete (finds counterexamples) |
| **Best For** | Quick verification, large models | Precise bounds, small models |

**Recommendation**:
1. Use ERAN for initial screening and fast results
2. Use α,β-CROWN for precise verification of challenging samples
3. Compare results to understand verification tightness

## Troubleshooting

### ELINA Import Error

```
ImportError: libelina_auxiliary.so: cannot open shared object file
```

**Solution**:
```bash
source ~/.bashrc  # Reload LD_LIBRARY_PATH
sudo ldconfig     # Update library cache
```

### ONNX Compatibility Error

```
ValueError: Unsupported attribute: auto_pad
```

**Solution**: Run `python scripts/patch_3dcertify.py` to apply compatibility patches.

### Low Verification Rate

If most samples are "Unknown":

1. **Increase timeout**: Edit `timeout: 240` in config
2. **Use refinepoly**: Change `domain: refinepoly` (slower but more precise)
3. **Reduce epsilon**: Try smaller perturbations first
4. **Check model accuracy**: Low accuracy = few samples to verify

### Memory Issues

If verification crashes with OOM:

1. **Reduce batch size**: Verify one sample at a time (already default)
2. **Reduce num_samples**: Start with 10-20 samples
3. **Use smaller model**: Reduce `max_features` in architecture

## Performance Tips

1. **GPU Usage**: ERAN uses CPU only (ELINA library). GPU helps only for model inference.

2. **Parallel Verification**: To verify multiple samples in parallel:
   ```bash
   # Split data into chunks and run multiple processes
   python scripts/verify_eran_local.py --num-samples 10 &
   python scripts/verify_eran_local.py --num-samples 10 --offset 10 &
   ```

3. **Incremental Epsilon**: Start with small epsilon and increase:
   ```bash
   python scripts/verify_eran_local.py --epsilon 0.001
   python scripts/verify_eran_local.py --epsilon 0.003
   python scripts/verify_eran_local.py --epsilon 0.005
   ```

## References

- **3DCertify Paper**: [Certified Adversarial Robustness for Point Clouds](https://arxiv.org/abs/2103.16652) (ICCV 2021)
- **ERAN Paper**: [An Abstract Domain for Certifying Neural Networks](https://dl.acm.org/doi/10.1145/3290354) (POPL 2019)
- **3DCertify Repository**: https://github.com/eth-sri/3dcertify
- **ERAN Repository**: https://github.com/eth-sri/eran

## Citation

If you use this verification pipeline in your research, please cite:

```bibtex
@inproceedings{wei2021certified,
  title={Certified Adversarial Robustness for Point Clouds},
  author={Wei, Daniel and Liu, Yuanrui and Mitra, Swaroop},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={7277--7286},
  year={2021}
}
```
