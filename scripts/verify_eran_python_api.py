#!/usr/bin/env python3
"""
ERAN verification using Python API directly (not CLI).

This follows 3DCertify's approach exactly:
1. Load PyTorch model
2. Export to ONNX using 3DCertify's onnx_converter
3. Create EranVerifier with ONNX model object
4. Call analyze_classification_box() directly

This bypasses ERAN's CLI and ONNX translator, avoiding the 4D shape issue.
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, '3dcertify')
sys.path.insert(0, 'ERAN/tf_verify')

from timeit import default_timer as timer
import numpy as np
import torch
import json
from pathlib import Path

from pointnet.model import PointNet
from relaxations.interval import Interval
from util import onnx_converter
from verifier.eran_verifier import EranVerifier

print("="*70)
print("ERAN Verification via Python API")
print("="*70)
print()

# Configuration
MODEL_PATH = 'models/pointnet_3dcertify_64p.pth'
ONNX_PATH = 'models/pointnet_3dcertify_64p_api.onnx'
TEST_DATA_PATH = 'data/pointnet/test_groups.npy'
TEST_LABELS_PATH = 'data/pointnet/test_labels.npy'
RESULTS_PATH = 'results/eran_python_api_verification.json'

NUM_POINTS = 64
NUM_CLASSES = 2
N_VERIFY_SAMPLES = 20
EPSILONS = [0.001, 0.003, 0.005, 0.007, 0.01]
DOMAIN = 'deepzono'  # Fast domain for testing

print("Configuration:")
print(f"  Model: {MODEL_PATH}")
print(f"  Domain: {DOMAIN}")
print(f"  Samples: {N_VERIFY_SAMPLES}")
print(f"  Epsilons: {EPSILONS}")
print()

# Load model
print("Loading model...")
checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
print(f"  Test Accuracy: {checkpoint['test_accuracy']:.2f}%")

torch_model = PointNet(
    number_points=NUM_POINTS,
    num_classes=NUM_CLASSES,
    max_features=1024,
    pool_function='improved_max',
    disable_assertions=True,
    transposed_input=True  # 3DCertify uses transposed_input=True
)
torch_model.load_state_dict(checkpoint['model_state_dict'])
torch_model = torch_model.eval()

# Export ONNX using 3DCertify's converter
print("Exporting to ONNX...")
onnx_model = onnx_converter.convert(torch_model, NUM_POINTS, ONNX_PATH)
print(f"  ✓ ONNX exported to {ONNX_PATH}")
print()

# Initialize ERAN with Python API (not CLI!)
print("Initializing ERAN via Python API...")

# Monkey-patch to use deepzono domain
class FastEranVerifier(EranVerifier):
    def analyze_classification_box(self, bounds: Interval):
        (dominant_class, _, nlb, nub, _) = self.eran.analyze_box(
            specLB=bounds.lower_bound,
            specUB=bounds.upper_bound,
            domain=DOMAIN,  # Use fast deepzono domain
            timeout_lp=self._EranVerifier__TIMEOUT_LP,
            timeout_milp=self._EranVerifier__TIMEOUT_MILP,
            use_default_heuristic=True,
            testing=True
        )
        return dominant_class, nlb, nub

eran = FastEranVerifier(onnx_model)
print("  ✓ ERAN ready!")
print()

# Load test data
print("Loading test data...")
test_groups = np.load(TEST_DATA_PATH)
test_labels = np.load(TEST_LABELS_PATH)

def subsample_points(data, n_points):
    orig_points = data.shape[1]
    indices = np.linspace(0, orig_points-1, n_points, dtype=int)
    return data[:, indices, :3]

test_xyz = subsample_points(test_groups, NUM_POINTS)
print(f"  Test shape: {test_xyz.shape}")
print()

# Test model (non-transposed for inference)
test_model = PointNet(
    number_points=NUM_POINTS,
    num_classes=NUM_CLASSES,
    max_features=1024,
    pool_function='improved_max',
    transposed_input=False
)
test_model.load_state_dict(checkpoint['model_state_dict'])
test_model.eval()

# Select samples
np.random.seed(42)
sample_indices = []
samples_per_class = N_VERIFY_SAMPLES // NUM_CLASSES
for class_id in range(NUM_CLASSES):
    class_indices = np.where(test_labels == class_id)[0]
    selected = np.random.choice(class_indices, min(samples_per_class, len(class_indices)), replace=False)
    sample_indices.extend(selected)

print(f"Selected {len(sample_indices)} samples")
print()

# Verification
print("="*70)
print("Starting Verification")
print("="*70)
print()

all_results = {}

for eps in EPSILONS:
    print(f"ε = {eps}")
    print("-"*40)

    verified_count = 0
    total_tested = 0
    total_time = 0
    results_list = []

    for count, idx in enumerate(sample_indices):
        sample_xyz = test_xyz[idx]
        label = test_labels[idx]
        label_str = "CRITICAL" if label == 0 else "NON_CRITICAL"

        # Check prediction
        points_tensor = torch.from_numpy(sample_xyz).unsqueeze(0)
        prediction = test_model(points_tensor)
        pred_label = prediction.data.max(1)[1].item()

        if pred_label != label:
            print(f"  [{count+1:2d}/{len(sample_indices)}] Sample {idx:4d}: ⊘ SKIPPED (wrong prediction)")
            continue

        total_tested += 1

        # Verify using Python API
        print(f"  [{count+1:2d}/{len(sample_indices)}] Sample {idx:4d} ({label_str:12}): ", end="", flush=True)

        lower_bound = sample_xyz - eps
        upper_bound = sample_xyz + eps

        start = timer()
        try:
            # This is the key - use Python API directly!
            (dominant_class, nlb, nub) = eran.analyze_classification_box(
                Interval(lower_bound, upper_bound)
            )
            elapsed = timer() - start
            total_time += elapsed

            verified = (dominant_class == label)

            if verified:
                verified_count += 1
                print(f"✓ ({elapsed:.2f}s)")
            else:
                print(f"✗ (dom={dominant_class}, {elapsed:.2f}s)")

            results_list.append({
                'sample_idx': int(idx),
                'label': int(label),
                'verified': bool(verified),
                'dominant_class': int(dominant_class) if dominant_class != -1 else -1,
                'time': elapsed
            })

        except Exception as e:
            elapsed = timer() - start
            total_time += elapsed
            print(f"✗ ERROR: {str(e)[:40]} ({elapsed:.2f}s)")

            results_list.append({
                'sample_idx': int(idx),
                'label': int(label),
                'verified': False,
                'error': str(e),
                'time': elapsed
            })

    # Summary
    if total_tested > 0:
        rate = 100*verified_count/total_tested
        avg_time = total_time/total_tested
        print(f"\n  Summary: {verified_count}/{total_tested} verified ({rate:.1f}%)")
        print(f"  Avg time: {avg_time:.2f}s, Total: {total_time:.1f}s")
    else:
        print("\n  No samples tested")

    all_results[str(eps)] = {
        'epsilon': eps,
        'verified_count': verified_count,
        'total_tested': total_tested,
        'verification_rate': (100*verified_count/total_tested) if total_tested > 0 else 0,
        'avg_time': (total_time/total_tested) if total_tested > 0 else 0,
        'total_time': total_time,
        'samples': results_list
    }
    print()

# Final summary
print("="*70)
print("VERIFICATION SUMMARY")
print("="*70)
print()
print(f"{'Epsilon':>10} | {'Verified':>10} | {'Total':>8} | {'Rate %':>8} | {'Avg Time':>10}")
print("-"*60)
for eps_str, r in all_results.items():
    print(f"{float(eps_str):>10.4f} | {r['verified_count']:>10} | {r['total_tested']:>8} | "
          f"{r['verification_rate']:>7.1f}% | {r['avg_time']:>9.2f}s")

# Save results
final_results = {
    'model_path': MODEL_PATH,
    'verifier': 'ERAN',
    'domain': DOMAIN,
    'architecture': 'improved_max (cascading MaxPools)',
    'method': 'Python API (not CLI)',
    'results': all_results
}

Path(RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, 'w') as f:
    json.dump(final_results, f, indent=2)

print()
print(f"✓ Results saved to: {RESULTS_PATH}")
print()
print("SUCCESS: Used ERAN Python API directly, bypassing CLI and ONNX translator!")
