#!/usr/bin/env python3
"""
ERAN verification using Python API directly (not CLI).

This follows 3DCertify's approach exactly:
1. Load PyTorch model
2. Export to ONNX using 3DCertify's onnx_converter
3. Create EranVerifier with ONNX model object
4. Call analyze_classification_box() directly
5. Verify on a selection of correctly classified test samples
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent  # mola-pointnet-verification/
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "3dcertify"))
sys.path.insert(0, str(BASE_DIR / "ERAN/tf_verify"))

import json
from datetime import datetime
from timeit import default_timer as timer

import numpy as np
import torch
from pointnet.model import PointNet
from relaxations.interval import Interval
from util import onnx_converter
from verifier.eran_verifier import EranVerifier


def get_next_result_number(results_dir: Path, prefix: str) -> int:
    """Find the next available result number."""
    existing = list(results_dir.glob(f"{prefix}_*.json"))
    if not existing:
        return 1
    numbers = []
    for f in existing:
        try:
            num = int(f.stem.split("_")[-1])
            numbers.append(num)
        except ValueError:
            pass
    return max(numbers) + 1 if numbers else 1


print("=" * 70)
print("ERAN Verification via Python API")
print("=" * 70)
print()

# Configuration - Small model (64 points, 1024 features) for faster verification
MODEL_PATH = BASE_DIR / "saved_models/pointnet_3dcertify_64p.pth"
ONNX_PATH = BASE_DIR / "saved_models/pointnet_3dcertify_64p_api.onnx"
TEST_DATA_PATH = BASE_DIR / "data/pointnet/test_groups.npy"
TEST_LABELS_PATH = BASE_DIR / "data/pointnet/test_labels.npy"

NUM_POINTS = 64
NUM_CLASSES = 2
MAX_FEATURES = 1024
N_VERIFY_SAMPLES = 10
EPSILONS = [0.001, 0.003, 0.005, 0.007, 0.01, 0.02]
DOMAIN = "deepzono"  # Fast and stable

print("Configuration:")
print(f"  Model: {MODEL_PATH}")
print(f"  Domain: {DOMAIN}")
print(f"  Samples: {N_VERIFY_SAMPLES}")
print(f"  Epsilons: {EPSILONS}")
print()

# Load model
print("Loading model...")
checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
test_accuracy = checkpoint.get("test_accuracy", "N/A")
print(f"  Test Accuracy: {test_accuracy}%")

torch_model = PointNet(
    number_points=NUM_POINTS,
    num_classes=NUM_CLASSES,
    max_features=MAX_FEATURES,
    pool_function="improved_max",
    disable_assertions=True,
    transposed_input=True,  
)
torch_model.load_state_dict(checkpoint["model_state_dict"])
torch_model = torch_model.eval()

# Export ONNX using 3DCertify's converter
print("Exporting to ONNX...")
onnx_model = onnx_converter.convert(torch_model, NUM_POINTS, ONNX_PATH)
print(f"  ONNX exported to {ONNX_PATH}")
print()

# Initialize ERAN with Python API 
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
            testing=True,
        )
        return dominant_class, nlb, nub


eran = FastEranVerifier(onnx_model)
print("  ERAN ready!")
print()

# Load test data
print("Loading test data...")
test_groups = np.load(TEST_DATA_PATH)
test_labels = np.load(TEST_LABELS_PATH)


def subsample_points(data, n_points):
    orig_points = data.shape[1]
    indices = np.linspace(0, orig_points - 1, n_points, dtype=int)
    return data[:, indices, :3]


test_xyz = subsample_points(test_groups, NUM_POINTS)
print(f"  Test shape: {test_xyz.shape}")
print()

# Test model (non-transposed for inference)
test_model = PointNet(
    number_points=NUM_POINTS,
    num_classes=NUM_CLASSES,
    max_features=MAX_FEATURES,
    pool_function="improved_max",
    transposed_input=False,
)
test_model.load_state_dict(checkpoint["model_state_dict"])
test_model.eval()

# Pre-filter correctly classified samples (like alpha-beta-CROWN)
print("Pre-filtering correctly classified samples...")
correct_samples = []
for idx in range(len(test_xyz)):
    sample_xyz = test_xyz[idx]
    label = test_labels[idx]
    points_tensor = torch.from_numpy(sample_xyz).unsqueeze(0)
    with torch.no_grad():
        prediction = test_model(points_tensor)
    pred_label = prediction.data.max(1)[1].item()
    if pred_label == label:
        correct_samples.append(idx)

print(
    f"  Correctly classified: {len(correct_samples)}/{len(test_xyz)} ({100*len(correct_samples)/len(test_xyz):.1f}%)"
)

# Select N_VERIFY_SAMPLES from correctly classified (random, no class balancing)
np.random.seed(42)
sample_indices = np.random.choice(
    correct_samples, min(N_VERIFY_SAMPLES, len(correct_samples)), replace=False
).tolist()

# Count per class
n_crit = sum(1 for idx in sample_indices if test_labels[idx] == 0)
n_non_crit = len(sample_indices) - n_crit
print(f"Selected {len(sample_indices)} correctly classified samples:")
print(f"  - CRITICAL: {n_crit}")
print(f"  - NON_CRITICAL: {n_non_crit}")
print()

# Verification
print("=" * 70)
print("Starting Verification")
print("=" * 70)
print()

all_results = {}
total_start = timer()

for eps in EPSILONS:
    print(f"epsilon = {eps}")
    print("-" * 40)

    verified_count = 0
    total_tested = 0
    total_time = 0
    results_list = []

    for count, idx in enumerate(sample_indices):
        sample_xyz = test_xyz[idx]
        label = test_labels[idx]
        label_str = "CRITICAL" if label == 0 else "NON_CRITICAL"

        total_tested += 1

        # Verify using Python API (all samples are pre-filtered as correctly classified)
        print(
            f"  [{count+1:2d}/{len(sample_indices)}] Sample {idx:4d} ({label_str:12}): ",
            end="",
            flush=True,
        )

        lower_bound = sample_xyz - eps
        upper_bound = sample_xyz + eps

        start = timer()
        try:
            # Use Python API
            (dominant_class, nlb, nub) = eran.analyze_classification_box(
                Interval(lower_bound, upper_bound)
            )
            elapsed = timer() - start
            total_time += elapsed

            verified = dominant_class == label

            if verified:
                verified_count += 1
                print(f"verified ({elapsed:.2f}s)")
            else:
                print(f"not verified (dom={dominant_class}, {elapsed:.2f}s)")

            results_list.append(
                {
                    "sample_idx": int(idx),
                    "label": int(label),
                    "verified": bool(verified),
                    "dominant_class": int(dominant_class) if dominant_class != -1 else -1,
                    "time": round(elapsed, 2),
                }
            )

        except Exception as e:
            elapsed = timer() - start
            total_time += elapsed
            print(f"ERROR: {str(e)[:40]} ({elapsed:.2f}s)")

            results_list.append(
                {
                    "sample_idx": int(idx),
                    "label": int(label),
                    "verified": False,
                    "error": str(e),
                    "time": round(elapsed, 2),
                }
            )

    # Summary
    if total_tested > 0:
        rate = 100 * verified_count / total_tested
        avg_time = total_time / total_tested
        print(f"\n  Summary: {verified_count}/{total_tested} verified ({rate:.1f}%)")
        print(f"  Avg time: {avg_time:.2f}s, Total: {total_time:.1f}s")
    else:
        rate = 0
        avg_time = 0
        print("\n  No samples tested")

    all_results[str(eps)] = {
        "epsilon": eps,
        "verified_count": verified_count,
        "total_tested": total_tested,
        "verification_rate": round(rate, 1),
        "avg_time": round(avg_time, 2),
        "total_time": round(total_time, 2),
        "samples": results_list,
    }
    print()

total_elapsed = timer() - total_start

# Final summary
print("=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)
print()
print(f"{'Epsilon':>10} | {'Verified':>10} | {'Total':>8} | {'Rate %':>8} | {'Avg Time':>10}")
print("-" * 60)
for eps_str, r in all_results.items():
    print(
        f"{float(eps_str):>10.4f} | {r['verified_count']:>10} | {r['total_tested']:>8} | "
        f"{r['verification_rate']:>7.1f}% | {r['avg_time']:>9.2f}s"
    )

print(f"\nTotal time: {total_elapsed:.1f}s")

# Save results with incremental numbering
results_dir = BASE_DIR / "results"
results_dir.mkdir(exist_ok=True)

result_num = get_next_result_number(results_dir, "eran_verification")
json_path = results_dir / f"eran_verification_{result_num}.json"
md_path = results_dir / f"eran_verification_{result_num}.md"

final_results = {
    "metadata": {
        "verifier": "ERAN",
        "domain": DOMAIN,
        "model": str(MODEL_PATH.name),
        "model_accuracy": test_accuracy,
        "timestamp": datetime.now().isoformat(),
        "total_time_seconds": round(total_elapsed, 2),
    },
    "summary": {
        eps: {
            "verified": r["verified_count"],
            "total": r["total_tested"],
            "rate_percent": r["verification_rate"],
        }
        for eps, r in all_results.items()
    },
    "details": all_results,
}

with open(json_path, "w") as f:
    json.dump(final_results, f, indent=2)

# Save markdown table
with open(md_path, "w") as f:
    f.write(f"# ERAN Verification Results #{result_num}\n\n")
    f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"**Model**: {MODEL_PATH.name} (accuracy: {test_accuracy}%)\n\n")
    f.write(f"**Domain**: {DOMAIN}\n\n")
    f.write(f"**Total time**: {total_elapsed:.1f}s\n\n")
    f.write("## Summary\n\n")
    f.write("| Epsilon | Verified | Total | Rate |\n")
    f.write("|---------|----------|-------|------|\n")
    for eps_str, r in all_results.items():
        f.write(
            f"| {float(eps_str)} | {r['verified_count']} | {r['total_tested']} | {r['verification_rate']}% |\n"
        )

print()
print(f"Results saved to: {json_path}")
print(f"Markdown saved to: {md_path}")
