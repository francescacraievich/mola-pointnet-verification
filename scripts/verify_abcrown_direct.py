#!/usr/bin/env python3
"""Direct alpha-beta-CROWN verification using Python API (no ONNX needed).

This script verifies PointNet robustness using the α,β-CROWN Python API directly,
bypassing ONNX conversion which has issues with max-pooling operations.
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "alpha-beta-CROWN", "complete_verifier"))

import torch
import numpy as np
import json
from datetime import datetime
from src.pointnet_model import PointNetForVerification

# Import abcrown API
from api import ABCrownSolver, ConfigBuilder, VerificationSpec, input_vars, output_vars


def load_model(pooling="max", use_tnet=True):
    """Load trained PointNet model.

    Args:
        pooling: "max" or "mean" pooling
        use_tnet: If False, disable T-Net to reduce memory usage during verification.
                  The T-Net introduces matrix multiplication (BoundMatMul) which is
                  memory-intensive for formal verification.
    """
    model = PointNetForVerification(
        num_points=64,
        num_classes=2,
        use_tnet=use_tnet,
        pooling=pooling
    )

    if pooling == "max":
        checkpoint = torch.load(f'{BASE_DIR}/models/pointnet.pth', map_location='cpu', weights_only=True)
    else:
        checkpoint = torch.load(f'{BASE_DIR}/models/pointnet_mean_pooling.pth', map_location='cpu', weights_only=True)

    # Load weights, handling T-Net mismatch if use_tnet=False
    if use_tnet:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Filter out T-Net weights
        state_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                      if not k.startswith('tnet')}
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model


def verify_sample(model, sample: np.ndarray, label: int, epsilon: float, timeout: int = 60):
    """Verify robustness of a single sample using alpha-beta-CROWN."""

    # Handle different input shapes
    if sample.shape == (3, 64):
        sample_flat = sample.T.flatten()  # (192,)
    else:
        sample_flat = sample.flatten()  # Already (64, 3) -> (192,)

    n_inputs = len(sample_flat)
    n_outputs = 2

    # Create input/output variables
    x = input_vars(n_inputs)
    y = output_vars(n_outputs)

    # Input constraints: sample +/- epsilon (vectorized API)
    input_constraint = (x >= sample_flat - epsilon) & (x <= sample_flat + epsilon)

    # Output constraint: true label should be greater than other label
    other_label = 1 - label
    output_constraint = y[label] > y[other_label]

    # Build verification spec
    spec = VerificationSpec.build_spec(
        input_vars=x,
        output_vars=y,
        input_constraint=input_constraint,
        output_constraint=output_constraint,
    )

    # Configure solver using correct API (.set() method)
    cfg = ConfigBuilder.from_defaults().set(
        bab__timeout=timeout,
        general__enable_incomplete_verification=True,
        general__complete_verifier="bab",
        general__conv_mode="matrix",  # Required for Conv1d support
        attack__pgd_order="before",
        attack__pgd_steps=100,
        attack__pgd_restarts=30,
    )

    # Create solver with model
    solver = ABCrownSolver(spec, model, config=cfg)

    # Solve
    result = solver.solve()

    # safe-incomplete means verified by incomplete verifier (α-CROWN)
    # safe means verified by complete verifier (β-CROWN + BaB)
    # unsat means the property is verified
    verified = result.status in ["safe", "safe-incomplete", "verified", "unsat"]

    return {
        "status": result.status,
        "success": result.success,
        "verified": verified
    }


def main():
    print("=" * 70)
    print("α,β-CROWN Direct Verification for PointNet (MAX-POOLING)")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Test with max-pooling (the original model)
    # Note: T-Net uses matrix multiplication (BoundMatMul) which is very memory-intensive
    # For larger epsilons, we may need to disable T-Net to avoid OOM
    pooling = "max"
    use_tnet = True  # Set to False to save GPU memory for larger epsilons

    print(f"\nModel: PointNet with {pooling.upper()}-pooling")
    print(f"T-Net: {'Enabled' if use_tnet else 'Disabled (for memory optimization)'}")
    print("Method: α,β-CROWN Python API (no ONNX conversion)")

    model = load_model(pooling, use_tnet=use_tnet)

    # Load test data
    X_test = np.load(f'{BASE_DIR}/data/pointnet/test_groups.npy')
    y_test = np.load(f'{BASE_DIR}/data/pointnet/test_labels.npy')

    # Test multiple epsilon values up to ~6cm
    # 0.04 * 150 = 6 cm
    epsilons = [0.005, 0.01, 0.02, 0.04]
    n_samples = 10

    all_results = {}

    for epsilon in epsilons:
        cm = epsilon * 150  # Approximate conversion to cm
        print(f"\n{'='*70}")
        print(f"Epsilon: {epsilon} (~{cm:.1f} cm)")
        print("-" * 70)
        print(f"{'Sample':^8} | {'Label':^6} | {'Status':^20}")
        print("-" * 70)

        verified_count = 0
        unsafe_count = 0
        error_count = 0
        sample_results = []

        for i in range(n_samples):
            sample = X_test[i]
            label = int(y_test[i])

            try:
                result = verify_sample(model, sample, label, epsilon, timeout=60)
                status = result["status"]

                if result["verified"]:
                    verified_count += 1
                    display_status = "VERIFIED"
                else:
                    unsafe_count += 1
                    display_status = f"UNSAFE ({status})"

                sample_results.append({
                    "sample": i,
                    "label": label,
                    "status": status,
                    "verified": result["verified"]
                })

            except Exception as e:
                error_count += 1
                display_status = f"ERROR: {str(e)[:30]}"
                sample_results.append({
                    "sample": i,
                    "label": label,
                    "status": "error",
                    "error": str(e)
                })

            print(f"{i:^8} | {label:^6} | {display_status:<20}")

        print("-" * 70)
        print(f"Summary: Verified={verified_count}/{n_samples} ({100*verified_count/n_samples:.1f}%), "
              f"Unsafe={unsafe_count}/{n_samples}, Errors={error_count}/{n_samples}")

        all_results[str(epsilon)] = {
            "epsilon": epsilon,
            "cm": cm,
            "verified": verified_count,
            "unsafe": unsafe_count,
            "errors": error_count,
            "verified_pct": 100 * verified_count / n_samples,
            "samples": sample_results
        }

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - PointNet MAX-POOLING Verification")
    print("=" * 70)
    print(f"{'Epsilon':>8} | {'~cm':>6} | {'Verified':>10} | {'Unsafe':>10} | {'Errors':>10}")
    print("-" * 55)
    for eps_str, r in all_results.items():
        print(f"{float(eps_str):>8.3f} | {r['cm']:>5.1f}  | {r['verified_pct']:>8.1f}%  | "
              f"{100*r['unsafe']/n_samples:>8.1f}%  | {100*r['errors']/n_samples:>8.1f}%")

    # Save results
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": "PointNet (Max-Pooling)",
        "method": "α,β-CROWN Python API",
        "n_samples": n_samples,
        "results": all_results
    }

    output_path = f'{BASE_DIR}/results/abcrown_maxpool_verification.json'
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
