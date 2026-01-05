#!/usr/bin/env python3
"""
Complete α,β-CROWN Verification for PointNet with NSGA3 Support.

This script:
1. Loads PointNet model (trained with NSGA3 weights or standard)
2. Exports to ONNX format compatible with α,β-CROWN
3. Generates VNN-LIB specifications for verification
4. Runs α,β-CROWN via CLI
5. Parses and displays verification results

Usage:
    python scripts/verify_with_abcrown_nsga3.py [--model MODEL_PATH] [--epsilon EPSILON] [--n-samples N]
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

# Add project paths
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR / "3dcertify"))

from src.pointnet_model import PointNetForVerification
from pointnet.model import PointNet as PointNet3DCertify


class PointNetFlatInputWrapper(nn.Module):
    """
    Wrapper for PointNet that accepts flat input (batch, num_points * 3)
    and reshapes it to (batch, 3, num_points) for the actual model.

    This is needed for α,β-CROWN verification which expects flat inputs.
    """
    def __init__(self, model: nn.Module, num_points: int):
        super().__init__()
        self.model = model
        self.num_points = num_points

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_points * 3)
        batch_size = x.shape[0]
        # Reshape to (batch, num_points, 3)
        x = x.view(batch_size, self.num_points, 3)
        # Transpose to (batch, 3, num_points)
        x = x.transpose(1, 2)
        # Forward through model
        return self.model(x)


def create_vnnlib_spec(
    sample: np.ndarray,
    label: int,
    epsilon: float,
    output_path: Path,
    num_classes: int = 2
) -> None:
    """
    Generate VNN-LIB specification for robustness verification.

    Args:
        sample: Flattened input (n_points * 3,)
        label: True label (0 or 1)
        epsilon: L-infinity perturbation bound
        output_path: Where to save .vnnlib file
        num_classes: Number of output classes
    """
    n_inputs = len(sample)

    with open(output_path, 'w') as f:
        # Declare input variables
        for i in range(n_inputs):
            f.write(f"(declare-const X_{i} Real)\n")

        # Declare output variables
        for i in range(num_classes):
            f.write(f"(declare-const Y_{i} Real)\n")

        f.write("\n; Input constraints\n")

        # Input bounds: sample[i] - epsilon <= X_i <= sample[i] + epsilon
        for i in range(n_inputs):
            lower = float(sample[i] - epsilon)
            upper = float(sample[i] + epsilon)
            f.write(f"(assert (>= X_{i} {lower:.10f}))\n")
            f.write(f"(assert (<= X_{i} {upper:.10f}))\n")

        f.write("\n; Output constraints\n")

        # Robustness property: true label should have highest score
        # For verification to fail, there must exist another class with score >= true class
        # So we assert the NEGATION: exists other_label where Y_label <= Y_other_label
        # α,β-CROWN will try to prove this is unsat (i.e., property is verified)

        # Collect constraints
        constraints = []
        for other_label in range(num_classes):
            if other_label != label:
                constraints.append(f"(<= Y_{label} Y_{other_label})")

        # Write constraint(s)
        if len(constraints) == 1:
            # Single constraint: no 'or' needed
            f.write(f"(assert {constraints[0]})\n")
        else:
            # Multiple constraints: use 'or'
            f.write("(assert (or\n")
            for constraint in constraints:
                f.write(f"    {constraint}\n")
            f.write("))\n")


def export_model_to_onnx(
    model: nn.Module,
    num_points: int,
    output_path: Path,
    model_arch: str = 'PointNet3DCertify',
    simplify: bool = True
) -> Tuple[bool, str]:
    """
    Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model (can be wrapped with PointNetFlatInputWrapper)
        num_points: Number of points per cloud
        output_path: Where to save .onnx file
        model_arch: Architecture type ('PointNet3DCertify' or 'PointNetForVerification')
        simplify: Whether to simplify ONNX graph

    Returns:
        (success, input_shape_description)
    """
    model.eval()

    # All models should accept flat input (batch, num_points * 3) for α,β-CROWN
    # The wrapper handles reshaping if needed
    dummy_input = torch.randn(1, num_points * 3)
    input_shape_desc = f"(batch, {num_points * 3})"

    try:
        # Export to ONNX
        # Use opset 11 for better compatibility with onnx2pytorch
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=11,  # Opset 11 for onnx2pytorch compatibility
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        print(f"  ✓ ONNX export successful: {output_path}")
        print(f"  ✓ Input shape: {input_shape_desc}")

        # Simplify ONNX graph if requested
        if simplify:
            try:
                import onnx
                from onnxsim import simplify as onnx_simplify

                onnx_model = onnx.load(str(output_path))
                simplified_model, check = onnx_simplify(onnx_model)

                if check:
                    onnx.save(simplified_model, str(output_path))
                    print(f"  ✓ ONNX graph simplified")
                else:
                    print(f"  ⚠ ONNX simplification check failed, using original")
            except ImportError:
                print(f"  ⚠ onnx-simplifier not available, using original ONNX")

        return True, input_shape_desc

    except Exception as e:
        print(f"  ✗ ONNX export failed: {e}")
        return False, ""


def create_abcrown_config(
    onnx_path: Path,
    vnnlib_path: Path,
    output_path: Path,
    timeout: int = 300,
    device: str = "cuda"
) -> None:
    """
    Create α,β-CROWN configuration YAML file.

    Args:
        onnx_path: Path to ONNX model
        vnnlib_path: Path to VNN-LIB spec
        output_path: Where to save config
        timeout: Verification timeout in seconds
        device: 'cuda' or 'cpu'
    """
    config = {
        'general': {
            'device': device,
            'seed': 42,
            'conv_mode': 'matrix',  # Required for Conv1d support
            'deterministic': False,
            'complete_verifier': 'bab',
            'enable_incomplete_verification': True,
            'save_adv_example': False,
        },
        'model': {
            'onnx_path': str(onnx_path.absolute()),
        },
        'specification': {
            'vnnlib_path': str(vnnlib_path.absolute()),
            'type': 'bound',
        },
        'solver': {
            'batch_size': 2048,
            'bound_prop_method': 'alpha-crown',
        },
        'bab': {
            'timeout': timeout,
            'max_domains': 100000,
        },
        'attack': {
            'pgd_order': 'before',
            'pgd_steps': 100,
            'pgd_restarts': 30,
        }
    }

    # Write YAML manually to avoid pyyaml dependency
    with open(output_path, 'w') as f:
        def write_dict(d, indent=0):
            for key, value in d.items():
                prefix = "  " * indent
                if isinstance(value, dict):
                    f.write(f"{prefix}{key}:\n")
                    write_dict(value, indent + 1)
                elif isinstance(value, bool):
                    f.write(f"{prefix}{key}: {str(value).lower()}\n")
                elif isinstance(value, str):
                    f.write(f"{prefix}{key}: {value}\n")
                else:
                    f.write(f"{prefix}{key}: {value}\n")

        write_dict(config)


def run_abcrown_verification(
    config_path: Path,
    abcrown_dir: Path
) -> Tuple[bool, str, Dict]:
    """
    Run α,β-CROWN verification via CLI.

    Args:
        config_path: Path to config YAML
        abcrown_dir: Path to α,β-CROWN installation

    Returns:
        (verified, status_string, result_dict)
    """
    abcrown_script = abcrown_dir / "complete_verifier" / "abcrown.py"

    if not abcrown_script.exists():
        return False, "ERROR: α,β-CROWN script not found", {}

    cmd = [
        sys.executable,
        str(abcrown_script),
        "--config", str(config_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes max
            cwd=str(abcrown_dir / "complete_verifier")
        )

        output = result.stdout + result.stderr

        # Parse result
        verified = False
        status = "unknown"

        # α,β-CROWN status strings
        if "unsat" in output.lower():
            verified = True
            status = "verified (unsat)"
        elif "verified" in output.lower() or "safe" in output.lower():
            verified = True
            status = "verified (safe)"
        elif "sat" in output.lower() or "unsafe" in output.lower():
            verified = False
            status = "falsified (counterexample found)"
        elif "timeout" in output.lower():
            status = "timeout"
        elif "unknown" in output.lower():
            status = "unknown"

        result_dict = {
            'verified': verified,
            'status': status,
            'output': output[-500:] if len(output) > 500 else output  # Last 500 chars
        }

        return verified, status, result_dict

    except subprocess.TimeoutExpired:
        return False, "timeout", {'error': 'Verification timeout'}
    except Exception as e:
        return False, f"error: {str(e)}", {'error': str(e)}


def verify_samples(
    model_path: Path,
    data_path: Path,
    labels_path: Path,
    epsilons: List[float],
    n_samples: int,
    output_dir: Path,
    device: str = "cuda"
) -> Dict:
    """
    Main verification loop.

    Args:
        model_path: Path to trained .pth model
        data_path: Path to test data .npy
        labels_path: Path to labels .npy
        epsilons: List of epsilon values to test
        n_samples: Number of samples to verify
        output_dir: Where to save results
        device: 'cuda' or 'cpu'

    Returns:
        Dictionary with all results
    """
    print("="*70)
    print("α,β-CROWN Verification with NSGA3 Support")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load model
    print("Loading model...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)

    # Determine model architecture from checkpoint
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    architecture = checkpoint.get('architecture', 'unknown')

    # Number of points (default 64)
    num_points = 64

    print(f"  Model: {model_path.name}")
    print(f"  Architecture: {architecture}")
    print(f"  Test Accuracy: {checkpoint.get('test_accuracy', 'N/A')}")
    print()

    # Load appropriate architecture
    if 'PointNet_max_single_maxpool' in architecture or 'mlp1.0.weight' in state_dict:
        # 3DCertify PointNet architecture
        print("  Using 3DCertify PointNet architecture")
        base_model = PointNet3DCertify(
            number_points=num_points,
            num_classes=2,
            max_features=1024,
            pool_function='max',  # Single MaxPool for α,β-CROWN compatibility
            disable_assertions=True,
            transposed_input=True  # For ONNX export: (batch, 3, num_points)
        )
        base_model.load_state_dict(state_dict)
        base_model.eval()

        # Wrap with flat input wrapper for α,β-CROWN
        model = PointNetFlatInputWrapper(base_model, num_points)
        model.eval()

        # Keep base_model for direct testing
        test_model = base_model

        model_arch = 'PointNet3DCertify'
        uses_transposed_input = True
    else:
        # Original PointNetForVerification
        use_tnet = any('tnet' in key or 'input_tnet' in key for key in state_dict.keys())
        print(f"  Using PointNetForVerification (T-Net: {use_tnet})")
        model = PointNetForVerification(
            num_points=num_points,
            num_classes=2,
            use_tnet=use_tnet,
            feature_transform=True,
            in_channels=3
        )
        model.load_state_dict(state_dict)
        model.eval()

        test_model = model
        model_arch = 'PointNetForVerification'
        uses_transposed_input = False

    # Create output directories
    onnx_dir = output_dir / "onnx"
    specs_dir = output_dir / "specs"
    configs_dir = output_dir / "configs"

    for d in [onnx_dir, specs_dir, configs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Export ONNX once
    onnx_path = onnx_dir / f"{model_path.stem}.onnx"
    print("Exporting ONNX...")
    success, input_shape_desc = export_model_to_onnx(model, num_points, onnx_path, model_arch, simplify=False)
    if not success:
        return {'error': 'ONNX export failed'}
    print()

    # Load test data
    print("Loading test data...")
    test_data = np.load(data_path)
    test_labels = np.load(labels_path)

    # Subsample to num_points if needed
    if test_data.shape[1] != num_points:
        print(f"  Subsampling from {test_data.shape[1]} to {num_points} points...")
        orig_points = test_data.shape[1]
        indices = np.linspace(0, orig_points-1, num_points, dtype=int)
        test_data = test_data[:, indices, :3]  # Take xyz only
    else:
        test_data = test_data[:, :, :3]

    print(f"  Test data shape: {test_data.shape}")
    print(f"  Labels shape: {test_labels.shape}")
    print(f"  Samples to verify: {n_samples}")
    print()

    # Select samples
    np.random.seed(42)
    sample_indices = np.random.choice(len(test_data), min(n_samples, len(test_data)), replace=False)

    # Verification loop
    all_results = {}
    abcrown_dir = BASE_DIR / "alpha-beta-CROWN"

    for eps in epsilons:
        print(f"{'='*70}")
        print(f"Epsilon = {eps} (~{eps*150:.1f} cm in MOLA coordinates)")
        print(f"{'='*70}")
        print()

        verified_count = 0
        falsified_count = 0
        timeout_count = 0
        error_count = 0
        sample_results = []

        for idx, sample_idx in enumerate(sample_indices):
            sample = test_data[sample_idx]  # (num_points, 3)
            label = int(test_labels[sample_idx])
            label_name = "CRITICAL" if label == 0 else "NON_CRITICAL"

            # Flatten sample for VNN-LIB
            sample_flat = sample.flatten()  # (num_points * 3,)

            # Check if model predicts correctly
            with torch.no_grad():
                if uses_transposed_input:
                    # Use base model for testing (expects transposed input)
                    sample_tensor = torch.from_numpy(sample).T.unsqueeze(0).float()  # (1, 3, 64)
                    pred = test_model(sample_tensor)
                else:
                    # Flatten to (batch, num_points * 3)
                    sample_tensor = torch.from_numpy(sample_flat).unsqueeze(0).float()
                    pred = test_model(sample_tensor)
                pred_label = pred.argmax(1).item()

            if pred_label != label:
                print(f"  [{idx+1}/{len(sample_indices)}] Sample {sample_idx:3d} ({label_name:12}): ⊘ SKIPPED (misclassified)")
                continue

            # Generate VNN-LIB spec
            spec_path = specs_dir / f"sample_{sample_idx}_eps{eps}.vnnlib"
            create_vnnlib_spec(sample_flat, label, eps, spec_path, num_classes=2)

            # Generate config
            config_path = configs_dir / f"config_{sample_idx}_eps{eps}.yaml"
            create_abcrown_config(
                onnx_path,
                spec_path,
                config_path,
                timeout=300,
                device=device
            )

            # Run verification
            print(f"  [{idx+1}/{len(sample_indices)}] Sample {sample_idx:3d} ({label_name:12}): ", end="", flush=True)

            verified, status, result_dict = run_abcrown_verification(config_path, abcrown_dir)

            if "verified" in status:
                verified_count += 1
                print(f"✓ VERIFIED")
            elif "falsified" in status or "unsafe" in status:
                falsified_count += 1
                print(f"✗ FALSIFIED")
            elif "timeout" in status:
                timeout_count += 1
                print(f"⏱ TIMEOUT")
            else:
                error_count += 1
                print(f"⚠ ERROR: {status}")

            sample_results.append({
                'sample_idx': int(sample_idx),
                'label': label,
                'verified': verified,
                'status': status,
                'result': result_dict
            })

        # Epsilon summary
        total_tested = verified_count + falsified_count + timeout_count + error_count
        print()
        print(f"Summary for ε={eps}:")
        print(f"  Verified:   {verified_count}/{total_tested} ({100*verified_count/total_tested if total_tested > 0 else 0:.1f}%)")
        print(f"  Falsified:  {falsified_count}/{total_tested}")
        print(f"  Timeout:    {timeout_count}/{total_tested}")
        print(f"  Errors:     {error_count}/{total_tested}")
        print()

        all_results[str(eps)] = {
            'epsilon': eps,
            'verified_count': verified_count,
            'falsified_count': falsified_count,
            'timeout_count': timeout_count,
            'error_count': error_count,
            'total_tested': total_tested,
            'verification_rate': 100*verified_count/total_tested if total_tested > 0 else 0,
            'samples': sample_results
        }

    # Final summary
    print("="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"{'Epsilon':>10} | {'Verified':>10} | {'Total':>8} | {'Rate %':>8}")
    print("-"*50)
    for eps_str, r in all_results.items():
        print(f"{float(eps_str):>10.4f} | {r['verified_count']:>10} | "
              f"{r['total_tested']:>8} | {r['verification_rate']:>7.1f}%")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Verify PointNet with α,β-CROWN")
    parser.add_argument(
        '--model',
        type=Path,
        default=BASE_DIR / 'models' / 'pointnet_max_64p.pth',
        help='Path to trained model'
    )
    parser.add_argument(
        '--data',
        type=Path,
        default=BASE_DIR / 'data' / 'pointnet' / 'test_groups.npy',
        help='Path to test data'
    )
    parser.add_argument(
        '--labels',
        type=Path,
        default=BASE_DIR / 'data' / 'pointnet' / 'test_labels.npy',
        help='Path to test labels'
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        nargs='+',
        default=[0.005, 0.01, 0.02],
        help='Epsilon values to test'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=10,
        help='Number of samples to verify'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=BASE_DIR / 'results' / 'abcrown_verification',
        help='Output directory'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )

    args = parser.parse_args()

    # Run verification
    results = verify_samples(
        model_path=args.model,
        data_path=args.data,
        labels_path=args.labels,
        epsilons=args.epsilon,
        n_samples=args.n_samples,
        output_dir=args.output,
        device=args.device
    )

    # Save results
    results_json = args.output / 'verification_results.json'
    with open(results_json, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'model': str(args.model),
            'method': 'alpha-beta-CROWN',
            'results': results
        }, f, indent=2)

    print(f"\n✓ Results saved to: {results_json}")


if __name__ == "__main__":
    main()
