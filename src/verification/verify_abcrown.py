#!/usr/bin/env python3
"""
Verification script for PointNet using α,β-CROWN.

Usage:
    python src/verification/verify_abcrown.py --samples 521 792 755 --epsilon 0.01 0.02 0.03 0.05
    python src/verification/verify_abcrown.py --n-samples 10 --epsilon 0.01 0.03
    python src/verification/verify_abcrown.py --by-margin low --n-samples 5 --epsilon 0.01
"""

import sys
import argparse
import subprocess
import time
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src" / "model"))

from pointnet_autolirpa_compatible import PointNetAutoLiRPA


def get_next_result_number(results_dir: Path, prefix: str) -> int:
    """Find the next available result number."""
    existing = list(results_dir.glob(f"{prefix}_*.json"))
    if not existing:
        return 1
    numbers = []
    for f in existing:
        try:
            num = int(f.stem.split('_')[-1])
            numbers.append(num)
        except ValueError:
            pass
    return max(numbers) + 1 if numbers else 1


def load_model():
    """Load the PointNet model."""
    checkpoint = torch.load(
        BASE_DIR / 'saved_models/pointnet_autolirpa_512.pth',
        map_location='cpu',
        weights_only=True
    )
    model = PointNetAutoLiRPA(64, 2, 512, use_batchnorm=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint.get('test_accuracy', 'N/A')


def get_sample_info(model, test_data, test_labels):
    """Get prediction and margin for all samples."""
    info = []
    for i in range(len(test_data)):
        sample = test_data[i, :, :3]
        x = torch.from_numpy(sample).unsqueeze(0).float()
        with torch.no_grad():
            output = model(x)
        margin = abs(output[0, 0] - output[0, 1]).item()
        pred = output.argmax(1).item()
        correct = (pred == test_labels[i])
        info.append({
            'idx': i,
            'margin': margin,
            'pred': pred,
            'label': int(test_labels[i]),
            'correct': correct
        })
    return info


def select_samples(args, sample_info):
    """Select samples based on arguments."""
    if args.samples:
        return args.samples

    # Filter correctly classified samples only
    correct_samples = [s for s in sample_info if s['correct']]

    if args.by_margin:
        # Sort by margin
        correct_samples.sort(key=lambda x: x['margin'])
        if args.by_margin == 'low':
            selected = correct_samples[:args.n_samples]
        elif args.by_margin == 'high':
            selected = correct_samples[-args.n_samples:]
        elif args.by_margin == 'mixed':
            n = args.n_samples
            low = correct_samples[:n//3]
            high = correct_samples[-(n//3):]
            mid_start = len(correct_samples)//2 - n//6
            mid = correct_samples[mid_start:mid_start + n//3]
            selected = low + mid + high
        return [s['idx'] for s in selected]

    # Random selection
    np.random.seed(args.seed)
    indices = np.random.choice(len(correct_samples), min(args.n_samples, len(correct_samples)), replace=False)
    return [correct_samples[i]['idx'] for i in indices]


def generate_vnnlib(sample, pred_label, epsilon, output_path):
    """Generate VNNLIB specification."""
    sample_flat = sample.flatten()
    n_inputs = len(sample_flat)

    with open(output_path, 'w') as f:
        f.write(f"; VNNLIB spec - eps={epsilon}, pred={pred_label}\n\n")

        for i in range(n_inputs):
            f.write(f"(declare-const X_{i} Real)\n")

        f.write("\n(declare-const Y_0 Real)\n")
        f.write("(declare-const Y_1 Real)\n\n")

        for i in range(n_inputs):
            lb = sample_flat[i] - epsilon
            ub = sample_flat[i] + epsilon
            f.write(f"(assert (>= X_{i} {lb:.10f}))\n")
            f.write(f"(assert (<= X_{i} {ub:.10f}))\n")

        # Adversary condition: other class >= predicted class
        other_label = 1 - pred_label
        f.write(f"\n(assert (>= Y_{other_label} Y_{pred_label}))\n")


def run_verification(vnnlib_path, config_path, timeout=300):
    """Run α,β-CROWN verification."""
    complete_verifier_dir = BASE_DIR / "alpha-beta-CROWN/complete_verifier"

    cmd = [
        sys.executable,
        str(complete_verifier_dir / "abcrown.py"),
        "--config", str(config_path),
        "--vnnlib_path", str(vnnlib_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(complete_verifier_dir),
            capture_output=True,
            text=True,
            timeout=timeout + 30
        )

        stdout = result.stdout.lower()

        if "result: sat" in stdout:
            return "SAT", "unsafe"
        elif "result: unsat" in stdout:
            return "UNSAT", "verified"
        elif "timeout" in stdout:
            return "TIMEOUT", "unknown"
        else:
            return "UNKNOWN", "unknown"

    except subprocess.TimeoutExpired:
        return "TIMEOUT", "timeout"
    except Exception as e:
        return "ERROR", str(e)[:20]


def main():
    parser = argparse.ArgumentParser(description='PointNet verification with α,β-CROWN')
    parser.add_argument('--samples', type=int, nargs='+', help='Specific sample indices')
    parser.add_argument('--n-samples', type=int, default=10, help='Number of samples')
    parser.add_argument('--by-margin', choices=['low', 'high', 'mixed'], help='Select by confidence margin')
    parser.add_argument('--epsilon', type=float, nargs='+', default=[0.01, 0.03, 0.05])
    parser.add_argument('--config', type=str, default='configs/abcrown_pointnet_complete.yaml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--timeout', type=int, default=300)
    args = parser.parse_args()

    # Load data
    test_data = np.load(BASE_DIR / 'data/pointnet/test_groups.npy')
    test_labels = np.load(BASE_DIR / 'data/pointnet/test_labels.npy')

    # Load model
    model, test_accuracy = load_model()

    # Get sample info
    print("Analyzing samples...")
    sample_info = get_sample_info(model, test_data, test_labels)
    sample_info_dict = {s['idx']: s for s in sample_info}

    # Select samples
    selected_samples = select_samples(args, sample_info)

    # Print header
    print("\n" + "=" * 80)
    print("α,β-CROWN VERIFICATION FOR POINTNET")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: pointnet_autolirpa_512.pth (accuracy: {test_accuracy}%)")
    print(f"Samples: {len(selected_samples)} | Epsilon values: {args.epsilon}")
    print(f"Timeout: {args.timeout}s per sample")
    print()

    # Results table header
    eps_headers = [f"ε={e}" for e in args.epsilon]
    header = f"{'Sample':>6} | {'Margin':>7} | {'Label':>8} | " + " | ".join(f"{h:>10}" for h in eps_headers)
    print(header)
    print("-" * len(header))

    # Specs directory
    specs_dir = BASE_DIR / "specs"
    specs_dir.mkdir(exist_ok=True)
    config_path = BASE_DIR / args.config

    # Results storage
    sample_details = []
    summary = {eps: {'verified': 0, 'unsafe': 0, 'timeout': 0} for eps in args.epsilon}

    total_start_time = time.time()

    for sample_idx in selected_samples:
        info = sample_info_dict[sample_idx]
        label_str = "CRIT" if info['label'] == 0 else "NON_CRIT"

        row = f"{sample_idx:>6} | {info['margin']:>7.4f} | {label_str:>8} |"
        results_row = []

        sample_result = {
            'sample_idx': sample_idx,
            'margin': round(info['margin'], 4),
            'label': info['label'],
            'label_str': label_str,
            'results': {}
        }

        for eps in args.epsilon:
            # Generate VNNLIB
            vnnlib_path = specs_dir / f"spec_sample{sample_idx}_eps{eps}.vnnlib"
            sample = test_data[sample_idx, :, :3]
            generate_vnnlib(sample, info['pred'], eps, vnnlib_path)

            # Run verification
            start = time.time()
            result, status = run_verification(vnnlib_path, config_path, args.timeout)
            elapsed = time.time() - start

            # Format result
            if result == "UNSAT":
                cell = f"✓ {elapsed:.1f}s"
                summary[eps]['verified'] += 1
                verified = True
            elif result == "SAT":
                cell = f"✗ {elapsed:.1f}s"
                summary[eps]['unsafe'] += 1
                verified = False
            else:
                cell = f"? {result[:6]}"
                summary[eps]['timeout'] += 1
                verified = None

            results_row.append(f"{cell:>10}")

            # Store result
            sample_result['results'][str(eps)] = {
                'result': result,
                'verified': verified,
                'time': round(elapsed, 2)
            }

        sample_details.append(sample_result)
        row += " | ".join(results_row)
        print(row)

    total_time = time.time() - total_start_time

    # Print summary
    print("-" * len(header))
    print("\nSUMMARY:")
    print(f"{'Epsilon':>10} | {'Verified':>10} | {'Unsafe':>10} | {'Timeout':>10} | {'Rate':>10}")
    print("-" * 60)

    summary_data = {}
    for eps in args.epsilon:
        s = summary[eps]
        total = s['verified'] + s['unsafe']
        rate = s['verified'] / total * 100 if total > 0 else 0
        print(f"{eps:>10.3f} | {s['verified']:>10} | {s['unsafe']:>10} | {s['timeout']:>10} | {rate:>9.1f}%")
        summary_data[str(eps)] = {
            'epsilon': eps,
            'verified': s['verified'],
            'unsafe': s['unsafe'],
            'timeout': s['timeout'],
            'total': total,
            'rate_percent': round(rate, 1)
        }

    print("\nLegend: ✓ = verified (unsat), ✗ = unsafe (sat), ? = timeout/unknown")
    print(f"Total time: {total_time:.1f}s")

    # Save results
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    result_num = get_next_result_number(results_dir, "abcrown_verification")
    result_path = results_dir / f"abcrown_verification_{result_num}.json"

    final_results = {
        'metadata': {
            'verifier': 'α,β-CROWN',
            'model': 'pointnet_autolirpa_512.pth',
            'model_accuracy': test_accuracy,
            'config': args.config,
            'timestamp': datetime.now().isoformat(),
            'total_time_seconds': round(total_time, 2),
            'timeout_per_sample': args.timeout
        },
        'parameters': {
            'samples': selected_samples,
            'epsilons': args.epsilon,
            'selection_method': args.by_margin if args.by_margin else ('manual' if args.samples else 'random')
        },
        'summary': summary_data,
        'details': sample_details
    }

    with open(result_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\nResults saved to: {result_path}")

    # Also save a human-readable markdown summary
    md_path = results_dir / f"abcrown_verification_{result_num}.md"
    with open(md_path, 'w') as f:
        f.write(f"# α,β-CROWN Verification Results #{result_num}\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Model**: pointnet_autolirpa_512.pth (accuracy: {test_accuracy}%)\n\n")
        f.write(f"**Samples**: {len(selected_samples)}\n\n")
        f.write(f"**Total time**: {total_time:.1f}s\n\n")
        f.write("## Summary\n\n")
        f.write("| Epsilon | Verified | Unsafe | Timeout | Rate |\n")
        f.write("|---------|----------|--------|---------|------|\n")
        for eps in args.epsilon:
            s = summary_data[str(eps)]
            f.write(f"| {eps} | {s['verified']} | {s['unsafe']} | {s['timeout']} | {s['rate_percent']}% |\n")
        f.write("\n## Details\n\n")
        f.write("| Sample | Margin | Label | " + " | ".join([f"ε={e}" for e in args.epsilon]) + " |\n")
        f.write("|--------|--------|-------|" + "|".join(["-------" for _ in args.epsilon]) + "|\n")
        for sd in sample_details:
            row = f"| {sd['sample_idx']} | {sd['margin']:.4f} | {sd['label_str']} |"
            for eps in args.epsilon:
                r = sd['results'][str(eps)]
                if r['verified'] is True:
                    row += f" ✓ {r['time']}s |"
                elif r['verified'] is False:
                    row += f" ✗ {r['time']}s |"
                else:
                    row += f" ? |"
            f.write(row + "\n")

    print(f"Markdown saved to: {md_path}")


if __name__ == '__main__':
    main()
