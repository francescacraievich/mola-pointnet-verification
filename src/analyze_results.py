"""
Analyze Verification Results and Generate Comparison Plots

Compares formal verification results from αβ-CROWN with empirical
adversarial attack results from NSGA-III on MOLA SLAM.

Key outputs:
    1. Verified rate vs epsilon plot
    2. Comparison with NSGA-III ATE (Absolute Trajectory Error)
    3. Critical epsilon identification
    4. Summary statistics
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# NSGA-III baseline results (from mola-adversarial-nsga3)
# Perturbation in cm -> ATE in cm
NSGA3_RESULTS = {
    0.0: 23.0,   # Baseline ATE
    1.5: 32.0,   # Degradation starts
    3.5: 65.0,   # Significant degradation
    4.6: 85.0,   # SLAM failure
}

# Critical thresholds from NSGA-III analysis
NSGA3_CRITICAL_PERTURBATION = 1.5  # cm - where degradation starts
NSGA3_FAILURE_PERTURBATION = 4.6   # cm - where SLAM fails


def load_verification_results(results_path: Path) -> Dict:
    """Load verification results from JSON file."""
    with open(results_path) as f:
        return json.load(f)


def extract_verification_data(results: Dict) -> Tuple[Dict, Dict]:
    """
    Extract verification data organized by property type.

    Returns:
        Tuple of (robustness_data, safety_data)
        Each is a dict with epsilon -> verified_rate mapping
    """
    robustness = {}
    safety = {}

    for prop_result in results["results"].get("robustness", []):
        eps = prop_result["epsilon"]
        rate = prop_result["verified_rate"]
        robustness[eps] = rate

    for prop_result in results["results"].get("safety", []):
        eps = prop_result["epsilon"]
        rate = prop_result["verified_rate"]
        safety[eps] = rate

    return robustness, safety


def find_critical_epsilon(
    epsilon_values: List[float],
    verified_rates: List[float],
    threshold: float = 50.0,
) -> Optional[float]:
    """
    Find the epsilon value where verified rate drops below threshold.

    Uses linear interpolation to estimate the exact crossing point.

    Args:
        epsilon_values: List of epsilon values (sorted)
        verified_rates: Corresponding verified rates
        threshold: Threshold percentage (default 50%)

    Returns:
        Estimated critical epsilon, or None if rate never drops below threshold
    """
    for i in range(len(verified_rates)):
        if verified_rates[i] < threshold:
            if i == 0:
                return epsilon_values[0]
            # Linear interpolation
            eps_prev, eps_curr = epsilon_values[i - 1], epsilon_values[i]
            rate_prev, rate_curr = verified_rates[i - 1], verified_rates[i]
            # Find epsilon where rate = threshold
            if rate_prev != rate_curr:
                critical_eps = eps_prev + (eps_curr - eps_prev) * (threshold - rate_prev) / (rate_curr - rate_prev)
                return critical_eps
            return eps_curr
    return None


def plot_verified_vs_epsilon(
    robustness: Dict[float, float],
    safety: Dict[float, float],
    output_path: Path,
    title: str = "Verification Rate vs Perturbation Bound",
) -> None:
    """
    Plot verified rate vs epsilon for both properties.

    Args:
        robustness: epsilon -> verified_rate for robustness property
        safety: epsilon -> verified_rate for safety property
        output_path: Path to save the figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot robustness
    eps_rob = sorted(robustness.keys())
    rates_rob = [robustness[e] for e in eps_rob]
    ax.plot(eps_rob, rates_rob, 'b-o', linewidth=2, markersize=8, label='Local Robustness')

    # Plot safety
    eps_safe = sorted(safety.keys())
    rates_safe = [safety[e] for e in eps_safe]
    ax.plot(eps_safe, rates_safe, 'r-s', linewidth=2, markersize=8, label='Safety Property')

    # Add 50% threshold line
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='50% Threshold')

    # Add NSGA-III critical perturbation reference
    ax.axvline(x=NSGA3_CRITICAL_PERTURBATION / 100, color='green', linestyle=':', linewidth=2,
               label=f'NSGA-III Critical ({NSGA3_CRITICAL_PERTURBATION}cm)')

    ax.set_xlabel('Perturbation Bound ε (meters)', fontsize=12)
    ax.set_ylabel('Verified Rate (%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    # Set x-axis to show values in cm as secondary axis
    ax2 = ax.secondary_xaxis('top', functions=(lambda x: x * 100, lambda x: x / 100))
    ax2.set_xlabel('Perturbation Bound (cm)', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def plot_comparison_with_nsga3(
    robustness: Dict[float, float],
    safety: Dict[float, float],
    output_path: Path,
) -> None:
    """
    Create comparison plot between verification results and NSGA-III ATE.

    This plot shows:
    - Left Y-axis: Verification rate (%)
    - Right Y-axis: SLAM ATE (cm)
    - X-axis: Perturbation bound

    Args:
        robustness: epsilon -> verified_rate for robustness
        safety: epsilon -> verified_rate for safety
        output_path: Path to save the figure
    """
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot verification results (left Y-axis)
    eps_rob = sorted(robustness.keys())
    rates_rob = [robustness[e] for e in eps_rob]
    eps_safe = sorted(safety.keys())
    rates_safe = [safety[e] for e in eps_safe]

    # Convert epsilon to cm for comparison
    eps_rob_cm = [e * 100 for e in eps_rob]
    eps_safe_cm = [e * 100 for e in eps_safe]

    line1, = ax1.plot(eps_rob_cm, rates_rob, 'b-o', linewidth=2, markersize=8, label='Robustness Verified')
    line2, = ax1.plot(eps_safe_cm, rates_safe, 'g-s', linewidth=2, markersize=8, label='Safety Verified')

    ax1.set_xlabel('Perturbation Magnitude (cm)', fontsize=12)
    ax1.set_ylabel('Verified Rate (%)', fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0, 105)

    # Plot NSGA-III ATE results (right Y-axis)
    ax2 = ax1.twinx()
    nsga_eps = sorted(NSGA3_RESULTS.keys())
    nsga_ate = [NSGA3_RESULTS[e] for e in nsga_eps]

    line3, = ax2.plot(nsga_eps, nsga_ate, 'r-^', linewidth=2, markersize=10, label='SLAM ATE (NSGA-III)')
    ax2.fill_between(nsga_eps, nsga_ate, alpha=0.2, color='red')

    ax2.set_ylabel('SLAM Absolute Trajectory Error (cm)', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 100)

    # Add critical thresholds
    ax1.axvline(x=NSGA3_CRITICAL_PERTURBATION, color='orange', linestyle='--', linewidth=2,
                label=f'SLAM Degradation ({NSGA3_CRITICAL_PERTURBATION}cm)')
    ax1.axvline(x=NSGA3_FAILURE_PERTURBATION, color='darkred', linestyle='--', linewidth=2,
                label=f'SLAM Failure ({NSGA3_FAILURE_PERTURBATION}cm)')

    # Combined legend
    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center left', fontsize=10)

    ax1.set_title('Formal Verification vs Empirical SLAM Robustness\n(αβ-CROWN vs NSGA-III)', fontsize=14)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def plot_certified_accuracy(
    robustness: Dict[float, float],
    base_accuracy: float,
    output_path: Path,
) -> None:
    """
    Plot certified accuracy vs epsilon.

    Certified accuracy = base_accuracy * (verified_rate / 100)

    Args:
        robustness: epsilon -> verified_rate
        base_accuracy: Model accuracy on test set (e.g., 99.97%)
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    eps_values = sorted(robustness.keys())
    verified_rates = [robustness[e] for e in eps_values]
    certified_acc = [base_accuracy * (r / 100) for r in verified_rates]

    ax.plot(eps_values, certified_acc, 'purple', linewidth=2, marker='o', markersize=8)
    ax.fill_between(eps_values, certified_acc, alpha=0.3, color='purple')

    ax.axhline(y=base_accuracy, color='green', linestyle='--', linewidth=1,
               label=f'Clean Accuracy ({base_accuracy:.1f}%)')

    ax.set_xlabel('Perturbation Bound ε (meters)', fontsize=12)
    ax.set_ylabel('Certified Accuracy (%)', fontsize=12)
    ax.set_title('Certified Accuracy vs Perturbation Bound', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    # Add annotations
    for i, (eps, acc) in enumerate(zip(eps_values, certified_acc)):
        ax.annotate(f'{acc:.1f}%', (eps, acc), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def generate_summary_report(
    robustness: Dict[float, float],
    safety: Dict[float, float],
    output_path: Path,
    base_accuracy: float = 99.97,
) -> None:
    """
    Generate a text summary report of the analysis.

    Args:
        robustness: epsilon -> verified_rate for robustness
        safety: epsilon -> verified_rate for safety
        output_path: Path to save the report
        base_accuracy: Model's clean accuracy
    """
    lines = []
    lines.append("=" * 70)
    lines.append("VERIFICATION ANALYSIS REPORT")
    lines.append("MLP-LiDAR Formal Verification with αβ-CROWN")
    lines.append("=" * 70)
    lines.append("")

    # Model info
    lines.append("MODEL INFORMATION")
    lines.append("-" * 40)
    lines.append(f"Architecture: MLP (3 → 256 → 256 → 128 → 3)")
    lines.append(f"Parameters: ~100K")
    lines.append(f"Clean Accuracy: {base_accuracy:.2f}%")
    lines.append("")

    # Robustness results
    lines.append("LOCAL ROBUSTNESS PROPERTY")
    lines.append("-" * 40)
    lines.append("∀x' : ||x' - x₀||_∞ ≤ ε → f(x') = f(x₀)")
    lines.append("")
    lines.append(f"{'Epsilon (m)':>12} | {'Epsilon (cm)':>12} | {'Verified Rate':>15}")
    lines.append("-" * 45)

    eps_rob = sorted(robustness.keys())
    for eps in eps_rob:
        lines.append(f"{eps:>12.3f} | {eps*100:>12.1f} | {robustness[eps]:>14.1f}%")

    critical_rob = find_critical_epsilon(eps_rob, [robustness[e] for e in eps_rob])
    if critical_rob:
        lines.append(f"\nCritical ε (50% threshold): {critical_rob:.4f}m ({critical_rob*100:.2f}cm)")
    else:
        lines.append("\nVerified rate never drops below 50%")
    lines.append("")

    # Safety results
    lines.append("SAFETY PROPERTY")
    lines.append("-" * 40)
    lines.append("OBSTACLE → never classified as GROUND")
    lines.append("")
    lines.append(f"{'Epsilon (m)':>12} | {'Epsilon (cm)':>12} | {'Verified Rate':>15}")
    lines.append("-" * 45)

    eps_safe = sorted(safety.keys())
    for eps in eps_safe:
        lines.append(f"{eps:>12.3f} | {eps*100:>12.1f} | {safety[eps]:>14.1f}%")

    critical_safe = find_critical_epsilon(eps_safe, [safety[e] for e in eps_safe])
    if critical_safe:
        lines.append(f"\nCritical ε (50% threshold): {critical_safe:.4f}m ({critical_safe*100:.2f}cm)")
    else:
        lines.append("\nVerified rate never drops below 50%")
    lines.append("")

    # NSGA-III comparison
    lines.append("COMPARISON WITH NSGA-III ADVERSARIAL ATTACKS")
    lines.append("-" * 40)
    lines.append("MOLA SLAM degradation under adversarial perturbations:")
    lines.append("")
    lines.append(f"{'Perturbation (cm)':>18} | {'ATE (cm)':>10} | {'Status':>20}")
    lines.append("-" * 55)
    lines.append(f"{'0.0':>18} | {'23':>10} | {'Baseline':>20}")
    lines.append(f"{'1.5':>18} | {'32':>10} | {'Degradation starts':>20}")
    lines.append(f"{'3.5':>18} | {'65':>10} | {'Significant degradation':>20}")
    lines.append(f"{'4.6':>18} | {'85':>10} | {'SLAM failure':>20}")
    lines.append("")

    # Correlation analysis
    lines.append("CORRELATION ANALYSIS")
    lines.append("-" * 40)

    if critical_rob:
        diff_rob = abs(critical_rob * 100 - NSGA3_CRITICAL_PERTURBATION)
        lines.append(f"Robustness critical ε: {critical_rob*100:.2f}cm")
        lines.append(f"NSGA-III critical perturbation: {NSGA3_CRITICAL_PERTURBATION}cm")
        lines.append(f"Difference: {diff_rob:.2f}cm")

        if diff_rob < 1.0:
            lines.append("→ STRONG correlation with empirical failure threshold!")
        elif diff_rob < 2.0:
            lines.append("→ MODERATE correlation with empirical failure threshold")
        else:
            lines.append("→ WEAK correlation with empirical failure threshold")
    lines.append("")

    # Conclusions
    lines.append("CONCLUSIONS")
    lines.append("-" * 40)

    if critical_rob and critical_rob * 100 < 3.0:
        lines.append("✓ Formal verification successfully identifies vulnerability")
        lines.append("  region that correlates with empirical SLAM failures.")
        lines.append("")
        lines.append("  The MLP classifier loses robustness guarantees at")
        lines.append(f"  perturbations around {critical_rob*100:.2f}cm, which is")
        lines.append(f"  {'close to' if abs(critical_rob*100 - NSGA3_CRITICAL_PERTURBATION) < 1.5 else 'in the same order as'} the {NSGA3_CRITICAL_PERTURBATION}cm threshold")
        lines.append("  where MOLA SLAM begins to degrade.")
    else:
        lines.append("Further analysis needed to correlate verification")
        lines.append("results with empirical SLAM failure thresholds.")

    lines.append("")
    lines.append("=" * 70)

    report = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(report)

    print(f"Saved: {output_path}")
    print("\n" + report)


def main():
    parser = argparse.ArgumentParser(description="Analyze verification results")
    parser.add_argument(
        "--results",
        type=str,
        default="results/verification_results.json",
        help="Path to verification results JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/figures",
        help="Directory to save figures",
    )
    parser.add_argument(
        "--base-accuracy",
        type=float,
        default=99.97,
        help="Model's clean accuracy (%)",
    )

    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    print(f"Loading results from {results_path}...")
    if not results_path.exists():
        print(f"Error: Results file not found at {results_path}")
        print("Run verification first: python src/verify.py")
        return 1

    results = load_verification_results(results_path)

    # Extract data
    robustness, safety = extract_verification_data(results)

    if not robustness and not safety:
        print("Error: No verification results found in file")
        return 1

    print(f"Loaded {len(robustness)} robustness and {len(safety)} safety results")

    # Generate plots
    print("\nGenerating plots...")

    plot_verified_vs_epsilon(
        robustness,
        safety,
        output_dir / "verified_vs_epsilon.png",
    )

    plot_comparison_with_nsga3(
        robustness,
        safety,
        output_dir / "comparison_nsga3.png",
    )

    if robustness:
        plot_certified_accuracy(
            robustness,
            args.base_accuracy,
            output_dir / "certified_accuracy.png",
        )

    # Generate report
    generate_summary_report(
        robustness,
        safety,
        output_dir.parent / "analysis_report.txt",
        args.base_accuracy,
    )

    print("\nAnalysis complete!")
    print(f"Figures saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
