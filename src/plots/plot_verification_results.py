#!/usr/bin/env python3
"""Plot verification results comparing ERAN and α,β-CROWN."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "results"


def load_results():
    """Load verification results from JSON files."""
    # Load ERAN results
    eran_path = RESULTS_DIR / "eran_verification_2.json"
    with open(eran_path) as f:
        eran_data = json.load(f)

    # Load α,β-CROWN results
    abcrown_path = RESULTS_DIR / "abcrown_verification_1.json"
    with open(abcrown_path) as f:
        abcrown_data = json.load(f)

    return eran_data, abcrown_data


def plot_verification_comparison(save_path=None):
    """Create plot comparing ERAN and α,β-CROWN verification rates."""
    eran_data, abcrown_data = load_results()

    # Extract ERAN data
    eran_epsilons = []
    eran_rates = []
    for eps_str, data in eran_data["summary"].items():
        eran_epsilons.append(float(eps_str))
        eran_rates.append(data["rate_percent"])

    # Sort by epsilon
    eran_sorted = sorted(zip(eran_epsilons, eran_rates))
    eran_epsilons = [x[0] for x in eran_sorted]
    eran_rates = [x[1] for x in eran_sorted]

    # Extract α,β-CROWN data
    abcrown_epsilons = []
    abcrown_rates = []
    for eps_str, data in abcrown_data["summary"].items():
        abcrown_epsilons.append(float(eps_str))
        abcrown_rates.append(data["rate_percent"])

    # Sort by epsilon
    abcrown_sorted = sorted(zip(abcrown_epsilons, abcrown_rates))
    abcrown_epsilons = [x[0] for x in abcrown_sorted]
    abcrown_rates = [x[1] for x in abcrown_sorted]

    # Create figure with style similar to the reference
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot lines with markers
    ax.plot(
        eran_epsilons,
        eran_rates,
        "b-s",
        linewidth=2,
        markersize=8,
        label="ERAN (DeepZono)",
        markerfacecolor="blue",
        markeredgecolor="blue",
    )
    ax.plot(
        abcrown_epsilons,
        abcrown_rates,
        "r-o",
        linewidth=2,
        markersize=8,
        label="α,β-CROWN",
        markerfacecolor="red",
        markeredgecolor="red",
    )

    # Formatting
    ax.set_xlabel("ε (perturbation radius)", fontsize=12)
    ax.set_ylabel("Verified robustness", fontsize=12)
    ax.set_title(
        "PointNet Robustness Verification\nMOLA LiDAR Dataset", fontsize=14, fontweight="bold"
    )

    # Y-axis as percentage
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])

    # X-axis - use log scale for better visualization
    ax.set_xscale("log")
    ax.set_xlim(0.0008, 0.6)

    # Grid
    ax.grid(True, alpha=0.3, linestyle="-")
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)

    # Border
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_verification_single_scale(save_path=None):
    """Create plot with linear scale with all epsilon values."""
    eran_data, abcrown_data = load_results()

    # Extract ERAN data
    eran_epsilons = []
    eran_rates = []
    for eps_str, data in eran_data["summary"].items():
        eran_epsilons.append(float(eps_str))
        eran_rates.append(data["rate_percent"])

    eran_sorted = sorted(zip(eran_epsilons, eran_rates))
    eran_epsilons = [x[0] for x in eran_sorted]
    eran_rates = [x[1] for x in eran_sorted]

    # Extract α,β-CROWN data (only epsilon <= 0.1)
    abcrown_epsilons = []
    abcrown_rates = []
    for eps_str, data in abcrown_data["summary"].items():
        eps = float(eps_str)
        if eps <= 0.1:
            abcrown_epsilons.append(eps)
            abcrown_rates.append(data["rate_percent"])

    abcrown_sorted = sorted(zip(abcrown_epsilons, abcrown_rates))
    abcrown_epsilons = [x[0] for x in abcrown_sorted]
    abcrown_rates = [x[1] for x in abcrown_sorted]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot lines with markers - thinner and more elegant
    ax.plot(
        eran_epsilons,
        eran_rates,
        "b-s",
        linewidth=1.2,
        markersize=6,
        label="ERAN (DeepZono)",
        markerfacecolor="white",
        markeredgecolor="blue",
        markeredgewidth=1.2,
    )
    ax.plot(
        abcrown_epsilons,
        abcrown_rates,
        "r-o",
        linewidth=1.2,
        markersize=6,
        label="α,β-CROWN",
        markerfacecolor="white",
        markeredgecolor="red",
        markeredgewidth=1.2,
    )

    # Formatting
    ax.set_xlabel("ε", fontsize=12)
    ax.set_ylabel("Verified robustness", fontsize=12)

    # Y-axis as percentage
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])

    # X-axis linear - up to 0.1
    ax.set_xlim(0, 0.11)

    # Grid - lighter
    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    # Border - thinner
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    print("=" * 60)
    print("Verification Results Visualization")
    print("=" * 60)

    # Plot with log scale (full range)
    print("\n1. Creating comparison plot (log scale)...")
    plot_verification_comparison(save_path=RESULTS_DIR / "verification_comparison.png")

    # Plot with linear scale (focused range)
    print("\n2. Creating comparison plot (linear scale, ε ≤ 0.1)...")
    plot_verification_single_scale(save_path=RESULTS_DIR / "verification_comparison_linear.png")

    print("\n" + "=" * 60)
    print("Done! Check results/ folder:")
    print("  - verification_comparison.png (log scale)")
    print("  - verification_comparison_linear.png (linear scale)")
    print("=" * 60)


if __name__ == "__main__":
    main()
