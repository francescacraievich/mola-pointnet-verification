#!/usr/bin/env python3
"""
Plot verification results for ERAN and/or α,β-CROWN.

Usage:
  # Single result:
  python plot_verification_results.py --single results/eran_verification_2.json

  # Compare two results:
  python plot_verification_results.py --compare results/eran_verification_2.json results/abcrown_verification_2.json

  # Specify output path:
  python plot_verification_results.py --single results/eran_verification_2.json -o my_plot.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.facecolor": "#E8E8E8",  # Light gray background
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "grid.color": "white",
        "grid.linewidth": 1.2,
        "grid.linestyle": "-",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
    }
)

BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"  # Output to results/figures


def load_json_results(path: Path) -> dict:
    """Load verification results from JSON file."""
    with open(path) as f:
        return json.load(f)


def extract_verification_data(data: dict) -> tuple:
    """Extract epsilons, rates, and times from verification results."""
    epsilons = []
    rates = []
    times = []

    # Check if details is a list (α,β-CROWN format) or dict (ERAN format)
    details = data.get("details", {})
    is_list_details = isinstance(details, list)

    # Pre-compute average times from list-based details (α,β-CROWN)
    avg_times_from_list = {}
    if is_list_details:
        for eps_str in data["summary"].keys():
            sample_times = []
            for sample in details:
                if "results" in sample and eps_str in sample["results"]:
                    t = sample["results"][eps_str].get("time", 0)
                    if t and t > 0:
                        sample_times.append(t)
            if sample_times:
                avg_times_from_list[eps_str] = sum(sample_times) / len(sample_times)

    for eps_str, summary in data["summary"].items():
        eps = float(eps_str)
        # Handle both ERAN and α,β-CROWN formats
        if "rate_percent" in summary:
            rate = summary["rate_percent"]
        elif "verified" in summary and "total" in summary:
            rate = 100 * summary["verified"] / summary["total"]
        elif "verified" in summary:
            # α,β-CROWN format: verified count out of 100
            rate = summary["verified"]
        else:
            continue
        epsilons.append(eps)
        rates.append(rate)

        # Extract time from details
        if is_list_details:
            # α,β-CROWN: use pre-computed average
            times.append(avg_times_from_list.get(eps_str, 0))
        elif eps_str in details:
            # ERAN: use avg_time from dict
            times.append(details[eps_str].get("avg_time", 0))
        else:
            times.append(0)

    # Sort by epsilon
    sorted_data = sorted(zip(epsilons, rates, times))
    return ([x[0] for x in sorted_data], [x[1] for x in sorted_data], [x[2] for x in sorted_data])


def get_verifier_name(data: dict) -> str:
    """Get verifier name from metadata."""
    metadata = data.get("metadata", {})
    verifier = metadata.get("verifier", "Unknown")
    model = metadata.get("model", "")

    if "eran" in verifier.lower() or "eran" in model.lower():
        domain = metadata.get("domain", "")
        return f"ERAN ({domain})" if domain else "ERAN"
    elif (
        "crown" in verifier.lower() or "abcrown" in verifier.lower() or "autolirpa" in model.lower()
    ):
        return "α,β-CROWN"
    else:
        return verifier


def plot_single(json_path: Path, output_path: Path = None, title: str = None):
    """Create two-panel plot: (a) Verified Robustness and (b) Time vs epsilon."""
    data = load_json_results(json_path)
    epsilons, rates, times = extract_verification_data(data)
    verifier_name = get_verifier_name(data)
    model_name = data.get("metadata", {}).get("model", "PointNet")

    # Choose color based on verifier
    if "ERAN" in verifier_name:
        color = "#C44E52"  # Red tone
    elif "CROWN" in verifier_name:
        color = "#4C72B0"  # Blue tone
    else:
        color = "#55A868"  # Green tone

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Add main title with verifier name 
    fig.suptitle(verifier_name, fontsize=13, fontweight="bold", y=1.02)

    #Left plot: Verified Robustness
    ax1.plot(
        epsilons,
        rates,
        "o-",
        linewidth=2,
        markersize=7,
        color=color,
        markeredgecolor="white",
        markeredgewidth=1.2,
        zorder=10,
    )
    ax1.axhline(y=50, color="#888888", linestyle="--", linewidth=1.2, zorder=5)

    ax1.set_xlabel("ε", labelpad=10)
    ax1.set_ylabel("Verified robustness")
    ax1.set_ylim(0, 105)
    ax1.set_xlim(min(epsilons) * 0.9, max(epsilons) * 1.05)
    ax1.grid(True, zorder=0)

    # Format y-axis as percentage
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}%"))

    # Remove top and right spines
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Subplot title
    ax1.set_title("Verified robustness", fontsize=11, pad=8)

    #  Right plot: Time 
    ax2.plot(
        epsilons,
        times,
        "o-",
        linewidth=2,
        markersize=7,
        color=color,
        markeredgecolor="white",
        markeredgewidth=1.2,
        zorder=10,
    )

    ax2.set_xlabel("ε", labelpad=10)
    ax2.set_ylabel("Time (s)")
    ax2.set_xlim(min(epsilons) * 0.9, max(epsilons) * 1.05)
    # Set y-axis to start from first epsilon's time value (not zero)
    if times and max(times) > 0:
        first_time = times[0]  # Time at first epsilon
        y_min = first_time * 0.9 if first_time > 0 else 0
        ax2.set_ylim(y_min, max(times) * 1.05)
    ax2.grid(True, zorder=0)

    # Remove top and right spines
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Subplot title
    ax2.set_title("Time", fontsize=11, pad=8)

    plt.tight_layout()

    # Add model name below the plots 
    fig.text(
        0.5, 0.01, f"Model: {model_name}", ha="center", fontsize=9, style="italic", color="#555555"
    )

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_comparison(
    json_path1: Path, json_path2: Path, output_path: Path = None, title: str = None
):
    """Create comparison line chart for two verification results."""
    data1 = load_json_results(json_path1)
    data2 = load_json_results(json_path2)

    eps1, rates1, times1 = extract_verification_data(data1)
    eps2, rates2, times2 = extract_verification_data(data2)

    name1 = get_verifier_name(data1)
    name2 = get_verifier_name(data2)

    # Get model names
    model1 = data1.get("metadata", {}).get("model", "")
    model2 = data2.get("metadata", {}).get("model", "")

    # Create figure with academic style
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot lines 
    ax.plot(
        eps1,
        rates1,
        "o-",
        linewidth=2,
        markersize=7,
        color="#C44E52",
        markeredgecolor="white",
        markeredgewidth=1.2,
        label=name1,
        zorder=10,
    )
    ax.plot(
        eps2,
        rates2,
        "s--",
        linewidth=2,
        markersize=7,
        color="#4C72B0",
        markeredgecolor="white",
        markeredgewidth=1.2,
        label=name2,
        zorder=10,
    )

    # Formatting
    ax.set_xlabel("ε", labelpad=10)
    ax.set_ylabel("Verified robustness")

    if title:
        ax.set_title(title, fontweight="bold", pad=10)
    else:
        ax.set_title(f"Verification Comparison: {name1} vs {name2}", fontweight="bold", pad=10)

    ax.set_ylim(0, 105)
    ax.set_xlim(min(min(eps1), min(eps2)) * 0.9, max(max(eps1), max(eps2)) * 1.05)
    ax.legend(loc="lower left", framealpha=0.95, edgecolor="white", fancybox=False)
    ax.grid(True, zorder=0)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}%"))

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    # Add model names below the x-axis label
    models_text = f"Models: {model1}, {model2}"
    fig.text(0.5, 0.01, models_text, ha="center", fontsize=9, style="italic", color="#555555")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot verification results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single result:
  python plot_verification_results.py --single results/eran_verification_2.json

  # Compare two results:
  python plot_verification_results.py --compare results/eran_verification_2.json results/abcrown_verification_2.json
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--single", type=str, metavar="JSON", help="Single JSON result files"
    )
    group.add_argument(
        "--compare",
        type=str,
        nargs=2,
        metavar="JSON",
        help="Two JSON result files to compare",
    )

    parser.add_argument("-o", "--output", type=str, help="Output path (default: auto-generated)")
    parser.add_argument("--title", type=str, help="Custom plot title")

    args = parser.parse_args()

    # Create output directory
    FIGURES_DIR.mkdir(exist_ok=True)

    if args.single:
        json_path = Path(args.single)
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = FIGURES_DIR / f"{json_path.stem}_plot.png"

        print(f"Creating single plot from: {json_path}")
        plot_single(json_path, output_path, args.title)

    elif args.compare:
        json_path1 = Path(args.compare[0])
        json_path2 = Path(args.compare[1])

        if args.output:
            output_path = Path(args.output)
        else:
            output_path = FIGURES_DIR / f"comparison_{json_path1.stem}_{json_path2.stem}.png"

        print(f"Creating comparison plot from:")
        print(f"  1. {json_path1}")
        print(f"  2. {json_path2}")
        plot_comparison(json_path1, json_path2, output_path, args.title)


if __name__ == "__main__":
    main()
