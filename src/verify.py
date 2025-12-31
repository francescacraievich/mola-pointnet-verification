"""
Formal Verification with αβ-CROWN

Verifies robustness properties of the MLP classifier using αβ-CROWN.
This script generates VNN-LIB specifications and runs verification.

Properties verified:
    1. Local Robustness: ∀x' : ||x' - x₀||_∞ ≤ ε → f(x') = f(x₀)
    2. Safety: ∀x' : ||x' - x₀||_∞ ≤ ε ∧ f(x₀)=OBSTACLE → f(x') ≠ GROUND

Epsilon values tested: {0.01, 0.02, 0.03, 0.05, 0.10} meters
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from model import CLASS_GROUND, CLASS_OBSTACLE, CLASS_OTHER, CLASS_NAMES, load_model


# Epsilon values to test (in meters)
EPSILON_VALUES = [0.01, 0.02, 0.03, 0.05, 0.10]


@dataclass
class VerificationResult:
    """Result of a single verification query."""

    sample_idx: int
    epsilon: float
    property_type: str  # "robustness" or "safety"
    true_label: int
    status: str  # "verified", "violated", "timeout", "unknown"
    time_seconds: float
    counterexample: Optional[np.ndarray] = None


@dataclass
class VerificationSummary:
    """Summary of verification results for a property and epsilon."""

    property_type: str
    epsilon: float
    total_samples: int
    verified: int = 0
    violated: int = 0
    timeout: int = 0
    unknown: int = 0
    total_time: float = 0.0
    results: List[VerificationResult] = field(default_factory=list)

    @property
    def verified_rate(self) -> float:
        """Percentage of samples verified."""
        if self.total_samples == 0:
            return 0.0
        return 100.0 * self.verified / self.total_samples

    @property
    def avg_time(self) -> float:
        """Average verification time per sample."""
        if self.total_samples == 0:
            return 0.0
        return self.total_time / self.total_samples

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "property_type": self.property_type,
            "epsilon": self.epsilon,
            "total_samples": self.total_samples,
            "verified": self.verified,
            "violated": self.violated,
            "timeout": self.timeout,
            "unknown": self.unknown,
            "verified_rate": self.verified_rate,
            "total_time": self.total_time,
            "avg_time": self.avg_time,
        }


def generate_vnnlib_robustness(
    x: np.ndarray,
    true_class: int,
    epsilon: float,
    num_classes: int = 3,
) -> str:
    """
    Generate VNN-LIB specification for local robustness property.

    The property states: the output class should remain the same as true_class
    for all inputs within epsilon L∞ ball of x.

    Args:
        x: Input point (normalized coordinates)
        true_class: The correct class label
        epsilon: L∞ perturbation bound
        num_classes: Number of output classes

    Returns:
        VNN-LIB format string
    """
    lines = ["; Local robustness property", f"; True class: {CLASS_NAMES.get(true_class, true_class)}", f"; Epsilon: {epsilon}", ""]

    # Declare input variables
    for i in range(len(x)):
        lines.append(f"(declare-const X_{i} Real)")

    # Declare output variables
    for i in range(num_classes):
        lines.append(f"(declare-const Y_{i} Real)")

    lines.append("")

    # Input constraints: x - epsilon <= X <= x + epsilon
    lines.append("; Input constraints (L∞ ball)")
    for i, xi in enumerate(x):
        lines.append(f"(assert (>= X_{i} {xi - epsilon:.10f}))")
        lines.append(f"(assert (<= X_{i} {xi + epsilon:.10f}))")

    lines.append("")

    # Output constraint: true_class has maximum logit
    # This means: Y_true_class > Y_i for all i != true_class
    # Negated for verification (we verify the negation is UNSAT):
    # There exists some class j != true_class where Y_j >= Y_true_class
    lines.append("; Output constraint: some other class has >= logit (negation of robustness)")

    # Create disjunction: OR over all other classes
    other_classes = [i for i in range(num_classes) if i != true_class]
    if len(other_classes) == 1:
        j = other_classes[0]
        lines.append(f"(assert (>= Y_{j} Y_{true_class}))")
    else:
        # Multiple other classes: use OR
        or_clauses = [f"(>= Y_{j} Y_{true_class})" for j in other_classes]
        lines.append(f"(assert (or {' '.join(or_clauses)}))")

    return "\n".join(lines)


def generate_vnnlib_safety(
    x: np.ndarray,
    epsilon: float,
) -> str:
    """
    Generate VNN-LIB specification for safety property.

    The property states: an OBSTACLE should never be classified as GROUND.
    For verification, we check if there exists a perturbation that causes
    GROUND to have the maximum logit.

    Args:
        x: Input point (normalized coordinates) classified as OBSTACLE
        epsilon: L∞ perturbation bound

    Returns:
        VNN-LIB format string
    """
    lines = [
        "; Safety property: OBSTACLE never classified as GROUND",
        f"; Epsilon: {epsilon}",
        "",
    ]

    # Declare input variables
    for i in range(len(x)):
        lines.append(f"(declare-const X_{i} Real)")

    # Declare output variables (3 classes)
    for i in range(3):
        lines.append(f"(declare-const Y_{i} Real)")

    lines.append("")

    # Input constraints: x - epsilon <= X <= x + epsilon
    lines.append("; Input constraints (L∞ ball)")
    for i, xi in enumerate(x):
        lines.append(f"(assert (>= X_{i} {xi - epsilon:.10f}))")
        lines.append(f"(assert (<= X_{i} {xi + epsilon:.10f}))")

    lines.append("")

    # Output constraint for safety violation:
    # GROUND (class 0) has higher logit than both OBSTACLE (class 1) and OTHER (class 2)
    # This means the point would be classified as GROUND
    lines.append("; Safety violation: GROUND has maximum logit")
    lines.append(f"(assert (>= Y_{CLASS_GROUND} Y_{CLASS_OBSTACLE}))")
    lines.append(f"(assert (>= Y_{CLASS_GROUND} Y_{CLASS_OTHER}))")

    return "\n".join(lines)


def run_abcrown_verification(
    onnx_path: Path,
    vnnlib_path: Path,
    config_path: Optional[Path] = None,
    timeout: int = 60,
) -> Tuple[str, float, Optional[np.ndarray]]:
    """
    Run αβ-CROWN verification on a single property.

    Args:
        onnx_path: Path to ONNX model
        vnnlib_path: Path to VNN-LIB specification
        config_path: Optional path to αβ-CROWN config
        timeout: Timeout in seconds

    Returns:
        Tuple of (status, time, counterexample)
        status: "verified", "violated", "timeout", or "unknown"
    """
    # Build command
    cmd = [
        "python",
        "-m",
        "complete_verifier.abcrown",
        "--config", str(config_path) if config_path else "configs/verification_config.yaml",
        "--onnx_path", str(onnx_path),
        "--vnnlib_path", str(vnnlib_path),
        "--timeout", str(timeout),
    ]

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 10,  # Extra time for process overhead
        )
        elapsed = time.time() - start_time

        # Parse output
        output = result.stdout + result.stderr

        if "Result: safe" in output or "verified" in output.lower():
            return "verified", elapsed, None
        elif "Result: unsafe" in output or "sat" in output.lower():
            # Try to extract counterexample
            counterexample = None
            # αβ-CROWN typically outputs counterexample in specific format
            return "violated", elapsed, counterexample
        elif "timeout" in output.lower():
            return "timeout", elapsed, None
        else:
            return "unknown", elapsed, None

    except subprocess.TimeoutExpired:
        return "timeout", timeout, None
    except FileNotFoundError:
        print("Warning: αβ-CROWN not found. Using mock verification.")
        return "unknown", 0.0, None
    except Exception as e:
        print(f"Verification error: {e}")
        return "unknown", 0.0, None


def mock_verification(
    model: torch.nn.Module,
    x: np.ndarray,
    epsilon: float,
    property_type: str,
    true_class: int,
    num_samples: int = 1000,
) -> Tuple[str, float]:
    """
    Mock verification using random sampling (for testing without αβ-CROWN).

    This is NOT sound verification - just empirical testing.

    Args:
        model: PyTorch model
        x: Input point
        epsilon: Perturbation bound
        property_type: "robustness" or "safety"
        true_class: Original class label
        num_samples: Number of random samples to test

    Returns:
        Tuple of (status, time)
    """
    start_time = time.time()
    model.eval()

    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        for _ in range(num_samples):
            # Random perturbation within L∞ ball
            perturbation = np.random.uniform(-epsilon, epsilon, size=x.shape)
            x_perturbed = x + perturbation
            x_perturbed_tensor = torch.tensor(x_perturbed, dtype=torch.float32).unsqueeze(0)

            pred = model.predict(x_perturbed_tensor).item()

            if property_type == "robustness":
                if pred != true_class:
                    return "violated", time.time() - start_time
            elif property_type == "safety":
                # Safety: OBSTACLE should not become GROUND
                if true_class == CLASS_OBSTACLE and pred == CLASS_GROUND:
                    return "violated", time.time() - start_time

    # No violation found (but not proven safe!)
    return "verified", time.time() - start_time


def verify_property(
    model: torch.nn.Module,
    onnx_path: Path,
    x: np.ndarray,
    true_class: int,
    epsilon: float,
    property_type: str,
    sample_idx: int,
    config_path: Optional[Path] = None,
    timeout: int = 60,
    use_mock: bool = False,
) -> VerificationResult:
    """
    Verify a property for a single input point.

    Args:
        model: PyTorch model (for mock verification)
        onnx_path: Path to ONNX model
        x: Input point
        true_class: True class label
        epsilon: Perturbation bound
        property_type: "robustness" or "safety"
        sample_idx: Index of the sample
        config_path: Path to αβ-CROWN config
        timeout: Timeout in seconds
        use_mock: Use mock verification instead of αβ-CROWN

    Returns:
        VerificationResult
    """
    if use_mock:
        status, elapsed = mock_verification(model, x, epsilon, property_type, true_class)
        return VerificationResult(
            sample_idx=sample_idx,
            epsilon=epsilon,
            property_type=property_type,
            true_label=true_class,
            status=status,
            time_seconds=elapsed,
        )

    # Generate VNN-LIB specification
    with tempfile.NamedTemporaryFile(mode="w", suffix=".vnnlib", delete=False) as f:
        if property_type == "robustness":
            vnnlib = generate_vnnlib_robustness(x, true_class, epsilon)
        else:  # safety
            vnnlib = generate_vnnlib_safety(x, epsilon)
        f.write(vnnlib)
        vnnlib_path = Path(f.name)

    try:
        status, elapsed, counterexample = run_abcrown_verification(
            onnx_path, vnnlib_path, config_path, timeout
        )
    finally:
        # Clean up temp file
        vnnlib_path.unlink(missing_ok=True)

    return VerificationResult(
        sample_idx=sample_idx,
        epsilon=epsilon,
        property_type=property_type,
        true_label=true_class,
        status=status,
        time_seconds=elapsed,
        counterexample=counterexample,
    )


def select_verification_samples(
    points: np.ndarray,
    labels: np.ndarray,
    model: torch.nn.Module,
    num_samples: int = 100,
    property_type: str = "robustness",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select samples for verification.

    For robustness: select correctly classified samples from all classes.
    For safety: select correctly classified OBSTACLE samples.

    Args:
        points: Test points
        labels: True labels
        model: Trained model
        num_samples: Number of samples to select
        property_type: "robustness" or "safety"

    Returns:
        Tuple of (selected_points, selected_labels, selected_indices)
    """
    model.eval()
    with torch.no_grad():
        predictions = model.predict(torch.tensor(points, dtype=torch.float32)).numpy()

    # Find correctly classified samples
    correct_mask = predictions == labels

    if property_type == "safety":
        # For safety property, only use OBSTACLE samples
        obstacle_mask = labels == CLASS_OBSTACLE
        valid_mask = correct_mask & obstacle_mask
    else:
        valid_mask = correct_mask

    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < num_samples:
        print(f"Warning: Only {len(valid_indices)} valid samples available, requested {num_samples}")
        num_samples = len(valid_indices)

    # Random selection
    selected_indices = np.random.choice(valid_indices, size=num_samples, replace=False)

    return points[selected_indices], labels[selected_indices], selected_indices


def run_verification(
    model: torch.nn.Module,
    onnx_path: Path,
    test_points: np.ndarray,
    test_labels: np.ndarray,
    epsilon_values: List[float],
    num_samples: int = 100,
    config_path: Optional[Path] = None,
    timeout: int = 60,
    use_mock: bool = False,
    output_dir: Path = Path("results"),
) -> Dict[str, List[VerificationSummary]]:
    """
    Run full verification for all properties and epsilon values.

    Args:
        model: Trained PyTorch model
        onnx_path: Path to ONNX model
        test_points: Test dataset points
        test_labels: Test dataset labels
        epsilon_values: List of epsilon values to test
        num_samples: Number of samples to verify per epsilon
        config_path: Path to αβ-CROWN configuration
        timeout: Timeout per verification query
        use_mock: Use mock verification
        output_dir: Directory to save results

    Returns:
        Dictionary with verification summaries for each property
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "robustness": [],
        "safety": [],
    }

    # Select samples for each property
    print("\nSelecting samples for verification...")

    robustness_points, robustness_labels, robustness_indices = select_verification_samples(
        test_points, test_labels, model, num_samples, "robustness"
    )
    print(f"  Robustness: {len(robustness_points)} samples")

    safety_points, safety_labels, safety_indices = select_verification_samples(
        test_points, test_labels, model, num_samples, "safety"
    )
    print(f"  Safety: {len(safety_points)} samples")

    # Verify each property and epsilon
    for property_type in ["robustness", "safety"]:
        if property_type == "robustness":
            points, labels, indices = robustness_points, robustness_labels, robustness_indices
        else:
            points, labels, indices = safety_points, safety_labels, safety_indices

        for epsilon in epsilon_values:
            print(f"\n{'='*60}")
            print(f"Property: {property_type.upper()}, Epsilon: {epsilon}")
            print(f"{'='*60}")

            summary = VerificationSummary(
                property_type=property_type,
                epsilon=epsilon,
                total_samples=len(points),
            )

            for i, (x, label, idx) in enumerate(zip(points, labels, indices)):
                result = verify_property(
                    model=model,
                    onnx_path=onnx_path,
                    x=x,
                    true_class=int(label),
                    epsilon=epsilon,
                    property_type=property_type,
                    sample_idx=int(idx),
                    config_path=config_path,
                    timeout=timeout,
                    use_mock=use_mock,
                )

                summary.results.append(result)
                summary.total_time += result.time_seconds

                if result.status == "verified":
                    summary.verified += 1
                elif result.status == "violated":
                    summary.violated += 1
                elif result.status == "timeout":
                    summary.timeout += 1
                else:
                    summary.unknown += 1

                # Progress update
                if (i + 1) % 10 == 0 or i == len(points) - 1:
                    print(f"  Progress: {i+1}/{len(points)} | "
                          f"Verified: {summary.verified} | "
                          f"Violated: {summary.violated} | "
                          f"Rate: {summary.verified_rate:.1f}%")

            results[property_type].append(summary)

            print(f"\nSummary for {property_type}, ε={epsilon}:")
            print(f"  Verified: {summary.verified}/{summary.total_samples} ({summary.verified_rate:.1f}%)")
            print(f"  Violated: {summary.violated}")
            print(f"  Timeout: {summary.timeout}")
            print(f"  Avg time: {summary.avg_time:.2f}s")

    return results


def save_results(
    results: Dict[str, List[VerificationSummary]],
    output_path: Path,
) -> None:
    """Save verification results to JSON file."""
    output_data = {
        "metadata": {
            "epsilon_values": EPSILON_VALUES,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "results": {},
    }

    for property_type, summaries in results.items():
        output_data["results"][property_type] = [s.to_dict() for s in summaries]

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {output_path}")


def print_summary_table(results: Dict[str, List[VerificationSummary]]) -> None:
    """Print a summary table of verification results."""
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    for property_type in ["robustness", "safety"]:
        print(f"\n{property_type.upper()} Property:")
        print("-" * 50)
        print(f"{'Epsilon':>10} | {'Verified':>10} | {'Rate':>10} | {'Avg Time':>10}")
        print("-" * 50)

        for summary in results[property_type]:
            print(f"{summary.epsilon:>10.3f} | "
                  f"{summary.verified:>10} | "
                  f"{summary.verified_rate:>9.1f}% | "
                  f"{summary.avg_time:>9.2f}s")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Verify MLP model with αβ-CROWN")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/mlp_lidar.pth",
        help="Path to PyTorch model",
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        default="models/mlp_lidar.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/processed/test_points.npy",
        help="Path to test points",
    )
    parser.add_argument(
        "--test-labels",
        type=str,
        default="data/processed/test_labels.npy",
        help="Path to test labels",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/verification_config.yaml",
        help="Path to αβ-CROWN config",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to verify per epsilon",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout per verification query (seconds)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/verification_results.json",
        help="Path to save results",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock verification (random sampling) instead of αβ-CROWN",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        nargs="+",
        default=None,
        help="Custom epsilon values to test",
    )

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path)
    print(f"  Parameters: {model.count_parameters():,}")

    # Check ONNX model exists
    onnx_path = Path(args.onnx_path)
    if not onnx_path.exists():
        print(f"ONNX model not found at {onnx_path}")
        print("Run: python src/export_onnx.py")
        return 1

    # Load test data
    print(f"\nLoading test data...")
    test_points = np.load(args.test_data)
    test_labels = np.load(args.test_labels)
    print(f"  Test samples: {len(test_points)}")

    # Epsilon values
    epsilon_values = args.epsilon if args.epsilon else EPSILON_VALUES
    print(f"\nEpsilon values: {epsilon_values}")

    # Config path
    config_path = Path(args.config) if args.config else None

    # Run verification
    print("\n" + "=" * 60)
    print("STARTING VERIFICATION")
    print("=" * 60)

    if args.mock:
        print("WARNING: Using mock verification (random sampling)")
        print("         Results are NOT sound formal verification!")

    results = run_verification(
        model=model,
        onnx_path=onnx_path,
        test_points=test_points,
        test_labels=test_labels,
        epsilon_values=epsilon_values,
        num_samples=args.num_samples,
        config_path=config_path,
        timeout=args.timeout,
        use_mock=args.mock,
        output_dir=Path("results"),
    )

    # Print summary
    print_summary_table(results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, output_path)

    # Find critical epsilon
    print("\nCRITICAL EPSILON ANALYSIS:")
    for property_type in ["robustness", "safety"]:
        for summary in results[property_type]:
            if summary.verified_rate < 50:
                print(f"  {property_type}: Verified rate drops below 50% at ε={summary.epsilon}")
                break
        else:
            print(f"  {property_type}: Verified rate stays above 50% for all tested ε")

    print("\nVerification complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
