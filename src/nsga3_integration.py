"""
NSGA-III Integration Module.

Loads optimization results from NSGA-III and derives weights for PointNet labeling.
This eliminates hardcoded values and creates a direct link between:
- NSGA-III attack optimization (which parameters cause SLAM failure)
- PointNet criticality labeling (which regions are vulnerable)

The key insight: parameters that correlate with high ATE in NSGA-III
indicate which geometric properties make regions vulnerable to attack.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


# Genome parameter names (must match perturbation_generator.py)
GENOME_PARAMS = [
    "noise_dir_x",
    "noise_dir_y",
    "noise_dir_z",
    "noise_intensity",
    "curvature_targeting",
    "dropout_rate",
    "ghost_ratio",
    "cluster_dir_x",
    "cluster_dir_y",
    "cluster_dir_z",
    "cluster_strength",
    "spatial_correlation",
    "geometric_distortion",
    "edge_attack",
    "temporal_drift",
    "scanline_perturbation",
    "strategic_ghost",
]

# Mapping from genome parameters to geometric features we can compute
# These are the features that PointNet can potentially learn from xyz
PARAM_TO_FEATURE = {
    "curvature_targeting": "curvature",
    "edge_attack": "linearity",  # Edges have high linearity
    "temporal_drift": "nonplanarity",  # Drift affects non-planar regions more
    "scanline_perturbation": "density_var",  # Scanline affects density
    "geometric_distortion": "nonplanarity",  # Distortion affects complex geometry
    "noise_intensity": "curvature",  # Noise targets high-curvature
}


def load_nsga3_results(
    results_dir: Path, run_id: int = 12
) -> Dict[str, np.ndarray]:
    """
    Load NSGA-III optimization results.

    Args:
        results_dir: Directory containing optimized_genome{run_id}.*.npy files
        run_id: Run number (default: 12, the latest run)

    Returns:
        Dictionary with:
        - pareto_front: (n_solutions, 2) - [ATE, perturbation]
        - pareto_set: (n_solutions, 17) - genome parameters
        - all_points: (n_evals, 2) - all evaluated fitness values
        - valid_points: (n_valid, 2) - valid evaluations only
    """
    base_path = Path(results_dir) / f"optimized_genome{run_id}"

    results = {}

    # Load available files
    file_suffixes = [
        "pareto_front",
        "pareto_set",
        "all_points",
        "valid_points",
        "valid_genomes",
    ]

    for suffix in file_suffixes:
        file_path = base_path.with_suffix(f".{suffix}.npy")
        if file_path.exists():
            results[suffix] = np.load(file_path, allow_pickle=True)

    # Also load the best genome
    genome_path = base_path.with_suffix(".npy")
    if genome_path.exists():
        results["best_genome"] = np.load(genome_path)

    return results


def compute_parameter_correlations(
    valid_points: np.ndarray, valid_genomes: np.ndarray
) -> Dict[str, Tuple[float, float]]:
    """
    Compute correlation of each genome parameter with ATE and perturbation.

    Args:
        valid_points: (n_valid, 2) - [negative_ATE, perturbation]
        valid_genomes: (n_valid, 17) - genome parameters

    Returns:
        Dictionary mapping parameter name to (ate_correlation, pert_correlation)
    """
    # Convert negative ATE to positive
    ate_values = -valid_points[:, 0] if valid_points[:, 0].mean() < 0 else valid_points[:, 0]
    pert_values = valid_points[:, 1]

    correlations = {}
    for i, param_name in enumerate(GENOME_PARAMS):
        param_values = valid_genomes[:, i]

        # Correlation with ATE (attack effectiveness)
        corr_ate = np.corrcoef(param_values, ate_values)[0, 1]
        corr_ate = corr_ate if not np.isnan(corr_ate) else 0.0

        # Correlation with perturbation magnitude
        corr_pert = np.corrcoef(param_values, pert_values)[0, 1]
        corr_pert = corr_pert if not np.isnan(corr_pert) else 0.0

        correlations[param_name] = (corr_ate, corr_pert)

    return correlations


def derive_feature_weights(
    correlations: Dict[str, Tuple[float, float]],
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Derive weights for geometric features based on NSGA-III correlations.

    The key insight: parameters with HIGH positive correlation with ATE
    indicate which geometric properties make regions vulnerable.

    Args:
        correlations: From compute_parameter_correlations()
        normalize: Whether to normalize weights to sum to 1

    Returns:
        Dictionary mapping feature names to weights
    """
    # Map parameters to features and aggregate correlations
    feature_correlations = {
        "linearity": 0.0,
        "curvature": 0.0,
        "density_var": 0.0,
        "nonplanarity": 0.0,
    }

    feature_counts = {k: 0 for k in feature_correlations}

    for param_name, (corr_ate, _corr_pert) in correlations.items():
        if param_name in PARAM_TO_FEATURE:
            feature = PARAM_TO_FEATURE[param_name]
            # Only use positive correlations (parameters that INCREASE ATE)
            if corr_ate > 0:
                feature_correlations[feature] += corr_ate
                feature_counts[feature] += 1

    # Average if multiple parameters map to same feature
    for feature in feature_correlations:
        if feature_counts[feature] > 0:
            feature_correlations[feature] /= feature_counts[feature]

    # Normalize to sum to 1
    if normalize:
        total = sum(feature_correlations.values())
        if total > 0:
            feature_correlations = {k: v / total for k, v in feature_correlations.items()}

    return feature_correlations


def get_criticality_weights(
    nsga3_results_dir: Optional[Path] = None,
    run_id: int = 12,
    fallback_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Get weights for criticality scoring, derived from NSGA-III if available.

    This is the main entry point for the notebook to get weights dynamically.

    Args:
        nsga3_results_dir: Path to NSGA-III results (or None to use fallback)
        run_id: NSGA-III run ID
        fallback_weights: Weights to use if NSGA-III results not available

    Returns:
        Dictionary with weights for: linearity, curvature, density_var, nonplanarity
    """
    # Default fallback based on previous analysis
    if fallback_weights is None:
        fallback_weights = {
            "linearity": 0.0,  # Edge attack had low correlation
            "curvature": 0.15,  # Curvature targeting
            "density_var": 0.25,  # Scanline perturbation
            "nonplanarity": 0.60,  # Temporal drift + geometric distortion
        }

    if nsga3_results_dir is None:
        return fallback_weights

    try:
        results = load_nsga3_results(nsga3_results_dir, run_id)

        if "valid_points" in results and "valid_genomes" in results:
            correlations = compute_parameter_correlations(
                results["valid_points"], results["valid_genomes"]
            )
            weights = derive_feature_weights(correlations, normalize=True)
            print(f"Loaded NSGA-III weights from run {run_id}:")
            for feat, w in weights.items():
                print(f"  {feat}: {w:.4f}")
            return weights
        else:
            print("NSGA-III valid_genomes not found, using Pareto set analysis...")
            # Use Pareto set if full genomes not available
            if "pareto_set" in results and "pareto_front" in results:
                # Analyze most effective attack (highest ATE)
                pareto_front = results["pareto_front"]
                pareto_set = results["pareto_set"]

                # Find best attack (most negative ATE = highest attack effectiveness)
                best_idx = np.argmin(pareto_front[:, 0])
                best_genome = pareto_set[best_idx]

                # Derive weights from best genome's active parameters
                weights = analyze_best_genome(best_genome)
                print(f"Derived weights from best Pareto solution:")
                for feat, w in weights.items():
                    print(f"  {feat}: {w:.4f}")
                return weights

    except Exception as e:
        print(f"Could not load NSGA-III results: {e}")
        print("Using fallback weights")

    return fallback_weights


def analyze_best_genome(genome: np.ndarray) -> Dict[str, float]:
    """
    Analyze the best genome to derive feature weights.

    Args:
        genome: 17-dimensional genome vector

    Returns:
        Feature weights derived from active attack parameters
    """
    # Decode genome to [0, 1] scale
    genome_scaled = (genome + 1) / 2

    # Extract relevant parameters
    curvature_targeting = genome_scaled[4]
    edge_attack = genome_scaled[13]
    temporal_drift = genome_scaled[14]
    scanline = genome_scaled[15]
    geometric_distortion = genome_scaled[12]

    # Map to features based on attack strength
    weights = {
        "linearity": edge_attack * 0.5,  # Edge attack → linearity
        "curvature": curvature_targeting,  # Direct mapping
        "density_var": scanline,  # Scanline → density variation
        "nonplanarity": (temporal_drift + geometric_distortion) / 2,  # Combined
    }

    # Normalize
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    return weights


def compute_criticality_score(
    linearity: float,
    curvature: float,
    density_var: float,
    planarity: float,
    weights: Dict[str, float],
) -> float:
    """
    Compute criticality score for a point cloud region.

    Args:
        linearity: Mean linearity (edge-like features)
        curvature: Mean surface curvature
        density_var: Density variation (scanline vulnerability)
        planarity: Mean planarity (1 - nonplanarity)
        weights: Feature weights from get_criticality_weights()

    Returns:
        Criticality score in [0, 1] range
    """
    nonplanarity = 1.0 - planarity

    score = (
        linearity * weights.get("linearity", 0.0)
        + curvature * weights.get("curvature", 0.0)
        + density_var * weights.get("density_var", 0.0)
        + nonplanarity * weights.get("nonplanarity", 0.0)
    )

    return score


def get_pareto_front_summary(
    nsga3_results_dir: Path, run_id: int = 12
) -> Dict[str, any]:
    """
    Get a summary of the Pareto front for analysis.

    Returns:
        Dictionary with Pareto front statistics
    """
    results = load_nsga3_results(nsga3_results_dir, run_id)

    if "pareto_front" not in results:
        return {}

    pareto_front = results["pareto_front"]

    # Convert negative ATE to positive (cm)
    ate_values = -pareto_front[:, 0] * 100  # m to cm
    pert_values = pareto_front[:, 1]  # Already in cm

    return {
        "n_solutions": len(pareto_front),
        "best_ate_cm": float(ate_values.max()),
        "min_ate_cm": float(ate_values.min()),
        "min_perturbation_cm": float(pert_values.min()),
        "max_perturbation_cm": float(pert_values.max()),
        "baseline_ate_cm": 23.0,  # Known baseline
        "critical_threshold_cm": 1.5,  # Perturbation that causes unacceptable degradation
    }


def load_pareto_set(
    nsga3_results_dir: Path, run_id: int = 12
) -> Optional[np.ndarray]:
    """
    Load the Pareto set (genomes) from NSGA-III results.

    Args:
        nsga3_results_dir: Directory containing optimized_genome{run_id}.*.npy files
        run_id: Run number

    Returns:
        Array of shape (n_solutions, 17) with Pareto-optimal genomes,
        or None if not found
    """
    results = load_nsga3_results(nsga3_results_dir, run_id)
    return results.get("pareto_set", None)


def compute_vulnerability_from_genome(
    points: np.ndarray,
    genome: np.ndarray,
    curvature: Optional[np.ndarray] = None,
    linearity: Optional[np.ndarray] = None,
) -> float:
    """
    Compute vulnerability score for a point cloud region based on a genome.

    This applies the logic from perturbation_generator.py to determine
    how vulnerable a region would be if attacked with this genome.

    Args:
        points: Point cloud (N, 3) xyz coordinates
        genome: NSGA-III genome (17,) with values in [-1, 1]
        curvature: Pre-computed curvature per point (optional)
        linearity: Pre-computed linearity per point (optional)

    Returns:
        Vulnerability score in [0, 1] range
    """
    # Decode genome parameters from [-1, 1] to [0, 1]
    def decode(val):
        return (val + 1) / 2

    # Key attack parameters
    curvature_strength = decode(genome[4])  # curvature_targeting
    edge_strength = decode(genome[13])  # edge_attack
    geometric_distortion = abs(genome[12])  # Always active in best genomes
    noise_intensity = decode(genome[3])

    # Compute features if not provided
    if curvature is None or linearity is None:
        # Simple approximation: use point distances from centroid
        # Real implementation would compute eigenvalue-based features
        centroid = points.mean(axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)

        # Approximate curvature as variance of local distances
        curvature_approx = distances.std() / (distances.mean() + 1e-10)

        # Approximate linearity from point distribution
        # High spread in one direction = high linearity
        cov = np.cov(points.T)
        try:
            eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
            linearity_approx = (eigvals[0] - eigvals[1]) / (eigvals[0] + 1e-10)
        except Exception:
            linearity_approx = 0.0
    else:
        curvature_approx = curvature.mean()
        linearity_approx = linearity.mean()

    # Range-dependent vulnerability (geometric distortion affects far points more)
    ranges = np.linalg.norm(points, axis=1)
    range_factor = (ranges / (ranges.max() + 1e-10)).mean()

    # Combine into vulnerability score
    # Higher score = more vulnerable to attack
    vulnerability = (
        curvature_strength * curvature_approx * 0.3
        + edge_strength * linearity_approx * 0.2
        + geometric_distortion * range_factor * 0.4
        + noise_intensity * 0.1
    )

    # Clamp to [0, 1]
    return float(np.clip(vulnerability, 0, 1))


def compute_max_vulnerability(
    points: np.ndarray,
    pareto_set: np.ndarray,
    curvature: Optional[np.ndarray] = None,
    linearity: Optional[np.ndarray] = None,
) -> float:
    """
    Compute maximum vulnerability across all Pareto-optimal genomes.

    This identifies regions that are vulnerable to ANY of the optimal attacks.

    Args:
        points: Point cloud (N, 3) xyz coordinates
        pareto_set: Array of genomes (n_solutions, 17)
        curvature: Pre-computed curvature per point (optional)
        linearity: Pre-computed linearity per point (optional)

    Returns:
        Maximum vulnerability score in [0, 1] range
    """
    vulnerabilities = []
    for genome in pareto_set:
        v = compute_vulnerability_from_genome(points, genome, curvature, linearity)
        vulnerabilities.append(v)

    return float(np.max(vulnerabilities))


def compute_vulnerability_label(
    points: np.ndarray,
    pareto_set: np.ndarray,
    threshold: float = 0.5,
    curvature: Optional[np.ndarray] = None,
    linearity: Optional[np.ndarray] = None,
) -> int:
    """
    Compute binary vulnerability label for classification.

    Args:
        points: Point cloud (N, 3) xyz coordinates
        pareto_set: Array of genomes (n_solutions, 17)
        threshold: Vulnerability threshold (default 0.5)
        curvature: Pre-computed curvature per point (optional)
        linearity: Pre-computed linearity per point (optional)

    Returns:
        0 = CRITICAL (high vulnerability), 1 = NON_CRITICAL (low vulnerability)
    """
    max_vuln = compute_max_vulnerability(points, pareto_set, curvature, linearity)
    return 0 if max_vuln >= threshold else 1


if __name__ == "__main__":
    # Test with local data
    import sys

    # Try loading from mola-adversarial-nsga3 project
    nsga3_dir = Path("/home/francesca/mola-adversarial-nsga3/src/results/runs")

    if nsga3_dir.exists():
        print("Testing with NSGA-III results from mola-adversarial-nsga3...")
        weights = get_criticality_weights(nsga3_dir, run_id=10)
        print(f"\nFinal weights: {weights}")

        summary = get_pareto_front_summary(nsga3_dir, run_id=10)
        print(f"\nPareto front summary: {summary}")

        # Test vulnerability computation
        pareto_set = load_pareto_set(nsga3_dir, run_id=10)
        if pareto_set is not None:
            print(f"\nLoaded Pareto set: {pareto_set.shape}")

            # Create fake point cloud for testing
            fake_points = np.random.randn(1024, 3)
            max_vuln = compute_max_vulnerability(fake_points, pareto_set)
            label = compute_vulnerability_label(fake_points, pareto_set)
            print(f"Test vulnerability: {max_vuln:.4f}")
            print(f"Test label: {'CRITICAL' if label == 0 else 'NON_CRITICAL'}")
    else:
        print("NSGA-III results not found, testing fallback...")
        weights = get_criticality_weights(None)
        print(f"\nFallback weights: {weights}")
