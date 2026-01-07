#!/usr/bin/env python3
"""Visualize LiDAR point cloud data with NSGA-III based criticality classification."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

BASE_DIR = Path(__file__).parent.parent.parent  # mola-pointnet-verification/
sys.path.insert(0, str(BASE_DIR / "src"))

# Try to import NSGA3 integration for criticality weights
try:
    from scripts.nsga3_integration import get_criticality_weights

    NSGA3_AVAILABLE = True
except ImportError:
    NSGA3_AVAILABLE = False
    print("Warning: NSGA3 integration not available, using default weights")


def get_nsga3_weights(nsga3_dir: Path = None, run_id: int = 10):
    """Get NSGA-III optimized weights for criticality calculation."""
    if NSGA3_AVAILABLE and nsga3_dir and nsga3_dir.exists():
        return get_criticality_weights(nsga3_results_dir=nsga3_dir, run_id=run_id)
    else:
        # Default weights if NSGA3 not available
        return {
            "linearity": 0.25,
            "curvature": 0.25,
            "density_var": 0.25,
            "nonplanarity": 0.25,
        }


def compute_geometric_features(points):
    """
    Compute geometric features for a group of points.

    Returns: linearity, curvature, density_var, planarity
    """
    if len(points) < 3:
        return 0.0, 0.0, 0.0, 1.0

    # Covariance-based features
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)

    try:
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
        e1, e2, e3 = eigenvalues[0], eigenvalues[1], eigenvalues[2]

        # Geometric features
        total = e1 + e2 + e3 + 1e-10
        linearity = (e1 - e2) / (e1 + 1e-10)
        planarity = (e2 - e3) / (e1 + 1e-10)
        curvature = e3 / total

        # Density variance (simplified)
        distances = np.linalg.norm(centered, axis=1)
        density_var = np.var(distances) / (np.mean(distances) + 1e-10)

    except np.linalg.LinAlgError:
        linearity, curvature, density_var, planarity = 0.0, 0.0, 0.0, 1.0

    return linearity, curvature, density_var, planarity


def compute_criticality_score(points, weights):
    """
    Compute NSGA-III based criticality score for a group of points.

    Score >= 0.5 -> CRITICAL
    Score < 0.5 -> NON_CRITICAL
    """
    linearity, curvature, density_var, planarity = compute_geometric_features(points)

    score = (
        linearity * weights["linearity"]
        + curvature * weights["curvature"]
        + density_var * weights["density_var"]
        + (1.0 - planarity) * weights["nonplanarity"]
    )

    return score


def load_data():
    """Load raw point cloud data."""
    data = np.load(BASE_DIR / "data/raw/frame_sequence.npy", allow_pickle=True)
    timestamps = np.load(BASE_DIR / "data/raw/frame_sequence.timestamps.npy", allow_pickle=True)
    return data, timestamps


def load_preprocessed_data():
    """Load preprocessed data with labels from data/pointnet/."""
    test_groups = np.load(BASE_DIR / "data/pointnet/test_groups.npy")
    test_labels = np.load(BASE_DIR / "data/pointnet/test_labels.npy")
    return test_groups, test_labels


def visualize_frame(data, frame_idx=0, max_points=5000, save_path=None):
    """Visualize a single frame as 3D scatter plot."""
    frame = data[frame_idx]

    # Subsample for visualization (too many points is slow)
    if len(frame) > max_points:
        indices = np.random.choice(len(frame), max_points, replace=False)
        frame = frame[indices]

    x, y, z, intensity = frame[:, 0], frame[:, 1], frame[:, 2], frame[:, 3]

    fig = plt.figure(figsize=(14, 5))

    # 3D view
    ax1 = fig.add_subplot(131, projection="3d")
    scatter = ax1.scatter(x, y, z, c=intensity, cmap="viridis", s=1, alpha=0.6)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title(f"Frame {frame_idx} - 3D View")
    plt.colorbar(scatter, ax=ax1, label="Intensity", shrink=0.5)

    # Top-down view (Bird's Eye View)
    ax2 = fig.add_subplot(132)
    scatter2 = ax2.scatter(x, y, c=intensity, cmap="viridis", s=1, alpha=0.6)
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title(f"Frame {frame_idx} - Bird's Eye View")
    ax2.set_aspect("equal")
    plt.colorbar(scatter2, ax=ax2, label="Intensity")

    # Side view (height profile)
    ax3 = fig.add_subplot(133)
    scatter3 = ax3.scatter(x, z, c=intensity, cmap="viridis", s=1, alpha=0.6)
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Z (m)")
    ax3.set_title(f"Frame {frame_idx} - Side View")
    plt.colorbar(scatter3, ax=ax3, label="Intensity")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_nsga3_features(data, frame_idx=0, weights=None, save_path=None):
    """Visualize NSGA-III geometric features distribution."""
    if weights is None:
        weights = get_nsga3_weights()

    frame = data[frame_idx]
    xyz = frame[:, :3]

    # Divide frame into groups and compute features
    n_groups = 100
    group_size = len(xyz) // n_groups

    features = {"linearity": [], "curvature": [], "density_var": [], "nonplanarity": [], "score": []}

    for i in range(n_groups):
        start = i * group_size
        end = start + group_size
        group = xyz[start:end]

        lin, curv, dens, plan = compute_geometric_features(group)
        score = compute_criticality_score(group, weights)

        features["linearity"].append(lin)
        features["curvature"].append(curv)
        features["density_var"].append(dens)
        features["nonplanarity"].append(1.0 - plan)
        features["score"].append(score)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Feature histograms
    axes[0, 0].hist(features["linearity"], bins=30, edgecolor="black", alpha=0.7, color="blue")
    axes[0, 0].set_xlabel("Linearity")
    axes[0, 0].set_title(f"Linearity (weight={weights['linearity']:.2f})")

    axes[0, 1].hist(features["curvature"], bins=30, edgecolor="black", alpha=0.7, color="orange")
    axes[0, 1].set_xlabel("Curvature")
    axes[0, 1].set_title(f"Curvature (weight={weights['curvature']:.2f})")

    axes[0, 2].hist(features["density_var"], bins=30, edgecolor="black", alpha=0.7, color="green")
    axes[0, 2].set_xlabel("Density Variance")
    axes[0, 2].set_title(f"Density Var (weight={weights['density_var']:.2f})")

    axes[1, 0].hist(features["nonplanarity"], bins=30, edgecolor="black", alpha=0.7, color="purple")
    axes[1, 0].set_xlabel("Non-planarity")
    axes[1, 0].set_title(f"Non-planarity (weight={weights['nonplanarity']:.2f})")

    # Criticality score distribution
    axes[1, 1].hist(features["score"], bins=30, edgecolor="black", alpha=0.7, color="red")
    axes[1, 1].axvline(x=0.5, color="black", linestyle="--", linewidth=2, label="Threshold")
    axes[1, 1].set_xlabel("Criticality Score")
    axes[1, 1].set_title("NSGA-III Criticality Score")
    axes[1, 1].legend()

    # Classification bar
    scores = np.array(features["score"])
    n_critical = (scores >= 0.5).sum()
    n_non_critical = (scores < 0.5).sum()

    axes[1, 2].bar(
        ["CRITICAL\n(score >= 0.5)", "NON_CRITICAL\n(score < 0.5)"],
        [n_critical, n_non_critical],
        color=["red", "green"],
        alpha=0.7,
    )
    axes[1, 2].set_ylabel("Number of groups")
    axes[1, 2].set_title("Classification Result")

    plt.suptitle(f"NSGA-III Based Criticality Analysis - Frame {frame_idx}", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_preprocessed_samples(save_path=None):
    """Visualize samples from preprocessed dataset with their labels."""
    test_groups, test_labels = load_preprocessed_data()

    # Select some CRITICAL and NON_CRITICAL samples
    critical_idx = np.where(test_labels == 0)[0][:3]
    non_critical_idx = np.where(test_labels == 1)[0][:3]

    fig = plt.figure(figsize=(15, 8))

    # Plot CRITICAL samples
    for i, idx in enumerate(critical_idx):
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        sample = test_groups[idx][:, :3]  # x, y, z
        ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], c="red", s=5, alpha=0.8)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"CRITICAL #{idx}")

    # Plot NON_CRITICAL samples
    for i, idx in enumerate(non_critical_idx):
        ax = fig.add_subplot(2, 3, i + 4, projection="3d")
        sample = test_groups[idx][:, :3]
        ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], c="green", s=5, alpha=0.8)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"NON_CRITICAL #{idx}")

    plt.suptitle("Preprocessed Dataset Samples (64 points each)", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_dataset_statistics(save_path=None):
    """Visualize dataset statistics."""
    test_groups, test_labels = load_preprocessed_data()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Class distribution
    n_critical = (test_labels == 0).sum()
    n_non_critical = (test_labels == 1).sum()

    axes[0].bar(
        ["CRITICAL", "NON_CRITICAL"],
        [n_critical, n_non_critical],
        color=["red", "green"],
        alpha=0.7,
    )
    axes[0].set_ylabel("Number of samples")
    axes[0].set_title(f"Test Set Class Distribution\n(Total: {len(test_labels)})")

    # Add percentages
    total = len(test_labels)
    axes[0].text(0, n_critical + 10, f"{100*n_critical/total:.1f}%", ha="center", fontsize=12)
    axes[0].text(1, n_non_critical + 10, f"{100*n_non_critical/total:.1f}%", ha="center", fontsize=12)

    # Point coordinate distributions
    all_points = test_groups[:, :, :3].reshape(-1, 3)
    axes[1].hist(all_points[:, 0], bins=50, alpha=0.5, label="X", color="red")
    axes[1].hist(all_points[:, 1], bins=50, alpha=0.5, label="Y", color="green")
    axes[1].hist(all_points[:, 2], bins=50, alpha=0.5, label="Z", color="blue")
    axes[1].set_xlabel("Coordinate value (m)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Point Coordinate Distribution")
    axes[1].legend()

    # Sample info
    info_text = f"""Dataset Information:

- Test samples: {len(test_labels)}
- Points per sample: {test_groups.shape[1]}
- Features per point: {test_groups.shape[2]}
- CRITICAL (label=0): {n_critical} ({100*n_critical/total:.1f}%)
- NON_CRITICAL (label=1): {n_non_critical} ({100*n_non_critical/total:.1f}%)

Classification based on NSGA-III
optimized geometric features:
- Linearity
- Curvature
- Density variance
- Non-planarity
"""
    axes[2].text(0.1, 0.5, info_text, transform=axes[2].transAxes, fontsize=11, verticalalignment="center", family="monospace")
    axes[2].axis("off")
    axes[2].set_title("Dataset Info")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    print("=" * 60)
    print("LiDAR Point Cloud Visualization with NSGA-III Criticality")
    print("=" * 60)

    # Get NSGA3 weights
    nsga3_dir = BASE_DIR.parent / "mola-adversarial-nsga3/src/results/runs"
    weights = get_nsga3_weights(nsga3_dir, run_id=10)
    print(f"\nNSGA-III Weights:")
    print(f"  - Linearity:    {weights['linearity']:.4f}")
    print(f"  - Curvature:    {weights['curvature']:.4f}")
    print(f"  - Density Var:  {weights['density_var']:.4f}")
    print(f"  - Nonplanarity: {weights['nonplanarity']:.4f}")

    # Load raw data
    print("\nLoading LiDAR data...")
    data, timestamps = load_data()

    print(f"\nRaw Dataset info:")
    print(f"  - Number of frames: {len(data)}")
    print(f"  - Points per frame: {data[0].shape[0]:,}")
    print(f"  - Features per point: {data[0].shape[1]} (x, y, z, intensity)")
    print(f"  - Time span: {timestamps[-1] - timestamps[0]:.2f} seconds")

    # Create results directory
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    # Generate visualizations
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)

    # 1. 3D Point Cloud
    print("\n1. 3D Point Cloud (Frame 0)...")
    visualize_frame(data, frame_idx=0, save_path=results_dir / "pointcloud_3d.png")

    # 2. Middle frame
    mid_idx = len(data) // 2
    print(f"\n2. 3D Point Cloud (Frame {mid_idx})...")
    visualize_frame(data, frame_idx=mid_idx, save_path=results_dir / "pointcloud_3d_mid.png")

    # 3. NSGA-III features analysis
    print("\n3. NSGA-III geometric features analysis...")
    visualize_nsga3_features(data, frame_idx=0, weights=weights, save_path=results_dir / "nsga3_features.png")

    # 4. Preprocessed samples
    print("\n4. Preprocessed dataset samples...")
    visualize_preprocessed_samples(save_path=results_dir / "preprocessed_samples.png")

    # 5. Dataset statistics
    print("\n5. Dataset statistics...")
    visualize_dataset_statistics(save_path=results_dir / "dataset_statistics.png")

    print("\n" + "=" * 60)
    print("Done! Check results/ folder for images:")
    print("  - pointcloud_3d.png")
    print("  - pointcloud_3d_mid.png")
    print("  - nsga3_features.png")
    print("  - preprocessed_samples.png")
    print("  - dataset_statistics.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
