#!/usr/bin/env python3
"""Animate point cloud samples showing CRITICAL vs NON_CRITICAL classification."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree

BASE_DIR = Path(__file__).parent.parent.parent  # mola-pointnet-verification/
sys.path.insert(0, str(BASE_DIR / "src"))

# Try to import NSGA3 weights
try:
    from scripts.nsga3_integration import get_criticality_weights

    NSGA3_AVAILABLE = True
except ImportError:
    NSGA3_AVAILABLE = False


def get_nsga3_weights():
    """Get NSGA-III optimized weights."""
    if NSGA3_AVAILABLE:
        nsga3_dir = BASE_DIR.parent / "mola-adversarial-nsga3/src/results/runs"
        if nsga3_dir.exists():
            return get_criticality_weights(nsga3_results_dir=nsga3_dir, run_id=10)
    return {"linearity": 0.25, "curvature": 0.25, "density_var": 0.25, "nonplanarity": 0.25}


def compute_geometric_features(points, k=15):
    """
    Compute geometric features from eigenvalues of the covariance matrix.

    Based on Weinmann et al. (2015) "Semantic point cloud interpretation
    based on optimal neighborhoods, relevant features and efficient classifiers".

    The eigenvalues e1 >= e2 >= e3 represent the variance along the 3 principal axes:
    - e1: variance along the direction of maximum spread
    - e2: variance along the second principal direction
    - e3: variance along the direction of minimum spread

    Returns:
        linearity: High when points form a LINE (e1 >> e2 ≈ e3)
                   Example: edges, poles, cables
        curvature: High when points are scattered in 3D (e1 ≈ e2 ≈ e3)
                   Example: corners, vegetation, complex objects
        density_var: Measures how uniformly distributed the points are
                     High = irregular spacing, Low = uniform spacing
        planarity: High when points lie on a PLANE (e1 ≈ e2 >> e3)
                   Example: walls, floors, roofs
    """
    if len(points) < 4:
        return 0.0, 0.0, 0.0, 1.0

    xyz = points[:, :3] if points.shape[1] > 3 else points

    # Compute covariance matrix of centered points
    centered = xyz - xyz.mean(axis=0)
    cov = np.cov(centered.T)

    try:
        # Eigenvalues of covariance = variance along principal axes
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending: e1 >= e2 >= e3
        e1, e2, e3 = eigenvalues[0], eigenvalues[1], eigenvalues[2]

        total = e1 + e2 + e3 + 1e-10

        # Linearity: (e1 - e2) / e1 - high if one direction dominates (line-like)
        linearity = (e1 - e2) / (e1 + 1e-10)

        # Planarity: (e2 - e3) / e1 - high if two directions dominate (plane-like)
        planarity = (e2 - e3) / (e1 + 1e-10)

        # Curvature (sphericity): e3 / sum - high if all directions equal (scattered)
        curvature = e3 / total

        # Density variance: how irregular is the point spacing?
        distances = np.linalg.norm(centered, axis=1)
        density_var = np.var(distances) / (np.mean(distances) + 1e-10)

    except np.linalg.LinAlgError:
        linearity, curvature, density_var, planarity = 0.0, 0.0, 0.0, 1.0

    return linearity, curvature, density_var, planarity


def compute_criticality_score(points, weights, penalize_floor=True):
    """Compute NSGA-III based criticality score.

    For visualization, we penalize floor regions (low Z variance, high planarity)
    and favor edges/borders (high linearity, vertical structures).
    """
    linearity, curvature, density_var, planarity = compute_geometric_features(points)

    xyz = points[:, :3] if points.shape[1] > 3 else points

    # Base score from NSGA-III weights
    score = (
        linearity * weights["linearity"]
        + curvature * weights["curvature"]
        + density_var * weights["density_var"]
        + (1.0 - planarity) * weights["nonplanarity"]
    )

    if penalize_floor:
        # Penalize flat floor regions: low Z range + high planarity
        z_range = xyz[:, 2].max() - xyz[:, 2].min()

        # Floor detection: small Z range AND high planarity
        is_floor_like = (z_range < 0.3) and (planarity > 0.7)

        if is_floor_like:
            score *= 0.3  # Heavily penalize floor patterns

        # Boost edges/borders: high linearity OR significant Z variation
        if linearity > 0.6 or z_range > 0.5:
            score *= 1.5  # Boost edge-like structures

        # Boost vertical structures (walls, obstacles)
        z_extent_ratio = z_range / (np.sqrt(np.var(xyz[:, 0]) + np.var(xyz[:, 1])) + 1e-6)
        if z_extent_ratio > 0.8:  # Tall relative to horizontal spread
            score *= 1.3

    return score


def load_data():
    """Load preprocessed test data with labels."""
    test_groups = np.load(BASE_DIR / "data/pointnet/test_groups.npy")
    test_labels = np.load(BASE_DIR / "data/pointnet/test_labels.npy")
    return test_groups, test_labels


def load_raw_pointcloud(frame_idx=0):
    """Load original raw LiDAR point cloud."""
    data = np.load(BASE_DIR / "data/raw/frame_sequence.npy", allow_pickle=True)
    return data[frame_idx][:, :3]  # x, y, z only


def load_all_frames():
    """Load all raw LiDAR frames."""
    return np.load(BASE_DIR / "data/raw/frame_sequence.npy", allow_pickle=True)


def create_animation(n_samples=50, interval=500, save_path=None, max_bg_points=5000):
    """
    Create animated visualization of point cloud samples with original LiDAR background.

    Args:
        n_samples: Number of samples to include in animation
        interval: Milliseconds between frames
        save_path: Path to save animation (GIF or MP4)
        max_bg_points: Max points to show in background (for performance)
    """
    print("Loading data...")
    test_groups, test_labels = load_data()

    print("Loading raw LiDAR point cloud for background...")
    raw_pc = load_raw_pointcloud()

    # Subsample background for performance
    if len(raw_pc) > max_bg_points:
        bg_indices = np.random.choice(len(raw_pc), max_bg_points, replace=False)
        raw_pc = raw_pc[bg_indices]
    print(f"  Background points: {len(raw_pc)}")

    # Select samples (mix of CRITICAL and NON_CRITICAL)
    critical_idx = np.where(test_labels == 0)[0]
    non_critical_idx = np.where(test_labels == 1)[0]

    # Take equal samples from each class
    n_each = n_samples // 2
    selected_critical = critical_idx[:n_each]
    selected_non_critical = non_critical_idx[:n_each]
    selected_indices = np.concatenate([selected_critical, selected_non_critical])
    np.random.shuffle(selected_indices)

    print(f"Selected {len(selected_indices)} samples")
    print(f"  - CRITICAL: {len(selected_critical)}")
    print(f"  - NON_CRITICAL: {len(selected_non_critical)}")

    # Setup figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Set axis limits based on raw point cloud (full scene)
    x_min, x_max = raw_pc[:, 0].min(), raw_pc[:, 0].max()
    y_min, y_max = raw_pc[:, 1].min(), raw_pc[:, 1].max()
    z_min, z_max = raw_pc[:, 2].min(), raw_pc[:, 2].max()

    # Add some padding
    padding = 0.05
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    # Animation update function
    def update(frame):
        idx = selected_indices[frame]
        sample = test_groups[idx][:, :3]
        label = test_labels[idx]

        # Clear and redraw
        ax.clear()

        # Draw background point cloud in dark gray (visible but not overwhelming)
        ax.scatter(
            raw_pc[:, 0],
            raw_pc[:, 1],
            raw_pc[:, 2],
            c="dimgray",
            s=3,
            alpha=0.4,
            label="LiDAR scene",
        )

        # Color based on label
        if label == 0:
            color = "red"
            label_str = "CRITICAL"
        else:
            color = "green"
            label_str = "NON_CRITICAL"

        # Draw sample points (smaller but bright)
        ax.scatter(
            sample[:, 0],
            sample[:, 1],
            sample[:, 2],
            c=color,
            s=40,
            alpha=1.0,
            edgecolors="black",
            linewidths=0.5,
            label=f"{label_str} (64 pts)",
        )

        # Set limits
        ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
        ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
        ax.set_zlim(z_min - padding * z_range, z_max + padding * z_range)

        ax.set_xlabel("X (m)", fontsize=10)
        ax.set_ylabel("Y (m)", fontsize=10)
        ax.set_zlabel("Z (m)", fontsize=10)

        ax.set_title(
            f"Sample #{idx} - {label_str}\n"
            f"64 points highlighted on LiDAR scene ({frame + 1}/{len(selected_indices)})",
            fontsize=13,
            fontweight="bold",
            color=color,
        )

        ax.legend(loc="upper left", fontsize=9)

        # Rotate view for 3D effect
        ax.view_init(elev=25, azim=frame * 4)

        return []

    print(f"Creating animation with {len(selected_indices)} frames...")
    anim = animation.FuncAnimation(
        fig, update, frames=len(selected_indices), interval=interval, blit=False
    )

    if save_path:
        save_path = Path(save_path)
        print(f"Saving animation to {save_path}...")

        if save_path.suffix == ".gif":
            writer = animation.PillowWriter(fps=1000 / interval)
            anim.save(save_path, writer=writer)
        elif save_path.suffix == ".mp4":
            writer = animation.FFMpegWriter(fps=1000 / interval)
            anim.save(save_path, writer=writer)
        else:
            # Default to GIF
            save_path = save_path.with_suffix(".gif")
            writer = animation.PillowWriter(fps=1000 / interval)
            anim.save(save_path, writer=writer)

        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()
    return anim


def create_comparison_animation(n_samples=20, interval=800, save_path=None, max_bg_points=3000):
    """
    Create side-by-side animation comparing CRITICAL vs NON_CRITICAL samples
    with LiDAR background.
    """
    print("Loading data...")
    test_groups, test_labels = load_data()

    print("Loading raw LiDAR point cloud for background...")
    raw_pc = load_raw_pointcloud()

    # Subsample background for performance
    if len(raw_pc) > max_bg_points:
        bg_indices = np.random.choice(len(raw_pc), max_bg_points, replace=False)
        raw_pc = raw_pc[bg_indices]
    print(f"  Background points: {len(raw_pc)}")

    # Get indices for each class
    critical_idx = np.where(test_labels == 0)[0][:n_samples]
    non_critical_idx = np.where(test_labels == 1)[0][:n_samples]

    n_frames = min(len(critical_idx), len(non_critical_idx))

    print(f"Creating comparison animation with {n_frames} frame pairs")

    # Setup figure with two subplots
    fig = plt.figure(figsize=(18, 8))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    # Get axis limits from raw point cloud
    x_min, x_max = raw_pc[:, 0].min(), raw_pc[:, 0].max()
    y_min, y_max = raw_pc[:, 1].min(), raw_pc[:, 1].max()
    z_min, z_max = raw_pc[:, 2].min(), raw_pc[:, 2].max()

    padding = 0.05
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    def setup_ax(ax, title, color):
        ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
        ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
        ax.set_zlim(z_min - padding * z_range, z_max + padding * z_range)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(title, fontsize=11, fontweight="bold", color=color)

    def update(frame):
        # Clear both axes
        ax1.clear()
        ax2.clear()

        # Draw background on both (darker and larger points)
        ax1.scatter(raw_pc[:, 0], raw_pc[:, 1], raw_pc[:, 2], c="dimgray", s=2, alpha=0.35)
        ax2.scatter(raw_pc[:, 0], raw_pc[:, 1], raw_pc[:, 2], c="dimgray", s=2, alpha=0.35)

        # CRITICAL sample
        crit_idx = critical_idx[frame]
        crit_sample = test_groups[crit_idx][:, :3]
        ax1.scatter(
            crit_sample[:, 0],
            crit_sample[:, 1],
            crit_sample[:, 2],
            c="red",
            s=35,
            alpha=1.0,
            edgecolors="black",
            linewidths=0.4,
        )
        setup_ax(ax1, f"CRITICAL - Sample #{crit_idx}", "red")

        # NON_CRITICAL sample
        non_crit_idx = non_critical_idx[frame]
        non_crit_sample = test_groups[non_crit_idx][:, :3]
        ax2.scatter(
            non_crit_sample[:, 0],
            non_crit_sample[:, 1],
            non_crit_sample[:, 2],
            c="limegreen",
            s=35,
            alpha=1.0,
            edgecolors="black",
            linewidths=0.4,
        )
        setup_ax(ax2, f"NON_CRITICAL - Sample #{non_crit_idx}", "green")

        # Sync rotation
        ax1.view_init(elev=25, azim=frame * 6)
        ax2.view_init(elev=25, azim=frame * 6)

        fig.suptitle(
            f"CRITICAL vs NON_CRITICAL on LiDAR Scene ({frame + 1}/{n_frames})\n"
            f"Gray: full LiDAR scan | Colored: 64-point sample (NSGA-III classification)",
            fontsize=13,
        )

        return []

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)

    if save_path:
        save_path = Path(save_path)
        print(f"Saving to {save_path}...")

        if save_path.suffix == ".gif":
            writer = animation.PillowWriter(fps=1000 / interval)
        else:
            save_path = save_path.with_suffix(".gif")
            writer = animation.PillowWriter(fps=1000 / interval)

        anim.save(save_path, writer=writer)
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()
    return anim


def create_scene_regions_animation(
    n_regions=40, points_per_region=64, interval=700, save_path=None, max_bg_points=8000
):
    """
    Create animation showing CRITICAL vs NON_CRITICAL regions extracted from
    the ENTIRE LiDAR scene (not just preprocessed samples).

    Regions are sampled from different parts of the scene and classified
    in real-time using NSGA-III geometric features.
    Floor points are excluded from classification to avoid circular MOLA lines.
    """
    print("Loading raw LiDAR data...")
    raw_pc_full = load_raw_pointcloud(frame_idx=0)
    print(f"  Total points in scene: {len(raw_pc_full)}")

    # Filter out floor points for classification (keep full scene for background)
    floor_z = np.percentile(raw_pc_full[:, 2], 10)  # Bottom 10% is floor
    print(f"  Floor Z level: {floor_z:.2f}m")
    above_floor_mask = raw_pc_full[:, 2] > floor_z + 0.1  # 10cm above floor
    raw_pc = raw_pc_full[above_floor_mask]
    print(f"  Points above floor (for classification): {len(raw_pc)} / {len(raw_pc_full)}")

    print("Getting NSGA-III weights...")
    weights = get_nsga3_weights()
    print(f"  Weights: {weights}")

    # Build KD-tree for neighbor queries (only on non-floor points)
    print("Building spatial index...")
    tree = cKDTree(raw_pc)

    # Sample seed points from across the entire scene (not just center)
    print(f"Extracting {n_regions} regions from across the scene (excluding floor)...")

    # Divide scene into grid and sample from each cell
    x_min, x_max = raw_pc[:, 0].min(), raw_pc[:, 0].max()
    y_min, y_max = raw_pc[:, 1].min(), raw_pc[:, 1].max()

    # Create grid of seed points across the scene
    n_grid = int(np.sqrt(n_regions))
    x_edges = np.linspace(x_min, x_max, n_grid + 1)
    y_edges = np.linspace(y_min, y_max, n_grid + 1)

    regions = []
    scores = []

    for i in range(n_grid):
        for j in range(n_grid):
            # Find points in this grid cell
            mask = (
                (raw_pc[:, 0] >= x_edges[i])
                & (raw_pc[:, 0] < x_edges[i + 1])
                & (raw_pc[:, 1] >= y_edges[j])
                & (raw_pc[:, 1] < y_edges[j + 1])
            )
            cell_points = raw_pc[mask]

            if len(cell_points) < points_per_region:
                continue

            # Pick a random seed point in this cell
            seed_idx = np.random.randint(len(cell_points))
            seed_point = cell_points[seed_idx]

            # Get nearest neighbors from full scene
            _, neighbor_idx = tree.query(seed_point, k=points_per_region)
            region_points = raw_pc[neighbor_idx]

            # Compute criticality score
            score = compute_criticality_score(region_points, weights)

            regions.append(region_points)
            scores.append(score)

    # Use median as threshold (like the preprocessing script does for balance)
    scores = np.array(scores)
    threshold = np.median(scores)
    labels = np.where(scores >= threshold, 0, 1)  # 0=CRITICAL, 1=NON_CRITICAL
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  Threshold (median): {threshold:.4f}")

    regions = np.array(regions)

    n_critical = (labels == 0).sum()
    n_non_critical = (labels == 1).sum()
    print(f"  Extracted {len(regions)} regions:")
    print(f"    - CRITICAL: {n_critical}")
    print(f"    - NON_CRITICAL: {n_non_critical}")

    # Subsample background (use FULL scene including floor for context)
    if len(raw_pc_full) > max_bg_points:
        bg_idx = np.random.choice(len(raw_pc_full), max_bg_points, replace=False)
        bg_points = raw_pc_full[bg_idx]
    else:
        bg_points = raw_pc_full

    # Setup figure - make 3D plot fill the space
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection="3d")

    # Minimize margins so point cloud fills the figure
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.95)

    # Use full scene limits for axes
    x_min_full, x_max_full = raw_pc_full[:, 0].min(), raw_pc_full[:, 0].max()
    y_min_full, y_max_full = raw_pc_full[:, 1].min(), raw_pc_full[:, 1].max()
    z_min_full, z_max_full = raw_pc_full[:, 2].min(), raw_pc_full[:, 2].max()

    padding = 0.05
    x_range = x_max_full - x_min_full
    y_range = y_max_full - y_min_full
    z_range = z_max_full - z_min_full

    def update(frame):
        ax.clear()

        # Draw background
        ax.scatter(bg_points[:, 0], bg_points[:, 1], bg_points[:, 2], c="dimgray", s=1.5, alpha=0.3)

        # Draw current region
        region = regions[frame]
        label = labels[frame]

        if label == 0:
            color = "red"
            label_str = "CRITICAL"
        else:
            color = "limegreen"
            label_str = "NON_CRITICAL"

        ax.scatter(
            region[:, 0],
            region[:, 1],
            region[:, 2],
            c=color,
            s=60,
            alpha=1.0,
            edgecolors="black",
            linewidths=0.5,
        )

        # Set limits (use full scene dimensions)
        ax.set_xlim(x_min_full - padding * x_range, x_max_full + padding * x_range)
        ax.set_ylim(y_min_full - padding * y_range, y_max_full + padding * y_range)
        ax.set_zlim(z_min_full - padding * z_range, z_max_full + padding * z_range)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        ax.set_title(
            f"{label_str} Region - NSGA-III Classification",
            fontsize=18,
            fontweight="bold",
            color=color,
        )

        ax.view_init(elev=30, azim=frame * 5)

        return []

    print(f"Creating animation with {len(regions)} frames...")
    anim = animation.FuncAnimation(fig, update, frames=len(regions), interval=interval, blit=False)

    if save_path:
        save_path = Path(save_path)
        print(f"Saving to {save_path}...")
        writer = animation.PillowWriter(fps=1000 / interval)
        anim.save(save_path, writer=writer)
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()
    return anim


def create_full_scene_classification(
    save_path=None, n_regions=200, points_per_region=64, floor_z_threshold=-0.3
):
    """
    Create a STATIC image showing ALL regions classified across the scene.
    Red = CRITICAL, Green = NON_CRITICAL.

    Excludes floor regions (points below floor_z_threshold) from classification.
    """
    print("Loading raw LiDAR data...")
    raw_pc_full = load_raw_pointcloud(frame_idx=0)

    # Filter out floor points for classification (keep for background)
    floor_z = np.percentile(raw_pc_full[:, 2], 10)  # Bottom 10% is floor
    print(f"  Floor Z level: {floor_z:.2f}m")

    # Points above floor for classification
    above_floor_mask = raw_pc_full[:, 2] > floor_z + 0.1  # 10cm above floor
    raw_pc = raw_pc_full[above_floor_mask]
    print(f"  Points above floor: {len(raw_pc)} / {len(raw_pc_full)}")

    print("Getting NSGA-III weights...")
    weights = get_nsga3_weights()

    print("Building spatial index...")
    tree = cKDTree(raw_pc)

    # Sample many regions across the scene (from non-floor points)
    print(f"Classifying {n_regions} regions across scene (excluding floor)...")

    x_min, x_max = raw_pc[:, 0].min(), raw_pc[:, 0].max()
    y_min, y_max = raw_pc[:, 1].min(), raw_pc[:, 1].max()

    # Random seed points across the scene (from non-floor points)
    seed_indices = np.random.choice(len(raw_pc), min(n_regions, len(raw_pc)), replace=False)

    all_regions = []
    all_scores = []

    for seed_idx in seed_indices:
        seed_point = raw_pc[seed_idx]
        _, neighbor_idx = tree.query(seed_point, k=points_per_region)
        region_points = raw_pc[neighbor_idx]

        score = compute_criticality_score(region_points, weights)
        all_regions.append(region_points)
        all_scores.append(score)

    # Use median as threshold for balanced classification
    all_scores = np.array(all_scores)
    threshold = np.median(all_scores)
    print(f"  Score range: [{all_scores.min():.4f}, {all_scores.max():.4f}]")
    print(f"  Threshold (median): {threshold:.4f}")

    critical_points = []
    non_critical_points = []

    for region, score in zip(all_regions, all_scores):
        if score >= threshold:
            critical_points.append(region)
        else:
            non_critical_points.append(region)

    critical_points = np.vstack(critical_points) if critical_points else np.array([]).reshape(0, 3)
    non_critical_points = (
        np.vstack(non_critical_points) if non_critical_points else np.array([]).reshape(0, 3)
    )

    print(f"  CRITICAL regions: {len(critical_points) // points_per_region}")
    print(f"  NON_CRITICAL regions: {len(non_critical_points) // points_per_region}")

    # Plot
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection="3d")

    # Background - show FULL scene (including floor) for context
    bg_idx = np.random.choice(len(raw_pc_full), min(15000, len(raw_pc_full)), replace=False)
    ax.scatter(
        raw_pc_full[bg_idx, 0],
        raw_pc_full[bg_idx, 1],
        raw_pc_full[bg_idx, 2],
        c="dimgray",
        s=1.5,
        alpha=0.25,
        label="LiDAR scene",
    )

    # Critical regions in red
    if len(critical_points) > 0:
        ax.scatter(
            critical_points[:, 0],
            critical_points[:, 1],
            critical_points[:, 2],
            c="red",
            s=20,
            alpha=0.85,
            label="CRITICAL",
        )

    # Non-critical regions in green
    if len(non_critical_points) > 0:
        ax.scatter(
            non_critical_points[:, 0],
            non_critical_points[:, 1],
            non_critical_points[:, 2],
            c="limegreen",
            s=20,
            alpha=0.85,
            label="NON_CRITICAL",
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(
        "NSGA-III Criticality Classification Across Full LiDAR Scene",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper left")
    ax.view_init(elev=35, azim=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    print("=" * 60)
    print("Point Cloud Animation - CRITICAL vs NON_CRITICAL")
    print("=" * 60)

    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    # NEW: Scene-wide regions animation (samples from ENTIRE scene)
    print("\n1. Creating scene-wide regions animation...")
    create_scene_regions_animation(
        n_regions=36,  # 6x6 grid
        points_per_region=64,
        interval=800,
        save_path=results_dir / "scene_regions_animation.gif",
    )

    # NEW: Full scene classification (static image)
    print("\n2. Creating full scene classification image...")
    create_full_scene_classification(
        n_regions=150, points_per_region=64, save_path=results_dir / "full_scene_classification.png"
    )

    print("\n" + "=" * 60)
    print("Done! Check results/ folder:")
    print("  - scene_regions_animation.gif (animated, regions from entire scene)")
    print("  - full_scene_classification.png (static, all regions colored)")
    print("=" * 60)


if __name__ == "__main__":
    main()
