"""
Data Preparation for PointNet Classification (GPU + Parallel).

Creates groups of N points from LiDAR frames for PointNet training.
Uses PyTorch GPU for fast eigenvalue computation and multiprocessing for frames.

Feature-Augmented Input Strategy:
Instead of hard-coding weights, we include geometric features as extra input channels
so the network can learn the optimal weighting during training.

Input format: (N, 7) where channels are:
- [0:3] xyz coordinates (normalized)
- [3] linearity - edge/line feature strength
- [4] curvature - surface curvature
- [5] density_var - local density variation (scanline attack vulnerability)
- [6] planarity - how planar the neighborhood is

Labels:
- CRITICAL (0): Regions with high SLAM vulnerability (learned from geometric features)
- NON_CRITICAL (1): Regions with low SLAM vulnerability

The features are derived from NSGA-III adversarial attack analysis:
- Curvature targeting: targets high-curvature points
- Scanline perturbation: exploits density variations
- Edge attack: targets edges/corners
- Temporal drift: affects structured regions
"""

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree

# Import NSGA-III integration for dynamic weights
from scripts.nsga3_integration import get_criticality_weights

# Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = min(mp.cpu_count(), 8)  # Limit workers

LABEL_CRITICAL = 0
LABEL_NON_CRITICAL = 1


def load_frame_sequence(data_path: Path) -> np.ndarray:
    """Load LiDAR frame sequence."""
    frames_path = data_path / "frame_sequence.npy"
    frames = np.load(frames_path, allow_pickle=True)
    print(f"Loaded {len(frames)} frames")
    return frames


def compute_local_features_gpu(points: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute local geometric features using GPU acceleration.

    Uses batched eigenvalue computation on GPU for speed.
    """
    n = len(points)
    xyz = points[:, :3]

    # Subsample for very large point clouds
    max_points = 50000
    if n > max_points:
        sample_idx = np.random.choice(n, max_points, replace=False)
        xyz_sample = xyz[sample_idx]
    else:
        sample_idx = np.arange(n)
        xyz_sample = xyz

    # Build KD-tree on CPU (scipy is fast for this)
    tree = cKDTree(xyz_sample)
    _, neighbors_idx = tree.query(xyz_sample, k=min(k + 1, len(xyz_sample)))

    # Batch compute covariance matrices on GPU
    xyz_tensor = torch.from_numpy(xyz_sample).float().to(DEVICE)

    linearity = np.zeros(len(xyz_sample))
    curvature = np.zeros(len(xyz_sample))

    # Process in batches for GPU memory efficiency
    batch_size = 5000
    for start in range(0, len(xyz_sample), batch_size):
        end = min(start + batch_size, len(xyz_sample))
        batch_idx = neighbors_idx[start:end]

        # Get neighbor points for batch
        batch_neighbors = xyz_tensor[batch_idx]  # (batch, k, 3)

        # Center each neighborhood
        batch_centered = batch_neighbors - batch_neighbors.mean(dim=1, keepdim=True)

        # Compute covariance matrices: (batch, 3, 3)
        # cov = X^T @ X / (n-1)
        batch_cov = torch.bmm(batch_centered.transpose(1, 2), batch_centered) / (k - 1)

        # Compute eigenvalues on GPU
        eigvals = torch.linalg.eigvalsh(batch_cov)  # (batch, 3), ascending order
        eigvals = eigvals.flip(dims=[1])  # Descending order

        # Compute features
        total = eigvals.sum(dim=1) + 1e-10
        lin = (eigvals[:, 0] - eigvals[:, 1]) / (eigvals[:, 0] + 1e-10)
        curv = eigvals[:, 2] / total

        linearity[start:end] = lin.cpu().numpy()
        curvature[start:end] = curv.cpu().numpy()

    # Map back to all points if subsampled
    if n > max_points:
        full_linearity = np.zeros(n)
        full_curvature = np.zeros(n)
        _, nearest = tree.query(xyz, k=1)
        full_linearity = linearity[nearest]
        full_curvature = curvature[nearest]
        return full_linearity, full_curvature

    return linearity, curvature


def process_single_frame(args: Tuple) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a single frame - designed for parallel execution.

    Args:
        args: Tuple of (frame_idx, frame, n_points, n_samples, weights)

    Returns:
        Tuple of (groups, labels) for this frame
        groups shape: (n_samples, n_points, 7) where 7 = xyz(3) + features(4)

    Features included per point:
    - linearity: edge/line strength (NSGA3 edge attack target)
    - curvature: surface curvature (NSGA3 curvature target)
    - density_var: density variation (NSGA3 scanline attack target)
    - planarity: how planar the region is (NSGA3 temporal drift target)
    """
    frame_idx, frame, n_points, n_samples, weights = args

    if len(frame) < n_points * 2:
        return np.array([]), np.array([])

    xyz = frame[:, :3]

    # Use CPU for subprocess - get all 4 features
    linearity, curvature, density_var, planarity = compute_local_features_cpu(frame, k=15)

    # Build KD-tree
    tree = cKDTree(xyz)

    groups = []
    scores = []

    # Sample seed points
    seed_indices = np.random.choice(len(xyz), min(n_samples * 2, len(xyz)), replace=False)

    for seed_idx in seed_indices:
        if len(groups) >= n_samples:
            break

        _, neighbor_idx = tree.query(xyz[seed_idx], k=n_points)

        if len(neighbor_idx) < n_points:
            continue

        # Extract xyz coordinates
        group_xyz = xyz[neighbor_idx].copy()

        # Normalize xyz
        group_xyz = group_xyz - group_xyz.mean(axis=0)
        max_dist = np.abs(group_xyz).max()
        if max_dist > 0:
            group_xyz = group_xyz / max_dist

        # Extract features for this group (per-point features)
        group_linearity = linearity[neighbor_idx]
        group_curvature = curvature[neighbor_idx]
        group_density_var = density_var[neighbor_idx]
        group_planarity = planarity[neighbor_idx]

        # Normalize features to [0, 1] range
        def normalize_feature(f):
            f_min, f_max = f.min(), f.max()
            if f_max - f_min > 1e-6:
                return (f - f_min) / (f_max - f_min)
            return np.zeros_like(f)

        group_linearity = normalize_feature(group_linearity)
        group_curvature = normalize_feature(group_curvature)
        group_density_var = normalize_feature(group_density_var)
        group_planarity = normalize_feature(group_planarity)

        # Combine xyz + features: (n_points, 7)
        group = np.column_stack(
            [
                group_xyz,  # (n_points, 3)
                group_linearity[:, np.newaxis],  # (n_points, 1)
                group_curvature[:, np.newaxis],  # (n_points, 1)
                group_density_var[:, np.newaxis],  # (n_points, 1)
                group_planarity[:, np.newaxis],  # (n_points, 1)
            ]
        )

        # Compute criticality score using mean features
        # Let the network learn the actual weights, this is just for labeling
        mean_linearity = group_linearity.mean()
        mean_curvature = group_curvature.mean()
        mean_density_var = group_density_var.mean()
        mean_nonplanarity = 1.0 - group_planarity.mean()

        # Use NSGA3-derived weights (passed as parameter)
        # These are dynamically loaded from Pareto set analysis
        criticality_score = (
            mean_linearity * weights["linearity"]
            + mean_curvature * weights["curvature"]
            + mean_density_var * weights["density_var"]
            + mean_nonplanarity * weights["nonplanarity"]
        )

        groups.append(group)
        scores.append(criticality_score)

    if len(groups) == 0:
        return np.array([]), np.array([])

    groups = np.array(groups, dtype=np.float32)
    scores = np.array(scores)

    # Use absolute threshold instead of median to preserve natural NSGA3 distribution
    # Features are normalized to [0,1], weights sum to ~1.0, so scores are in [0,1]
    # Threshold of 0.5 represents regions with above-average criticality
    threshold = 0.5
    labels = np.where(scores >= threshold, LABEL_CRITICAL, LABEL_NON_CRITICAL)

    return groups, labels.astype(np.int64)


def compute_local_features_cpu(
    points: np.ndarray, k: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    CPU version for parallel processing in subprocesses.
    Uses numpy for compatibility with multiprocessing.

    Returns:
        linearity: Edge/line feature strength
        curvature: Surface curvature
        density_var: Local density variation (scanline vulnerability)
        planarity: How planar the local neighborhood is
    """
    n = len(points)
    xyz = points[:, :3]

    # Subsample for speed
    max_points = 30000
    if n > max_points:
        sample_idx = np.random.choice(n, max_points, replace=False)
        xyz_sample = xyz[sample_idx]
    else:
        sample_idx = np.arange(n)
        xyz_sample = xyz

    tree = cKDTree(xyz_sample)
    distances, neighbors_idx = tree.query(xyz_sample, k=min(k + 1, len(xyz_sample)))

    linearity = np.zeros(len(xyz_sample))
    curvature = np.zeros(len(xyz_sample))
    planarity = np.zeros(len(xyz_sample))
    density_var = np.zeros(len(xyz_sample))

    # Compute local density variation (scanline attack vulnerability)
    mean_dist = distances[:, 1:].mean(axis=1)
    std_dist = distances[:, 1:].std(axis=1)
    density_var = std_dist / (mean_dist + 1e-10)  # coefficient of variation

    # Vectorized covariance computation
    for i in range(len(xyz_sample)):
        neighbors = xyz_sample[neighbors_idx[i]]
        centered = neighbors - neighbors.mean(axis=0)

        if len(centered) >= 3:
            cov = np.cov(centered.T)
            try:
                eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
                total = eigvals.sum() + 1e-10
                linearity[i] = (eigvals[0] - eigvals[1]) / (eigvals[0] + 1e-10)
                curvature[i] = eigvals[2] / total
                planarity[i] = (eigvals[1] - eigvals[2]) / (eigvals[0] + 1e-10)
            except Exception:
                pass

    if n > max_points:
        full_linearity = np.zeros(n)
        full_curvature = np.zeros(n)
        full_planarity = np.zeros(n)
        full_density_var = np.zeros(n)
        _, nearest = tree.query(xyz, k=1)
        full_linearity = linearity[nearest]
        full_curvature = curvature[nearest]
        full_planarity = planarity[nearest]
        full_density_var = density_var[nearest]
        return full_linearity, full_curvature, full_density_var, full_planarity

    return linearity, curvature, density_var, planarity


def extract_point_groups(
    frame: np.ndarray,
    n_points: int = 1024,  # Original PointNet uses 1024 points
    n_samples: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract groups of N points from a frame with geometric features.

    Each group is a local neighborhood with shape (n_points, 7):
    - [0:3] xyz coordinates (normalized)
    - [3:7] geometric features (linearity, curvature, density_var, planarity)

    Label is based on median split of NSGA3-derived criticality scores.
    """
    if len(frame) < n_points:
        return np.array([]), np.array([])

    xyz = frame[:, :3]

    # Compute features with GPU acceleration
    linearity, curvature = compute_local_features_gpu(frame, k=15)

    # Compute density variation (need distances from GPU function)
    tree = cKDTree(xyz[:50000] if len(xyz) > 50000 else xyz)
    distances, _ = tree.query(xyz[:50000] if len(xyz) > 50000 else xyz, k=16)
    mean_dist = distances[:, 1:].mean(axis=1)
    std_dist = distances[:, 1:].std(axis=1)
    density_var = std_dist / (mean_dist + 1e-10)

    # Planarity is approximated from linearity and curvature
    # planarity = (λ1 - λ2) / λ0, but we can estimate from what we have
    planarity = 1.0 - linearity - curvature
    planarity = np.clip(planarity, 0, 1)

    # Extend to full size if subsampled
    if len(xyz) > 50000:
        _, nearest = tree.query(xyz, k=1)
        density_var_full = density_var[nearest]
        density_var = density_var_full

    groups = []
    scores = []

    # Sample seed points
    seed_indices = np.random.choice(len(xyz), min(n_samples * 2, len(xyz)), replace=False)

    for seed_idx in seed_indices:
        if len(groups) >= n_samples:
            break

        # Get N nearest neighbors
        _, neighbor_idx = tree.query(xyz[seed_idx], k=n_points)

        if len(neighbor_idx) < n_points:
            continue

        # Extract xyz coordinates
        group_xyz = xyz[neighbor_idx].copy()

        # Normalize xyz
        group_xyz = group_xyz - group_xyz.mean(axis=0)
        max_dist = np.abs(group_xyz).max()
        if max_dist > 0:
            group_xyz = group_xyz / max_dist

        # Extract features
        group_linearity = linearity[neighbor_idx]
        group_curvature = curvature[neighbor_idx]
        group_density_var = density_var[neighbor_idx]
        group_planarity = planarity[neighbor_idx]

        # Normalize features to [0, 1]
        def normalize_feature(f):
            f_min, f_max = f.min(), f.max()
            if f_max - f_min > 1e-6:
                return (f - f_min) / (f_max - f_min)
            return np.zeros_like(f)

        group_linearity = normalize_feature(group_linearity)
        group_curvature = normalize_feature(group_curvature)
        group_density_var = normalize_feature(group_density_var)
        group_planarity = normalize_feature(group_planarity)

        # Combine xyz + features: (n_points, 7)
        group = np.column_stack(
            [
                group_xyz,
                group_linearity[:, np.newaxis],
                group_curvature[:, np.newaxis],
                group_density_var[:, np.newaxis],
                group_planarity[:, np.newaxis],
            ]
        )

        # NSGA3-derived criticality score for labeling
        mean_linearity = group_linearity.mean()
        mean_curvature = group_curvature.mean()
        mean_density_var = group_density_var.mean()
        mean_nonplanarity = 1.0 - group_planarity.mean()

        criticality_score = (
            mean_linearity * 0.0
            + mean_curvature * 0.1057
            + mean_density_var * 0.2369
            + mean_nonplanarity * 0.6574
        )

        groups.append(group)
        scores.append(criticality_score)

    if len(groups) == 0:
        return np.array([]), np.array([])

    groups = np.array(groups, dtype=np.float32)
    scores = np.array(scores)

    # Use absolute threshold instead of median to preserve natural NSGA3 distribution
    # Features are normalized to [0,1], weights sum to ~1.0, so scores are in [0,1]
    # Threshold of 0.5 represents regions with above-average criticality
    threshold = 0.5
    labels = np.where(scores >= threshold, LABEL_CRITICAL, LABEL_NON_CRITICAL)

    return groups, labels.astype(np.int64)


def prepare_pointnet_dataset(
    source_path: Path,
    output_path: Path,
    n_points: int = 1024,  # Original PointNet uses 1024 points
    n_train_samples: int = 10000,
    n_test_samples: int = 2000,
    seed: int = 42,
    parallel: bool = True,
):
    """Prepare dataset of point groups for PointNet."""
    np.random.seed(seed)

    print("=" * 60)
    print("Preparing PointNet Dataset (GPU + Parallel)")
    print(f"Device: {DEVICE}, Workers: {NUM_WORKERS}")
    print("=" * 60)

    # Load frames
    print("\n1. Loading frames...")
    frames = load_frame_sequence(source_path)

    # Load NSGA-III weights dynamically from Pareto set
    print("\n2. Loading NSGA-III weights from Pareto set...")
    nsga3_results_dir = Path("runs")
    weights = get_criticality_weights(nsga3_results_dir=nsga3_results_dir, run_id=10)
    print(f"   Loaded weights:")
    for feat, w in weights.items():
        print(f"     {feat}: {w:.4f}")

    # Collect samples
    samples_per_frame = (n_train_samples + n_test_samples) // len(frames) + 1
    print(f"\n3. Extracting point groups (n_points={n_points}, {samples_per_frame}/frame)...")

    all_groups = []
    all_labels = []

    if parallel and NUM_WORKERS > 1:
        # Parallel processing
        print(f"   Using {NUM_WORKERS} parallel workers...")

        # Prepare arguments for each frame (include weights)
        args_list = [
            (i, frames[i], n_points, samples_per_frame, weights) for i in range(len(frames))
        ]

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {
                executor.submit(process_single_frame, args): i for i, args in enumerate(args_list)
            }

            completed = 0
            for future in as_completed(futures):
                completed += 1
                if completed % 10 == 0:
                    print(f"   Completed {completed}/{len(frames)} frames...")

                try:
                    groups, labels = future.result()
                    if len(groups) > 0:
                        all_groups.append(groups)
                        all_labels.append(labels)
                except Exception as e:
                    print(f"   Error in frame: {e}")
    else:
        # Sequential processing (fallback)
        for i, frame in enumerate(frames):
            if i % 10 == 0:
                print(f"   Processing frame {i+1}/{len(frames)}...")

            groups, labels = extract_point_groups(
                frame,
                n_points=n_points,
                n_samples=samples_per_frame,
            )

            if len(groups) > 0:
                all_groups.append(groups)
                all_labels.append(labels)

    all_groups = np.concatenate(all_groups, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print(f"\n   Total samples: {len(all_groups)}")
    print(f"   CRITICAL: {(all_labels == LABEL_CRITICAL).sum()}")
    print(f"   NON_CRITICAL: {(all_labels == LABEL_NON_CRITICAL).sum()}")

    # Shuffle
    shuffle_idx = np.random.permutation(len(all_groups))
    all_groups = all_groups[shuffle_idx]
    all_labels = all_labels[shuffle_idx]

    # Split preserving natural NSGA3 distribution (no forced balancing)
    print("\n3. Creating train/test split (preserving NSGA3 distribution)...")

    # Simple train/test split maintaining natural class distribution
    n_total = len(all_groups)
    n_train = min(n_train_samples, int(n_total * 0.8))  # 80% train
    n_test = min(n_test_samples, n_total - n_train)  # Remaining for test

    train_groups = all_groups[:n_train]
    train_labels = all_labels[:n_train]
    test_groups = all_groups[n_train : n_train + n_test]
    test_labels = all_labels[n_train : n_train + n_test]

    print(f"   Train: {len(train_groups)} samples")
    print(f"     CRITICAL: {(train_labels == LABEL_CRITICAL).sum()} ({100 * (train_labels == LABEL_CRITICAL).mean():.1f}%)")
    print(f"     NON_CRITICAL: {(train_labels == LABEL_NON_CRITICAL).sum()} ({100 * (train_labels == LABEL_NON_CRITICAL).mean():.1f}%)")
    print(f"   Test: {len(test_groups)} samples")
    print(f"     CRITICAL: {(test_labels == LABEL_CRITICAL).sum()} ({100 * (test_labels == LABEL_CRITICAL).mean():.1f}%)")
    print(f"     NON_CRITICAL: {(test_labels == LABEL_NON_CRITICAL).sum()} ({100 * (test_labels == LABEL_NON_CRITICAL).mean():.1f}%)")

    # Save
    print("\n4. Saving dataset...")
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(output_path / "train_groups.npy", train_groups)
    np.save(output_path / "train_labels.npy", train_labels)
    np.save(output_path / "test_groups.npy", test_groups)
    np.save(output_path / "test_labels.npy", test_labels)

    stats = {
        "n_points": n_points,
        "n_train": len(train_groups),
        "n_test": len(test_groups),
        "n_classes": 2,
        "class_names": {0: "CRITICAL", 1: "NON_CRITICAL"},
        "train_distribution": {
            "critical": int((train_labels == LABEL_CRITICAL).sum()),
            "non_critical": int((train_labels == LABEL_NON_CRITICAL).sum()),
            "critical_pct": float((train_labels == LABEL_CRITICAL).mean() * 100),
        },
        "test_distribution": {
            "critical": int((test_labels == LABEL_CRITICAL).sum()),
            "non_critical": int((test_labels == LABEL_NON_CRITICAL).sum()),
            "critical_pct": float((test_labels == LABEL_CRITICAL).mean() * 100),
        },
        "nsga3_weights_used": True,
        "balanced": False,  # Natural NSGA3 distribution preserved
    }
    np.save(output_path / "dataset_stats.npy", stats)

    print(f"\nDataset saved to {output_path}")
    print(f"Shape: ({len(train_groups)}, {n_points}, 3)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="/home/francesca/mola-adversarial-nsga3/data")
    parser.add_argument("--output", type=str, default="data/pointnet")
    parser.add_argument("--n-points", type=int, default=1024)
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    prepare_pointnet_dataset(
        source_path=Path(args.source),
        output_path=Path(args.output),
        n_points=args.n_points,
        n_train_samples=args.n_train,
        n_test_samples=args.n_test,
        seed=args.seed,
    )
