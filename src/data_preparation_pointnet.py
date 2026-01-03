"""
Data Preparation for PointNet Classification (GPU + Parallel).

Creates groups of N points from LiDAR frames for PointNet training.
Uses PyTorch GPU for fast eigenvalue computation and multiprocessing for frames.

Labels:
- CRITICAL (0): Regions with edges, corners, high-curvature features
- NON_CRITICAL (1): Flat surfaces, sparse regions
"""

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree

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
        args: Tuple of (frame_idx, frame, n_points, n_samples)

    Returns:
        Tuple of (groups, labels) for this frame
    """
    frame_idx, frame, n_points, n_samples = args

    if len(frame) < n_points * 2:
        return np.array([]), np.array([])

    xyz = frame[:, :3]

    # Use CPU for subprocess (GPU in main process)
    linearity, curvature = compute_local_features_cpu(frame, k=15)

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

        group = xyz[neighbor_idx].copy()
        group = group - group.mean(axis=0)
        max_dist = np.abs(group).max()
        if max_dist > 0:
            group = group / max_dist

        group_linearity = linearity[neighbor_idx].mean()
        group_curvature = curvature[neighbor_idx].mean()
        criticality_score = group_linearity * 0.7 + group_curvature * 0.3

        groups.append(group)
        scores.append(criticality_score)

    if len(groups) == 0:
        return np.array([]), np.array([])

    groups = np.array(groups, dtype=np.float32)
    scores = np.array(scores)

    threshold = np.median(scores)
    labels = np.where(scores >= threshold, LABEL_CRITICAL, LABEL_NON_CRITICAL)

    return groups, labels.astype(np.int64)


def compute_local_features_cpu(points: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    CPU version for parallel processing in subprocesses.
    Uses numpy for compatibility with multiprocessing.
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
    _, neighbors_idx = tree.query(xyz_sample, k=min(k + 1, len(xyz_sample)))

    linearity = np.zeros(len(xyz_sample))
    curvature = np.zeros(len(xyz_sample))

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
            except:
                pass

    if n > max_points:
        full_linearity = np.zeros(n)
        full_curvature = np.zeros(n)
        _, nearest = tree.query(xyz, k=1)
        full_linearity = linearity[nearest]
        full_curvature = curvature[nearest]
        return full_linearity, full_curvature

    return linearity, curvature


def extract_point_groups(
    frame: np.ndarray,
    n_points: int = 512,
    n_samples: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract groups of N points from a frame.

    Each group is a local neighborhood. Label is based on median split
    of criticality scores for balanced classes.
    """
    if len(frame) < n_points:
        return np.array([]), np.array([])

    xyz = frame[:, :3]

    # Compute features with GPU acceleration
    linearity, curvature = compute_local_features_gpu(frame, k=15)

    # Build KD-tree
    tree = cKDTree(xyz)

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

        # Extract group
        group = xyz[neighbor_idx].copy()

        # Center and normalize
        group = group - group.mean(axis=0)
        max_dist = np.abs(group).max()
        if max_dist > 0:
            group = group / max_dist

        # Compute criticality score
        group_linearity = linearity[neighbor_idx].mean()
        group_curvature = curvature[neighbor_idx].mean()
        criticality_score = group_linearity * 0.7 + group_curvature * 0.3

        groups.append(group)
        scores.append(criticality_score)

    if len(groups) == 0:
        return np.array([]), np.array([])

    groups = np.array(groups, dtype=np.float32)
    scores = np.array(scores)

    # Use median for balanced split
    threshold = np.median(scores)
    labels = np.where(scores >= threshold, LABEL_CRITICAL, LABEL_NON_CRITICAL)

    return groups, labels.astype(np.int64)


def prepare_pointnet_dataset(
    source_path: Path,
    output_path: Path,
    n_points: int = 512,
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

    # Collect samples
    samples_per_frame = (n_train_samples + n_test_samples) // len(frames) + 1
    print(f"\n2. Extracting point groups (n_points={n_points}, {samples_per_frame}/frame)...")

    all_groups = []
    all_labels = []

    if parallel and NUM_WORKERS > 1:
        # Parallel processing
        print(f"   Using {NUM_WORKERS} parallel workers...")

        # Prepare arguments for each frame
        args_list = [(i, frames[i], n_points, samples_per_frame) for i in range(len(frames))]

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

    # Split with balancing
    print("\n3. Creating balanced train/test split...")

    critical_mask = all_labels == LABEL_CRITICAL
    non_critical_mask = all_labels == LABEL_NON_CRITICAL

    critical_groups = all_groups[critical_mask]
    critical_labels = all_labels[critical_mask]
    non_critical_groups = all_groups[non_critical_mask]
    non_critical_labels = all_labels[non_critical_mask]

    n_per_class_train = min(n_train_samples // 2, len(critical_groups) - n_test_samples // 2)
    n_per_class_test = min(n_test_samples // 2, len(critical_groups) - n_per_class_train)

    train_groups = np.concatenate(
        [critical_groups[:n_per_class_train], non_critical_groups[:n_per_class_train]], axis=0
    )
    train_labels = np.concatenate(
        [critical_labels[:n_per_class_train], non_critical_labels[:n_per_class_train]], axis=0
    )

    test_groups = np.concatenate(
        [
            critical_groups[n_per_class_train : n_per_class_train + n_per_class_test],
            non_critical_groups[n_per_class_train : n_per_class_train + n_per_class_test],
        ],
        axis=0,
    )
    test_labels = np.concatenate(
        [
            critical_labels[n_per_class_train : n_per_class_train + n_per_class_test],
            non_critical_labels[n_per_class_train : n_per_class_train + n_per_class_test],
        ],
        axis=0,
    )

    # Final shuffle
    train_shuffle = np.random.permutation(len(train_groups))
    train_groups = train_groups[train_shuffle]
    train_labels = train_labels[train_shuffle]

    test_shuffle = np.random.permutation(len(test_groups))
    test_groups = test_groups[test_shuffle]
    test_labels = test_labels[test_shuffle]

    print(f"   Train: {len(train_groups)} samples")
    print(f"   Test: {len(test_groups)} samples")

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
    }
    np.save(output_path / "dataset_stats.npy", stats)

    print(f"\nDataset saved to {output_path}")
    print(f"Shape: ({len(train_groups)}, {n_points}, 3)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="/home/francesca/mola-adversarial-nsga3/data")
    parser.add_argument("--output", type=str, default="data/pointnet")
    parser.add_argument("--n-points", type=int, default=512)
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
