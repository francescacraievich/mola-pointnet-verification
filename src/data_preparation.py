"""
Data Preparation for MLP LiDAR Point Classification

This script processes raw LiDAR point cloud data from Isaac Sim:
1. Loads point clouds from frame_sequence.npy
2. Applies heuristic labeling based on geometry
3. Subsamples points to manageable size
4. Normalizes data and saves normalization parameters
5. Splits into train/test sets
6. Saves processed dataset

Classes:
    0 - GROUND: points with z < -0.3m (below robot level)
    1 - OBSTACLE: points with 0.2m < z < 2.5m AND distance_xy < 15m
    2 - OTHER: everything else (sky, far points, noise)
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import argparse


# Label definitions
LABEL_GROUND = 0
LABEL_OBSTACLE = 1
LABEL_OTHER = 2

# Heuristic thresholds
GROUND_Z_THRESHOLD = -0.3  # meters
OBSTACLE_Z_MIN = 0.2  # meters
OBSTACLE_Z_MAX = 2.5  # meters
OBSTACLE_XY_MAX_DIST = 15.0  # meters


def load_raw_data(data_path: Path) -> np.ndarray:
    """
    Load raw point cloud data from numpy file.

    Args:
        data_path: Path to frame_sequence.npy

    Returns:
        Array of shape (n_frames,) containing point clouds
        Each frame has shape (n_points, 4) with (x, y, z, intensity)
    """
    print(f"Loading data from {data_path}...")
    data = np.load(data_path, allow_pickle=True)
    print(f"Loaded {len(data)} frames")
    return data


def apply_heuristic_labels(points: np.ndarray) -> np.ndarray:
    """
    Apply geometric heuristics to label points.

    Args:
        points: Array of shape (n_points, 4) with (x, y, z, intensity)

    Returns:
        Labels array of shape (n_points,) with values 0, 1, or 2
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    labels = np.full(len(points), LABEL_OTHER, dtype=np.int64)

    # GROUND: z < -0.3m
    ground_mask = z < GROUND_Z_THRESHOLD
    labels[ground_mask] = LABEL_GROUND

    # OBSTACLE: 0.2m < z < 2.5m AND distance_xy < 15m
    dist_xy = np.sqrt(x**2 + y**2)
    obstacle_mask = (
        (z > OBSTACLE_Z_MIN) &
        (z < OBSTACLE_Z_MAX) &
        (dist_xy < OBSTACLE_XY_MAX_DIST)
    )
    labels[obstacle_mask] = LABEL_OBSTACLE

    return labels


def subsample_points(
    points: np.ndarray,
    labels: np.ndarray,
    n_samples: int,
    balanced: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subsample points from a frame.

    Args:
        points: Array of shape (n_points, 4)
        labels: Array of shape (n_points,)
        n_samples: Target number of samples
        balanced: If True, try to balance classes

    Returns:
        Tuple of (subsampled_points, subsampled_labels)
    """
    if balanced:
        # Sample equally from each class (or as much as available)
        samples_per_class = n_samples // 3
        indices = []

        for label in [LABEL_GROUND, LABEL_OBSTACLE, LABEL_OTHER]:
            class_indices = np.where(labels == label)[0]
            if len(class_indices) > 0:
                n_take = min(samples_per_class, len(class_indices))
                selected = np.random.choice(class_indices, n_take, replace=False)
                indices.extend(selected)

        indices = np.array(indices)
    else:
        # Random sampling
        if len(points) > n_samples:
            indices = np.random.choice(len(points), n_samples, replace=False)
        else:
            indices = np.arange(len(points))

    return points[indices], labels[indices]


def process_all_frames(
    raw_data: np.ndarray,
    samples_per_frame: int = 6000,
    balanced: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process all frames: label and subsample.

    Args:
        raw_data: Array of frames from load_raw_data
        samples_per_frame: Number of points to sample per frame
        balanced: Whether to balance classes during sampling

    Returns:
        Tuple of (all_points, all_labels) concatenated from all frames
    """
    all_points = []
    all_labels = []

    print(f"Processing {len(raw_data)} frames...")
    for i, frame in enumerate(raw_data):
        # Apply heuristic labels
        labels = apply_heuristic_labels(frame)

        # Subsample
        points_sub, labels_sub = subsample_points(
            frame[:, :3],  # Only x, y, z (no intensity)
            labels,
            samples_per_frame,
            balanced
        )

        all_points.append(points_sub)
        all_labels.append(labels_sub)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(raw_data)} frames")

    # Concatenate all
    points = np.vstack(all_points)
    labels = np.concatenate(all_labels)

    print(f"Total points after processing: {len(points)}")
    return points, labels


def compute_normalization_params(points: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute normalization parameters (mean and std).

    Args:
        points: Array of shape (n_points, 3)

    Returns:
        Dictionary with 'mean' and 'std' arrays
    """
    mean = points.mean(axis=0)
    std = points.std(axis=0)
    # Avoid division by zero
    std[std < 1e-6] = 1.0
    return {'mean': mean, 'std': std}


def normalize_points(
    points: np.ndarray,
    params: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Normalize points using precomputed parameters.

    Args:
        points: Array of shape (n_points, 3)
        params: Dictionary with 'mean' and 'std'

    Returns:
        Normalized points
    """
    return (points - params['mean']) / params['std']


def split_data(
    points: np.ndarray,
    labels: np.ndarray,
    test_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets.

    Args:
        points: Array of shape (n_points, 3)
        labels: Array of shape (n_points,)
        test_ratio: Fraction of data for testing
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_points, train_labels, test_points, test_labels)
    """
    np.random.seed(random_seed)
    n_samples = len(points)
    indices = np.random.permutation(n_samples)

    n_test = int(n_samples * test_ratio)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return (
        points[train_indices],
        labels[train_indices],
        points[test_indices],
        labels[test_indices]
    )


def print_dataset_stats(
    train_labels: np.ndarray,
    test_labels: np.ndarray
) -> None:
    """Print statistics about the dataset."""
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)

    label_names = {0: "GROUND", 1: "OBSTACLE", 2: "OTHER"}

    print(f"\nTraining set: {len(train_labels)} samples")
    for label, name in label_names.items():
        count = np.sum(train_labels == label)
        pct = 100 * count / len(train_labels)
        print(f"  {name}: {count} ({pct:.1f}%)")

    print(f"\nTest set: {len(test_labels)} samples")
    for label, name in label_names.items():
        count = np.sum(test_labels == label)
        pct = 100 * count / len(test_labels)
        print(f"  {name}: {count} ({pct:.1f}%)")

    print("="*50 + "\n")


def save_processed_data(
    output_dir: Path,
    train_points: np.ndarray,
    train_labels: np.ndarray,
    test_points: np.ndarray,
    test_labels: np.ndarray,
    norm_params: Dict[str, np.ndarray]
) -> None:
    """
    Save processed data to files.

    Args:
        output_dir: Directory to save files
        train_points, train_labels: Training data
        test_points, test_labels: Test data
        norm_params: Normalization parameters
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "train_points.npy", train_points.astype(np.float32))
    np.save(output_dir / "train_labels.npy", train_labels.astype(np.int64))
    np.save(output_dir / "test_points.npy", test_points.astype(np.float32))
    np.save(output_dir / "test_labels.npy", test_labels.astype(np.int64))
    np.save(output_dir / "normalization_params.npy", norm_params)

    print(f"Saved processed data to {output_dir}/")
    print(f"  - train_points.npy: {train_points.shape}")
    print(f"  - train_labels.npy: {train_labels.shape}")
    print(f"  - test_points.npy: {test_points.shape}")
    print(f"  - test_labels.npy: {test_labels.shape}")
    print(f"  - normalization_params.npy: mean={norm_params['mean']}, std={norm_params['std']}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare LiDAR point cloud data for MLP training"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/frame_sequence.npy",
        help="Path to raw frame_sequence.npy"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--samples-per-frame",
        type=int,
        default=6000,
        help="Number of points to sample per frame"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of data for testing"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Disable class balancing during sampling"
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Paths
    input_path = Path(args.input)
    output_dir = Path(args.output)

    # Load raw data
    raw_data = load_raw_data(input_path)

    # Process all frames
    points, labels = process_all_frames(
        raw_data,
        samples_per_frame=args.samples_per_frame,
        balanced=not args.no_balance
    )

    # Compute normalization parameters on ALL data before split
    # (to ensure consistent normalization)
    norm_params = compute_normalization_params(points)

    # Normalize
    points_normalized = normalize_points(points, norm_params)

    # Split
    train_points, train_labels, test_points, test_labels = split_data(
        points_normalized,
        labels,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )

    # Print stats
    print_dataset_stats(train_labels, test_labels)

    # Save
    save_processed_data(
        output_dir,
        train_points,
        train_labels,
        test_points,
        test_labels,
        norm_params
    )

    print("Data preparation complete!")


if __name__ == "__main__":
    main()
