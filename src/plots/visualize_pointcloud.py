#!/usr/bin/env python3
"""Visualize LiDAR point cloud data from Isaac Sim + MOLA SLAM."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

BASE_DIR = Path(__file__).parent.parent.parent  # mola-pointnet-verification/


def load_data():
    """Load raw point cloud data."""
    data = np.load(BASE_DIR / 'data/raw/frame_sequence.npy', allow_pickle=True)
    timestamps = np.load(BASE_DIR / 'data/raw/frame_sequence.timestamps.npy', allow_pickle=True)
    return data, timestamps


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
    ax1 = fig.add_subplot(131, projection='3d')
    scatter = ax1.scatter(x, y, z, c=intensity, cmap='viridis', s=1, alpha=0.6)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'Frame {frame_idx} - 3D View')
    plt.colorbar(scatter, ax=ax1, label='Intensity', shrink=0.5)

    # Top-down view (Bird's Eye View)
    ax2 = fig.add_subplot(132)
    scatter2 = ax2.scatter(x, y, c=intensity, cmap='viridis', s=1, alpha=0.6)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title(f'Frame {frame_idx} - Bird\'s Eye View')
    ax2.set_aspect('equal')
    plt.colorbar(scatter2, ax=ax2, label='Intensity')

    # Side view (height profile)
    ax3 = fig.add_subplot(133)
    scatter3 = ax3.scatter(x, z, c=intensity, cmap='viridis', s=1, alpha=0.6)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title(f'Frame {frame_idx} - Side View')
    plt.colorbar(scatter3, ax=ax3, label='Intensity')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_height_distribution(data, frame_idx=0, save_path=None):
    """Visualize height distribution to understand CRITICAL vs NON_CRITICAL."""
    frame = data[frame_idx]
    z = frame[:, 2]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Height histogram
    axes[0].hist(z, bins=100, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5m)')
    axes[0].axvline(x=-0.5, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Height Z (m)')
    axes[0].set_ylabel('Number of points')
    axes[0].set_title('Height Distribution')
    axes[0].legend()

    # Classification visualization
    critical = z[(z >= -0.5) & (z <= 0.5)]
    non_critical = z[(z < -0.5) | (z > 0.5)]

    axes[1].bar(['CRITICAL\n(obstacles)', 'NON_CRITICAL\n(ground/sky)'],
                [len(critical), len(non_critical)],
                color=['red', 'green'], alpha=0.7)
    axes[1].set_ylabel('Number of points')
    axes[1].set_title('Point Classification')

    # Add percentages
    total = len(z)
    axes[1].text(0, len(critical) + 1000, f'{100*len(critical)/total:.1f}%', ha='center', fontsize=12)
    axes[1].text(1, len(non_critical) + 1000, f'{100*len(non_critical)/total:.1f}%', ha='center', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_critical_regions(data, frame_idx=0, max_points=10000, save_path=None):
    """Visualize point cloud with CRITICAL regions highlighted."""
    frame = data[frame_idx]

    # Subsample
    if len(frame) > max_points:
        indices = np.random.choice(len(frame), max_points, replace=False)
        frame = frame[indices]

    x, y, z = frame[:, 0], frame[:, 1], frame[:, 2]

    # Classify points
    critical_mask = (z >= -0.5) & (z <= 0.5)

    fig = plt.figure(figsize=(12, 5))

    # 3D view with classification
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(x[~critical_mask], y[~critical_mask], z[~critical_mask],
                c='green', s=1, alpha=0.3, label='NON_CRITICAL')
    ax1.scatter(x[critical_mask], y[critical_mask], z[critical_mask],
                c='red', s=2, alpha=0.8, label='CRITICAL')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'Frame {frame_idx} - Critical Regions')
    ax1.legend()

    # Bird's eye view with classification
    ax2 = fig.add_subplot(122)
    ax2.scatter(x[~critical_mask], y[~critical_mask], c='green', s=1, alpha=0.3, label='NON_CRITICAL')
    ax2.scatter(x[critical_mask], y[critical_mask], c='red', s=2, alpha=0.8, label='CRITICAL')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title(f'Frame {frame_idx} - Bird\'s Eye (Critical in red)')
    ax2.set_aspect('equal')
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    print("Loading LiDAR data...")
    data, timestamps = load_data()

    print(f"\nDataset info:")
    print(f"  - Number of frames: {len(data)}")
    print(f"  - Points per frame: {data[0].shape[0]:,}")
    print(f"  - Features per point: {data[0].shape[1]} (x, y, z, intensity)")
    print(f"  - Time span: {timestamps[-1] - timestamps[0]:.2f} seconds")

    # Create visualizations
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)

    # Visualize first frame
    print("\n1. 3D Point Cloud (Frame 0)...")
    visualize_frame(data, frame_idx=0, save_path='results/pointcloud_3d.png')

    # Visualize middle frame
    mid_idx = len(data) // 2
    print(f"\n2. 3D Point Cloud (Frame {mid_idx})...")
    visualize_frame(data, frame_idx=mid_idx, save_path='results/pointcloud_3d_mid.png')

    # Height distribution
    print("\n3. Height distribution analysis...")
    visualize_height_distribution(data, frame_idx=0, save_path='results/height_distribution.png')

    # Critical regions
    print("\n4. Critical regions visualization...")
    visualize_critical_regions(data, frame_idx=0, save_path='results/critical_regions.png')

    print("\n" + "="*60)
    print("Done! Check results/ folder for images:")
    print("  - pointcloud_3d.png")
    print("  - pointcloud_3d_mid.png")
    print("  - height_distribution.png")
    print("  - critical_regions.png")
    print("="*60)


if __name__ == "__main__":
    main()
