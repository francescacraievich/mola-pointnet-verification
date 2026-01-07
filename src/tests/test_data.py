#!/usr/bin/env python3
"""Tests for data loading and preprocessing."""

import sys
from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path(__file__).parent.parent.parent


class TestDataLoading:
    """Test suite for data loading."""

    @pytest.fixture
    def data_dir(self):
        return BASE_DIR / "data" / "pointnet"

    def test_data_directory_exists(self, data_dir):
        """Test that data directory exists."""
        assert data_dir.exists(), f"Data directory not found at {data_dir}"

    def test_train_data_exists(self, data_dir):
        """Test that training data files exist."""
        train_groups = data_dir / "train_groups.npy"
        train_labels = data_dir / "train_labels.npy"

        assert train_groups.exists(), "train_groups.npy not found"
        assert train_labels.exists(), "train_labels.npy not found"

    def test_test_data_exists(self, data_dir):
        """Test that test data files exist."""
        test_groups = data_dir / "test_groups.npy"
        test_labels = data_dir / "test_labels.npy"

        assert test_groups.exists(), "test_groups.npy not found"
        assert test_labels.exists(), "test_labels.npy not found"

    def test_load_train_data(self, data_dir):
        """Test loading training data."""
        if not (data_dir / "train_groups.npy").exists():
            pytest.skip("Training data not found")

        train_groups = np.load(data_dir / "train_groups.npy")
        train_labels = np.load(data_dir / "train_labels.npy")

        # Check shapes
        assert len(train_groups.shape) == 3  # (samples, points, features)
        assert len(train_labels.shape) == 1  # (samples,)
        assert train_groups.shape[0] == train_labels.shape[0]

    def test_load_test_data(self, data_dir):
        """Test loading test data."""
        if not (data_dir / "test_groups.npy").exists():
            pytest.skip("Test data not found")

        test_groups = np.load(data_dir / "test_groups.npy")
        test_labels = np.load(data_dir / "test_labels.npy")

        # Check shapes
        assert len(test_groups.shape) == 3
        assert len(test_labels.shape) == 1
        assert test_groups.shape[0] == test_labels.shape[0]

    def test_label_values(self, data_dir):
        """Test that labels are binary (0 or 1)."""
        if not (data_dir / "train_labels.npy").exists():
            pytest.skip("Training data not found")

        train_labels = np.load(data_dir / "train_labels.npy")
        test_labels = np.load(data_dir / "test_labels.npy")

        assert set(np.unique(train_labels)).issubset({0, 1})
        assert set(np.unique(test_labels)).issubset({0, 1})

    def test_data_not_nan(self, data_dir):
        """Test that data contains no NaN values."""
        if not (data_dir / "train_groups.npy").exists():
            pytest.skip("Training data not found")

        train_groups = np.load(data_dir / "train_groups.npy")
        test_groups = np.load(data_dir / "test_groups.npy")

        assert not np.isnan(train_groups).any(), "Training data contains NaN"
        assert not np.isnan(test_groups).any(), "Test data contains NaN"

    def test_xyz_coordinates(self, data_dir):
        """Test that XYZ coordinates are in reasonable range."""
        if not (data_dir / "test_groups.npy").exists():
            pytest.skip("Test data not found")

        test_groups = np.load(data_dir / "test_groups.npy")

        # Extract XYZ (first 3 columns)
        xyz = test_groups[:, :, :3]

        # Check that coordinates are not all zeros
        assert not np.allclose(xyz, 0), "XYZ coordinates are all zeros"

        # Check reasonable range (normalized data typically in [-1, 1] or similar)
        assert xyz.min() > -1000, "XYZ min value seems too small"
        assert xyz.max() < 1000, "XYZ max value seems too large"


class TestDataStats:
    """Test data statistics."""

    @pytest.fixture
    def data_dir(self):
        return BASE_DIR / "data" / "pointnet"

    def test_class_balance(self, data_dir):
        """Test class distribution in data."""
        if not (data_dir / "train_labels.npy").exists():
            pytest.skip("Training data not found")

        train_labels = np.load(data_dir / "train_labels.npy")

        class_counts = np.bincount(train_labels)

        # Both classes should have samples
        assert len(class_counts) == 2, "Should have exactly 2 classes"
        assert class_counts[0] > 0, "Class 0 (CRITICAL) has no samples"
        assert class_counts[1] > 0, "Class 1 (NON_CRITICAL) has no samples"

        # Print distribution for info
        print(f"\nClass distribution: CRITICAL={class_counts[0]}, NON_CRITICAL={class_counts[1]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
