"""Tests for verification functionality."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import CLASS_GROUND, CLASS_OBSTACLE, CLASS_OTHER, create_model


class TestVNNLIBGeneration:
    """Test VNN-LIB specification generation."""

    def test_generate_robustness_vnnlib(self):
        """Test robustness VNN-LIB generation."""
        from verify import generate_vnnlib_robustness

        x = np.array([0.5, 0.3, -0.2])
        epsilon = 0.01
        true_class = CLASS_GROUND

        vnnlib = generate_vnnlib_robustness(x, true_class, epsilon)

        # Check basic structure
        assert "(declare-const X_0 Real)" in vnnlib
        assert "(declare-const X_1 Real)" in vnnlib
        assert "(declare-const X_2 Real)" in vnnlib
        assert "(declare-const Y_0 Real)" in vnnlib
        assert "(declare-const Y_1 Real)" in vnnlib
        assert "(declare-const Y_2 Real)" in vnnlib

        # Check input bounds
        assert "(assert (>=" in vnnlib
        assert "(assert (<=" in vnnlib

    def test_generate_safety_vnnlib(self):
        """Test safety VNN-LIB generation."""
        from verify import generate_vnnlib_safety

        x = np.array([1.0, 2.0, 0.5])
        epsilon = 0.02

        vnnlib = generate_vnnlib_safety(x, epsilon)

        # Check declares
        assert "(declare-const X_0 Real)" in vnnlib
        assert "(declare-const Y_0 Real)" in vnnlib

        # Check safety constraint (GROUND >= OBSTACLE and GROUND >= OTHER)
        assert f"Y_{CLASS_GROUND}" in vnnlib
        assert f"Y_{CLASS_OBSTACLE}" in vnnlib

    def test_robustness_vnnlib_bounds(self):
        """Test that VNN-LIB bounds are correct."""
        from verify import generate_vnnlib_robustness

        x = np.array([1.0, 2.0, 3.0])
        epsilon = 0.1
        true_class = CLASS_OBSTACLE

        vnnlib = generate_vnnlib_robustness(x, true_class, epsilon)

        # Check that bounds include x +/- epsilon
        assert f"{x[0] - epsilon:.10f}" in vnnlib
        assert f"{x[0] + epsilon:.10f}" in vnnlib


class TestMockVerification:
    """Test mock verification (sampling-based)."""

    def test_mock_verification_robustness(self):
        """Test mock robustness verification."""
        from verify import mock_verification

        model = create_model()
        model.eval()

        # Create a simple input
        x = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Get true class
        with torch.no_grad():
            true_class = model.predict(torch.tensor(x).unsqueeze(0)).item()

        # Very small epsilon should likely be "verified"
        status, elapsed = mock_verification(
            model, x, epsilon=0.0001, property_type="robustness",
            true_class=true_class, num_samples=100
        )

        assert status in ["verified", "violated"]
        assert elapsed >= 0

    def test_mock_verification_safety(self):
        """Test mock safety verification."""
        from verify import mock_verification

        model = create_model()

        # Use an input classified as OBSTACLE
        # We'll search for one
        found_obstacle = False
        for _ in range(100):
            x = np.random.randn(3).astype(np.float32)
            with torch.no_grad():
                pred = model.predict(torch.tensor(x).unsqueeze(0)).item()
            if pred == CLASS_OBSTACLE:
                found_obstacle = True
                break

        if not found_obstacle:
            pytest.skip("Could not find OBSTACLE sample")

        status, elapsed = mock_verification(
            model, x, epsilon=0.001, property_type="safety",
            true_class=CLASS_OBSTACLE, num_samples=100
        )

        assert status in ["verified", "violated"]


class TestSampleSelection:
    """Test sample selection for verification."""

    def test_select_robustness_samples(self):
        """Test selecting samples for robustness verification."""
        from verify import select_verification_samples

        model = create_model()

        # Create synthetic test data
        points = np.random.randn(1000, 3).astype(np.float32)
        with torch.no_grad():
            labels = model.predict(torch.tensor(points)).numpy()

        selected_points, selected_labels, indices = select_verification_samples(
            points, labels, model, num_samples=50, property_type="robustness"
        )

        assert len(selected_points) == 50
        assert len(selected_labels) == 50
        assert len(indices) == 50

        # All selected should be correctly classified
        with torch.no_grad():
            preds = model.predict(torch.tensor(selected_points)).numpy()
        assert np.all(preds == selected_labels)

    def test_select_safety_samples(self):
        """Test selecting OBSTACLE samples for safety verification."""
        from verify import select_verification_samples

        model = create_model()

        # Create synthetic test data with points likely to be OBSTACLE
        # Based on heuristics: z in [0.2, 2.5]m and dist_xy < 15m
        # Generate points that match OBSTACLE characteristics
        n_samples = 1000
        points = np.zeros((n_samples, 3), dtype=np.float32)
        # x, y within 15m range
        points[:, 0] = np.random.uniform(-10, 10, n_samples)
        points[:, 1] = np.random.uniform(-10, 10, n_samples)
        # z in OBSTACLE range (0.2 to 2.5m)
        points[:, 2] = np.random.uniform(0.3, 2.0, n_samples)

        with torch.no_grad():
            labels = model.predict(torch.tensor(points)).numpy()

        # Count obstacles
        num_obstacles = np.sum(labels == CLASS_OBSTACLE)
        if num_obstacles < 10:
            pytest.skip("Not enough OBSTACLE samples in random data")

        num_to_select = min(20, num_obstacles)
        selected_points, selected_labels, indices = select_verification_samples(
            points, labels, model, num_samples=num_to_select, property_type="safety"
        )

        # All selected should be OBSTACLE
        assert np.all(selected_labels == CLASS_OBSTACLE)


class TestVerificationResult:
    """Test VerificationResult and VerificationSummary classes."""

    def test_verification_result_creation(self):
        """Test creating a VerificationResult."""
        from verify import VerificationResult

        result = VerificationResult(
            sample_idx=0,
            epsilon=0.01,
            property_type="robustness",
            true_label=CLASS_GROUND,
            status="verified",
            time_seconds=1.5,
        )

        assert result.sample_idx == 0
        assert result.epsilon == 0.01
        assert result.property_type == "robustness"
        assert result.status == "verified"

    def test_verification_summary(self):
        """Test VerificationSummary calculations."""
        from verify import VerificationSummary

        summary = VerificationSummary(
            property_type="robustness",
            epsilon=0.01,
            total_samples=100,
            verified=80,
            violated=15,
            timeout=5,
            total_time=100.0,
        )

        assert summary.verified_rate == 80.0
        assert summary.avg_time == 1.0

    def test_summary_to_dict(self):
        """Test VerificationSummary JSON serialization."""
        from verify import VerificationSummary

        summary = VerificationSummary(
            property_type="safety",
            epsilon=0.02,
            total_samples=50,
            verified=40,
            violated=8,
            timeout=2,
            total_time=50.0,
        )

        d = summary.to_dict()
        assert d["property_type"] == "safety"
        assert d["epsilon"] == 0.02
        assert d["verified_rate"] == 80.0
        assert d["avg_time"] == 1.0
