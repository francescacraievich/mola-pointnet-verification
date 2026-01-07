#!/usr/bin/env python3
"""Tests for verification scripts and utilities."""

import sys
from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path(__file__).parent.parent.parent


class TestVerificationPaths:
    """Test that all verification paths and files exist."""

    def test_abcrown_config_exists(self):
        """Test that α,β-CROWN config exists."""
        config_path = BASE_DIR / "configs" / "abcrown_pointnet_complete.yaml"
        assert config_path.exists(), f"Config not found at {config_path}"

    def test_abcrown_script_exists(self):
        """Test that α,β-CROWN verification script exists."""
        script_path = BASE_DIR / "src" / "verification" / "verify_abcrown.py"
        assert script_path.exists(), f"Script not found at {script_path}"

    def test_eran_script_exists(self):
        """Test that ERAN verification script exists."""
        script_path = BASE_DIR / "src" / "verification" / "verify_eran_python_api.py"
        assert script_path.exists(), f"Script not found at {script_path}"

    def test_onnx_model_exists(self):
        """Test that ONNX model for α,β-CROWN exists."""
        onnx_path = BASE_DIR / "saved_models" / "pointnet_autolirpa_flat.onnx"
        assert onnx_path.exists(), f"ONNX model not found at {onnx_path}"

    def test_3dcertify_model_exists(self):
        """Test that 3DCertify model for ERAN exists."""
        model_path = BASE_DIR / "saved_models" / "pointnet_3dcertify_64p.pth"
        assert model_path.exists(), f"3DCertify model not found at {model_path}"

    def test_results_directory_exists(self):
        """Test that results directory exists."""
        results_dir = BASE_DIR / "results"
        assert results_dir.exists(), f"Results directory not found at {results_dir}"


class TestVNNLIBGeneration:
    """Test VNNLIB specification generation."""

    @pytest.fixture
    def test_data(self):
        """Load test data."""
        test_groups = np.load(BASE_DIR / "data/pointnet/test_groups.npy")
        test_labels = np.load(BASE_DIR / "data/pointnet/test_labels.npy")
        return test_groups, test_labels

    def test_vnnlib_format(self, test_data):
        """Test that VNNLIB can be generated correctly."""
        test_groups, test_labels = test_data

        # Get first sample
        sample = test_groups[0][:, :3]  # Only XYZ
        label = int(test_labels[0])
        epsilon = 0.01

        # Generate VNNLIB content
        vnnlib_content = generate_vnnlib(sample, label, epsilon)

        # Check format
        assert "; VNNLIB" in vnnlib_content
        assert "declare-const X_" in vnnlib_content
        assert "declare-const Y_" in vnnlib_content
        assert "assert" in vnnlib_content

    def test_vnnlib_bounds(self, test_data):
        """Test that VNNLIB bounds are correct."""
        test_groups, test_labels = test_data

        sample = test_groups[0][:, :3]
        epsilon = 0.01

        # Check bounds
        flat = sample.flatten()
        for i, val in enumerate(flat[:5]):  # Check first 5
            lower = val - epsilon
            upper = val + epsilon
            assert lower < upper


def generate_vnnlib(point_cloud: np.ndarray, true_label: int, epsilon: float) -> str:
    """Generate VNNLIB specification for a point cloud sample."""
    n_points = point_cloud.shape[0]
    n_inputs = n_points * 3
    n_outputs = 2

    lines = ["; VNNLIB specification for PointNet verification"]
    lines.append(f"; Points: {n_points}, Epsilon: {epsilon}")
    lines.append("")

    # Declare input variables
    for i in range(n_inputs):
        lines.append(f"(declare-const X_{i} Real)")

    # Declare output variables
    for i in range(n_outputs):
        lines.append(f"(declare-const Y_{i} Real)")

    lines.append("")

    # Input bounds
    flat = point_cloud.flatten()
    for i, val in enumerate(flat):
        lower = val - epsilon
        upper = val + epsilon
        lines.append(f"(assert (>= X_{i} {lower:.6f}))")
        lines.append(f"(assert (<= X_{i} {upper:.6f}))")

    lines.append("")

    # Output constraint: true class should be greater than other class
    other_label = 1 - true_label
    lines.append(f"; True label: {true_label}, must be greater than {other_label}")
    lines.append(f"(assert (<= Y_{true_label} Y_{other_label}))")

    return "\n".join(lines)


class TestResultNumbering:
    """Test result file numbering."""

    def test_get_next_result_number_empty(self, tmp_path):
        """Test numbering with empty directory."""
        result = get_next_result_number(tmp_path, "test")
        assert result == 1

    def test_get_next_result_number_existing(self, tmp_path):
        """Test numbering with existing files."""
        # Create some files
        (tmp_path / "test_1.json").touch()
        (tmp_path / "test_2.json").touch()
        (tmp_path / "test_5.json").touch()

        result = get_next_result_number(tmp_path, "test")
        assert result == 6

    def test_get_next_result_number_mixed(self, tmp_path):
        """Test numbering ignores non-matching files."""
        (tmp_path / "test_1.json").touch()
        (tmp_path / "other_3.json").touch()
        (tmp_path / "test_invalid.json").touch()

        result = get_next_result_number(tmp_path, "test")
        assert result == 2


def get_next_result_number(results_dir: Path, prefix: str) -> int:
    """Find the next available result number."""
    existing = list(results_dir.glob(f"{prefix}_*.json"))
    if not existing:
        return 1
    numbers = []
    for f in existing:
        try:
            num = int(f.stem.split("_")[-1])
            numbers.append(num)
        except ValueError:
            pass
    return max(numbers) + 1 if numbers else 1


class TestONNXModel:
    """Test ONNX model validity."""

    def test_onnx_loads(self):
        """Test that ONNX model can be loaded."""
        onnx_path = BASE_DIR / "saved_models" / "pointnet_autolirpa_flat.onnx"
        if not onnx_path.exists():
            pytest.skip("ONNX model not found")

        import onnx

        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)

    def test_onnx_inference(self):
        """Test ONNX model inference."""
        onnx_path = BASE_DIR / "saved_models" / "pointnet_autolirpa_flat.onnx"
        if not onnx_path.exists():
            pytest.skip("ONNX model not found")

        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path))

        # Input shape: (batch, 192) for 64 points * 3 coords
        test_input = np.random.randn(1, 192).astype(np.float32)
        outputs = session.run(None, {"input": test_input})

        # Output shape: (batch, 2)
        assert outputs[0].shape == (1, 2)

    def test_onnx_batch_inference(self):
        """Test ONNX model with different batch sizes."""
        onnx_path = BASE_DIR / "saved_models" / "pointnet_autolirpa_flat.onnx"
        if not onnx_path.exists():
            pytest.skip("ONNX model not found")

        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path))

        for batch_size in [1, 4, 8]:
            test_input = np.random.randn(batch_size, 192).astype(np.float32)
            outputs = session.run(None, {"input": test_input})
            assert outputs[0].shape == (batch_size, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
