#!/usr/bin/env python3
"""Tests for PointNet model."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src" / "model"))

from pointnet_autolirpa_compatible import PointNetAutoLiRPA


class TestPointNetAutoLiRPA:
    """Test suite for PointNetAutoLiRPA model."""

    @pytest.fixture
    def model(self):
        """Create a model instance for testing."""
        return PointNetAutoLiRPA(
            num_points=64, num_classes=2, max_features=512, use_batchnorm=False
        )

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        # Shape: (batch, n_points, 3)
        return torch.randn(4, 64, 3)

    def test_model_creation(self, model):
        """Test that model is created correctly."""
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_forward_pass(self, model, sample_input):
        """Test forward pass produces correct output shape."""
        model.eval()
        with torch.no_grad():
            output = model(sample_input)

        # Output should be (batch, num_classes)
        assert output.shape == (4, 2)

    def test_output_range(self, model, sample_input):
        """Test that output values are reasonable (not NaN/Inf)."""
        model.eval()
        with torch.no_grad():
            output = model(sample_input)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_batch_size_independence(self, model):
        """Test that model works with different batch sizes."""
        model.eval()

        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 64, 3)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (batch_size, 2)

    def test_deterministic_eval(self, model, sample_input):
        """Test that eval mode gives deterministic results."""
        model.eval()

        with torch.no_grad():
            out1 = model(sample_input)
            out2 = model(sample_input)

        assert torch.allclose(out1, out2)

    def test_gradient_flow(self, model, sample_input):
        """Test that gradients flow through the model."""
        model.train()
        sample_input.requires_grad = True

        output = model(sample_input)
        loss = output.sum()
        loss.backward()

        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()


class TestModelLoading:
    """Test model loading from checkpoint."""

    @pytest.fixture
    def model_path(self):
        return BASE_DIR / "saved_models" / "pointnet_autolirpa_512.pth"

    def test_checkpoint_exists(self, model_path):
        """Test that the model checkpoint exists."""
        assert model_path.exists(), f"Model not found at {model_path}"

    def test_load_checkpoint(self, model_path):
        """Test loading model from checkpoint."""
        if not model_path.exists():
            pytest.skip("Model checkpoint not found")

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)

        assert "model_state_dict" in checkpoint
        assert "test_accuracy" in checkpoint or "epoch" in checkpoint

    def test_load_and_inference(self, model_path):
        """Test loading model and running inference."""
        if not model_path.exists():
            pytest.skip("Model checkpoint not found")

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)

        model = PointNetAutoLiRPA(64, 2, 512, use_batchnorm=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        x = torch.randn(1, 64, 3)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
