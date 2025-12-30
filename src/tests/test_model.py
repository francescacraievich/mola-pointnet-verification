"""Tests for the MLP model."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import CLASS_GROUND, CLASS_OBSTACLE, CLASS_OTHER, MLPClassifier, create_model


class TestMLPClassifier:
    """Test suite for MLPClassifier."""

    def test_create_model_default(self):
        """Test model creation with default parameters."""
        model = create_model()
        assert model.input_dim == 3
        assert model.hidden_dims == (256, 256, 128)
        assert model.num_classes == 3

    def test_parameter_count(self):
        """Test that model has approximately 100K parameters."""
        model = create_model()
        params = model.count_parameters()
        assert 95000 < params < 105000, f"Expected ~100K params, got {params}"

    def test_forward_pass_shape(self):
        """Test forward pass output shape."""
        model = create_model()
        batch_size = 32
        x = torch.randn(batch_size, 3)
        output = model(x)
        assert output.shape == (batch_size, 3)

    def test_predict_shape(self):
        """Test predict method output shape."""
        model = create_model()
        batch_size = 16
        x = torch.randn(batch_size, 3)
        predictions = model.predict(x)
        assert predictions.shape == (batch_size,)
        assert predictions.dtype == torch.int64

    def test_predict_proba_shape(self):
        """Test predict_proba output shape and values."""
        model = create_model()
        x = torch.randn(8, 3)
        probs = model.predict_proba(x)
        assert probs.shape == (8, 3)
        # Probabilities should sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.ones(8), atol=1e-6)
        # All probabilities should be non-negative
        assert (probs >= 0).all()

    def test_no_batchnorm(self):
        """Test that model has no BatchNorm layers."""
        model = create_model()
        for module in model.modules():
            assert not isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))

    def test_no_dropout(self):
        """Test that model has no Dropout layers."""
        model = create_model()
        for module in model.modules():
            assert not isinstance(module, torch.nn.Dropout)

    def test_custom_architecture(self):
        """Test model with custom architecture."""
        model = create_model(input_dim=4, hidden_dims=(64, 64), num_classes=5)
        assert model.input_dim == 4
        assert model.hidden_dims == (64, 64)
        assert model.num_classes == 5

        x = torch.randn(10, 4)
        output = model(x)
        assert output.shape == (10, 5)

    def test_class_constants(self):
        """Test class constant definitions."""
        assert CLASS_GROUND == 0
        assert CLASS_OBSTACLE == 1
        assert CLASS_OTHER == 2


class TestModelGradients:
    """Test gradient computation for verification compatibility."""

    def test_gradients_flow(self):
        """Test that gradients flow through the model."""
        model = create_model()
        x = torch.randn(4, 3, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_deterministic_output(self):
        """Test that model output is deterministic in eval mode."""
        model = create_model()
        model.eval()
        x = torch.randn(5, 3)
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1, out2)
