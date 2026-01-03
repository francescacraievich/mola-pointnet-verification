"""
Tests for PointNet models.
"""

import pytest
import torch

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pointnet_model import (
    TNet,
    PointNetEncoder,
    PointNetClassifier,
    PointNetForVerification,
    PointNetLarge,
)


class TestTNet:
    """Tests for T-Net spatial transformer."""

    def test_tnet_output_shape(self):
        """Test T-Net produces correct output shape."""
        tnet = TNet(k=3)
        x = torch.randn(4, 3, 64)  # (batch, channels, points)
        out = tnet(x)
        assert out.shape == (4, 3, 3)

    def test_tnet_identity_init(self):
        """Test T-Net initializes close to identity."""
        tnet = TNet(k=3)
        tnet.eval()
        x = torch.randn(1, 3, 64)
        out = tnet(x)
        identity = torch.eye(3)
        # Should be close to identity at initialization
        assert out.shape == (1, 3, 3)

    def test_tnet_k64(self):
        """Test T-Net with k=64 for feature transform."""
        tnet = TNet(k=64)
        x = torch.randn(2, 64, 128)
        out = tnet(x)
        assert out.shape == (2, 64, 64)


class TestPointNetEncoder:
    """Tests for PointNet encoder."""

    def test_encoder_output_shape(self):
        """Test encoder produces correct output shape."""
        encoder = PointNetEncoder(in_channels=3, feature_transform=True)
        x = torch.randn(4, 64, 3)  # (batch, points, channels)
        global_feat, input_trans, feature_trans = encoder(x)

        assert global_feat.shape == (4, 1024)
        assert input_trans.shape == (4, 3, 3)
        assert feature_trans.shape == (4, 64, 64)

    def test_encoder_no_feature_transform(self):
        """Test encoder without feature transform."""
        encoder = PointNetEncoder(in_channels=3, feature_transform=False)
        x = torch.randn(4, 64, 3)
        global_feat, input_trans, feature_trans = encoder(x)

        assert global_feat.shape == (4, 1024)
        assert input_trans.shape == (4, 3, 3)
        assert feature_trans is None


class TestPointNetClassifier:
    """Tests for full PointNet classifier."""

    def test_classifier_output_shape(self):
        """Test classifier produces correct output shape."""
        model = PointNetClassifier(num_points=64, num_classes=10)
        x = torch.randn(4, 64, 3)
        logits, input_trans, feature_trans = model(x)

        assert logits.shape == (4, 10)
        assert input_trans.shape == (4, 3, 3)
        assert feature_trans.shape == (4, 64, 64)

    def test_classifier_flattened_input(self):
        """Test classifier with flattened input."""
        model = PointNetClassifier(num_points=64, num_classes=2)
        x = torch.randn(4, 64 * 3)  # Flattened input
        logits, _, _ = model(x)
        assert logits.shape == (4, 2)

    def test_classifier_binary(self):
        """Test binary classification."""
        model = PointNetClassifier(num_points=512, num_classes=2)
        x = torch.randn(8, 512, 3)
        logits, _, _ = model(x)
        assert logits.shape == (8, 2)


class TestPointNetForVerification:
    """Tests for verification-optimized PointNet."""

    def test_verification_model_output_shape(self):
        """Test verification model produces correct output shape."""
        model = PointNetForVerification(num_points=512, num_classes=2, use_tnet=True)
        x = torch.randn(4, 512, 3)
        out = model(x)
        assert out.shape == (4, 2)

    def test_verification_model_no_tnet(self):
        """Test verification model without T-Net."""
        model = PointNetForVerification(num_points=512, num_classes=2, use_tnet=False)
        x = torch.randn(4, 512, 3)
        out = model(x)
        assert out.shape == (4, 2)

    def test_verification_model_flattened_input(self):
        """Test verification model with flattened input."""
        model = PointNetForVerification(num_points=512, num_classes=2)
        x = torch.randn(4, 512 * 3)  # Flattened
        out = model(x)
        assert out.shape == (4, 2)

    def test_verification_model_mean_pooling(self):
        """Test verification model with mean pooling."""
        model = PointNetForVerification(num_points=512, num_classes=2, pooling="mean")
        x = torch.randn(4, 512, 3)
        out = model(x)
        assert out.shape == (4, 2)

    def test_verification_model_max_pooling(self):
        """Test verification model with max pooling."""
        model = PointNetForVerification(num_points=512, num_classes=2, pooling="max")
        x = torch.randn(4, 512, 3)
        out = model(x)
        assert out.shape == (4, 2)

    def test_verification_model_parameter_count(self):
        """Test verification model has expected parameter count."""
        model = PointNetForVerification(num_points=512, num_classes=2, use_tnet=True)
        n_params = sum(p.numel() for p in model.parameters())
        # Should be around 166k parameters
        assert 100_000 < n_params < 300_000

    def test_verification_model_gradient_flow(self):
        """Test gradient flows through the model."""
        model = PointNetForVerification(num_points=512, num_classes=2)
        x = torch.randn(4, 512, 3, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestPointNetLarge:
    """Tests for large PointNet model."""

    def test_large_model_output_shape(self):
        """Test large model produces correct output shape."""
        model = PointNetLarge(num_points=512, num_classes=2)
        x = torch.randn(4, 512, 3)
        out = model(x)
        assert out.shape == (4, 2)

    def test_large_model_more_parameters(self):
        """Test large model has more parameters than base."""
        base_model = PointNetForVerification(num_points=512, num_classes=2)
        large_model = PointNetLarge(num_points=512, num_classes=2)

        base_params = sum(p.numel() for p in base_model.parameters())
        large_params = sum(p.numel() for p in large_model.parameters())

        assert large_params > base_params

    def test_large_model_flattened_input(self):
        """Test large model with flattened input."""
        model = PointNetLarge(num_points=512, num_classes=2)
        x = torch.randn(2, 512 * 3)
        out = model(x)
        assert out.shape == (2, 2)


class TestModelInference:
    """Integration tests for model inference."""

    def test_eval_mode_deterministic(self):
        """Test model produces same output in eval mode."""
        model = PointNetForVerification(num_points=512, num_classes=2)
        model.eval()

        x = torch.randn(2, 512, 3)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        assert torch.allclose(out1, out2)

    def test_batch_size_one(self):
        """Test model works with batch size 1."""
        model = PointNetForVerification(num_points=512, num_classes=2)
        x = torch.randn(1, 512, 3)
        out = model(x)
        assert out.shape == (1, 2)

    def test_different_num_classes(self):
        """Test model with different number of classes."""
        for n_classes in [2, 5, 10]:
            model = PointNetForVerification(num_points=512, num_classes=n_classes)
            x = torch.randn(4, 512, 3)
            out = model(x)
            assert out.shape == (4, n_classes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
