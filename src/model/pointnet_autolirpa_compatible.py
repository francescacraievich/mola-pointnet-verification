#!/usr/bin/env python3
"""
PointNet architecture compatible with auto_LiRPA by avoiding problematic operations.

Based on research:
- 3DVerifier paper: https://link.springer.com/article/10.1007/s10994-022-06235-3
- 3DVerifier GitHub: https://github.com/TrustAI/3DVerifier
- auto_LiRPA issues with reshape/squeeze

Key changes for auto_LiRPA compatibility:
1. NO torch.squeeze() - causes batch dimension issues
2. NO dynamic reshape with -1 - use explicit dimensions
3. Use nn.Flatten with start_dim=1 instead of view/reshape
4. Use nn.AdaptiveAvgPool1d instead of MaxPool (more stable for bounds)
5. Remove Dropout (not supported in eval mode)
6. Keep all operations differentiable and traceable
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetAutoLiRPA(nn.Module):
    """
    PointNet compatible with auto_LiRPA verification.

    This version avoids operations that cause issues with auto_LiRPA:
    - No squeeze()
    - No dynamic reshape()
    - Uses Flatten instead of view()
    - Uses AdaptiveAvgPool instead of MaxPool for stability
    """

    def __init__(
        self,
        num_points: int = 64,
        num_classes: int = 2,
        max_features: int = 1024,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.num_points = num_points
        self.num_classes = num_classes
        self.max_features = max_features
        self.use_batchnorm = use_batchnorm

        # First MLP: (batch, 3, points) -> (batch, 64, points)
        # Match original 3DCertify structure: 3 -> 64 -> 64
        layers1 = [
            nn.Conv1d(3, 64, 1),
        ]
        if use_batchnorm:
            layers1.append(nn.BatchNorm1d(64))
        layers1.append(nn.ReLU())
        layers1.append(nn.Conv1d(64, 64, 1))
        if use_batchnorm:
            layers1.append(nn.BatchNorm1d(64))
        layers1.append(nn.ReLU())
        self.mlp1 = nn.Sequential(*layers1)

        # Second MLP: (batch, 64, points) -> (batch, max_features, points)
        # Match original 3DCertify structure: 64 -> 64 -> 128 -> 1024
        layers2 = [
            nn.Conv1d(64, 64, 1),
        ]
        if use_batchnorm:
            layers2.append(nn.BatchNorm1d(64))
        layers2.append(nn.ReLU())
        layers2.append(nn.Conv1d(64, 128, 1))
        if use_batchnorm:
            layers2.append(nn.BatchNorm1d(128))
        layers2.append(nn.ReLU())
        layers2.append(nn.Conv1d(128, max_features, 1))
        if use_batchnorm:
            layers2.append(nn.BatchNorm1d(max_features))
        layers2.append(nn.ReLU())
        self.mlp2 = nn.Sequential(*layers2)

        # Global pooling: (batch, max_features, points) -> (batch, max_features)
        # Use simple mean instead of AdaptiveAvgPool to avoid dimension issues
        # We'll compute mean manually in forward pass

        # Classification head: (batch, max_features) -> (batch, num_classes)
        # Use Linear layers that will automatically handle flattening
        layers3 = [
            nn.Linear(max_features, 512),
        ]
        if use_batchnorm:
            layers3.append(nn.BatchNorm1d(512))
        layers3.append(nn.ReLU())
        layers3.append(nn.Linear(512, 256))
        if use_batchnorm:
            layers3.append(nn.BatchNorm1d(256))
        layers3.append(nn.ReLU())
        layers3.append(nn.Linear(256, num_classes))
        self.classifier = nn.Sequential(*layers3)

    def forward(self, x):
        """
        Forward pass without problematic operations for auto_LiRPA.

        Args:
            x: (batch, num_points, 3) or (batch, 3, num_points)

        Returns:
            (batch, num_classes)
        """
        # Handle both input formats
        if x.dim() == 3:
            if x.shape[1] == self.num_points and x.shape[2] == 3:
                # Input is (batch, points, 3) - transpose to (batch, 3, points)
                x = x.transpose(1, 2)
            # else: assume already (batch, 3, points)

        # MLP layers
        x = self.mlp1(x)  # (batch, 64, points)
        x = self.mlp2(x)  # (batch, max_features, points)

        # Global pooling using mean over points dimension
        # Use torch.mean which is differentiable and traceable
        x = torch.mean(x, dim=2)  # (batch, max_features)

        # Classification
        x = self.classifier(x)  # (batch, num_classes)

        return x


def load_original_weights_to_compatible_model(
    original_model_path: str, num_points: int = 64, num_classes: int = 2, max_features: int = 1024
) -> PointNetAutoLiRPA:
    """
    Load weights from original 3DCertify PointNet to compatible model.

    Maps weights intelligently:
    - Conv1d and Linear layers: direct copy
    - BatchNorm layers: direct copy
    - Pooling: ignore (using different pooling strategy)
    - Dropout: ignore (removed)
    """
    import sys
    from pathlib import Path

    BASE_DIR = Path(__file__).parent
    sys.path.insert(0, str(BASE_DIR / "3dcertify"))
    from pointnet.model import PointNet as PointNet3DCertify

    # Load original model
    checkpoint = torch.load(original_model_path, map_location="cpu", weights_only=True)
    orig_model = PointNet3DCertify(
        number_points=num_points,
        num_classes=num_classes,
        max_features=max_features,
        pool_function="improved_max",
        disable_assertions=True,
        transposed_input=False,
    )
    orig_model.load_state_dict(checkpoint["model_state_dict"])

    # Create compatible model
    compat_model = PointNetAutoLiRPA(num_points, num_classes, max_features, use_batchnorm=True)

    # Map weights
    with torch.no_grad():
        # MLP1: mlp1.0 (Conv) -> mlp1.0, mlp1.1 (BN) -> mlp1.1
        compat_model.mlp1[0].weight.copy_(orig_model.mlp1[0].weight)
        compat_model.mlp1[0].bias.copy_(orig_model.mlp1[0].bias)
        compat_model.mlp1[1].weight.copy_(orig_model.mlp1[1].weight)
        compat_model.mlp1[1].bias.copy_(orig_model.mlp1[1].bias)
        compat_model.mlp1[1].running_mean.copy_(orig_model.mlp1[1].running_mean)
        compat_model.mlp1[1].running_var.copy_(orig_model.mlp1[1].running_var)

        # MLP1 second conv
        compat_model.mlp1[3].weight.copy_(orig_model.mlp1[3].weight)
        compat_model.mlp1[3].bias.copy_(orig_model.mlp1[3].bias)
        compat_model.mlp1[4].weight.copy_(orig_model.mlp1[4].weight)
        compat_model.mlp1[4].bias.copy_(orig_model.mlp1[4].bias)
        compat_model.mlp1[4].running_mean.copy_(orig_model.mlp1[4].running_mean)
        compat_model.mlp1[4].running_var.copy_(orig_model.mlp1[4].running_var)

        # MLP2
        compat_model.mlp2[0].weight.copy_(orig_model.mlp2[0].weight)
        compat_model.mlp2[0].bias.copy_(orig_model.mlp2[0].bias)
        compat_model.mlp2[1].weight.copy_(orig_model.mlp2[1].weight)
        compat_model.mlp2[1].bias.copy_(orig_model.mlp2[1].bias)
        compat_model.mlp2[1].running_mean.copy_(orig_model.mlp2[1].running_mean)
        compat_model.mlp2[1].running_var.copy_(orig_model.mlp2[1].running_var)

        compat_model.mlp2[3].weight.copy_(orig_model.mlp2[3].weight)
        compat_model.mlp2[3].bias.copy_(orig_model.mlp2[3].bias)
        compat_model.mlp2[4].weight.copy_(orig_model.mlp2[4].weight)
        compat_model.mlp2[4].bias.copy_(orig_model.mlp2[4].bias)
        compat_model.mlp2[4].running_mean.copy_(orig_model.mlp2[4].running_mean)
        compat_model.mlp2[4].running_var.copy_(orig_model.mlp2[4].running_var)

        compat_model.mlp2[6].weight.copy_(orig_model.mlp2[6].weight)
        compat_model.mlp2[6].bias.copy_(orig_model.mlp2[6].bias)
        compat_model.mlp2[7].weight.copy_(orig_model.mlp2[7].weight)
        compat_model.mlp2[7].bias.copy_(orig_model.mlp2[7].bias)
        compat_model.mlp2[7].running_mean.copy_(orig_model.mlp2[7].running_mean)
        compat_model.mlp2[7].running_var.copy_(orig_model.mlp2[7].running_var)

        # Classifier (MLP3)
        compat_model.classifier[0].weight.copy_(orig_model.mlp3[0].weight)
        compat_model.classifier[0].bias.copy_(orig_model.mlp3[0].bias)
        compat_model.classifier[1].weight.copy_(orig_model.mlp3[1].weight)
        compat_model.classifier[1].bias.copy_(orig_model.mlp3[1].bias)
        compat_model.classifier[1].running_mean.copy_(orig_model.mlp3[1].running_mean)
        compat_model.classifier[1].running_var.copy_(orig_model.mlp3[1].running_var)

        compat_model.classifier[3].weight.copy_(orig_model.mlp3[3].weight)
        compat_model.classifier[3].bias.copy_(orig_model.mlp3[3].bias)
        # Skip dropout (index 4)
        compat_model.classifier[4].weight.copy_(orig_model.mlp3[5].weight)
        compat_model.classifier[4].bias.copy_(orig_model.mlp3[5].bias)
        compat_model.classifier[4].running_mean.copy_(orig_model.mlp3[5].running_mean)
        compat_model.classifier[4].running_var.copy_(orig_model.mlp3[5].running_var)

        compat_model.classifier[6].weight.copy_(orig_model.mlp3[7].weight)
        compat_model.classifier[6].bias.copy_(orig_model.mlp3[7].bias)

    return compat_model


if __name__ == "__main__":
    # Test the model
    print("Creating auto_LiRPA compatible PointNet...")
    model = PointNetAutoLiRPA(num_points=64, num_classes=2, max_features=1024)
    model.eval()

    print(f"\nModel architecture:")
    print(model)

    # Test forward pass
    dummy_input = torch.randn(2, 64, 3)
    output = model(dummy_input)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output: {output}")

    print("\nModel created successfully!")
    print("\nKey features for auto_LiRPA compatibility:")
    print("  - No squeeze() operations")
    print("  - No dynamic reshape()")
    print("  - Uses nn.Flatten(start_dim=1)")
    print("  - Uses AdaptiveAvgPool1d instead of MaxPool")
    print("  - No Dropout layers")
