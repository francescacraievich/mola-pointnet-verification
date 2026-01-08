#!/usr/bin/env python3
"""
PointNet architecture compatible with auto_LiRPA by avoiding problematic operations.

Key changes for auto_LiRPA compatibility:
1. NO torch.squeeze() - causes batch dimension issues
2. NO dynamic reshape with -1 - use explicit dimensions
3. Use nn.Flatten with start_dim=1 instead of view/reshape
4. Use torch.mean() instead of MaxPool (more stable for bounds)
5. Remove Dropout 
6. Keep all operations differentiable and traceable
"""

import torch
import torch.nn as nn


class PointNetAutoLiRPA(nn.Module):

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
        # Using torch.mean in forward for compatibility

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
