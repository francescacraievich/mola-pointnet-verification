"""
PointNet Architecture for Point Cloud Classification.

Full PointNet with T-Net (Spatial Transformer Network) for point cloud alignment.
Based on: Qi et al., "PointNet: Deep Learning on Point Sets" (CVPR 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """T-Net for spatial transformer."""

    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # Initialize to identity
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.copy_(torch.eye(k).view(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, dim=2)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        x = x.view(batch_size, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    """PointNet encoder with T-Net."""

    def __init__(self, in_channels: int = 3, feature_transform: bool = True):
        super().__init__()
        self.feature_transform = feature_transform

        self.input_tnet = TNet(k=3)

        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        if feature_transform:
            self.feature_tnet = TNet(k=64)

        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)

    def forward(self, x: torch.Tensor):
        batch_size, num_points, _ = x.shape

        x = x.transpose(1, 2)
        input_trans = self.input_tnet(x)

        x = x.transpose(1, 2)
        x = torch.bmm(x, input_trans)
        x = x.transpose(1, 2)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        feature_trans = None
        if self.feature_transform:
            feature_trans = self.feature_tnet(x)
            x = x.transpose(1, 2)
            x = torch.bmm(x, feature_trans)
            x = x.transpose(1, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        global_feat = torch.max(x, dim=2)[0]

        return global_feat, input_trans, feature_trans


class PointNetClassifier(nn.Module):
    """Full PointNet classifier."""

    def __init__(
        self,
        num_points: int = 1024,
        num_classes: int = 2,
        feature_transform: bool = True,
    ):
        super().__init__()

        self.num_points = num_points
        self.num_classes = num_classes
        self.feature_transform = feature_transform

        self.encoder = PointNetEncoder(in_channels=3, feature_transform=feature_transform)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x: torch.Tensor):
        if x.dim() == 2:
            batch_size = x.shape[0]
            x = x.view(batch_size, self.num_points, 3)

        global_feat, input_trans, feature_trans = self.encoder(x)

        x = F.relu(self.bn1(self.fc1(global_feat)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        logits = self.fc3(x)

        return logits, input_trans, feature_trans


class PointNetForVerification(nn.Module):
    """
    PointNet for verification - IDENTICAL to original PointNet architecture.

    Based on: Qi et al., "PointNet: Deep Learning on Point Sets" (CVPR 2017)

    This uses the EXACT original PointNet architecture with:
    - Input T-Net (3x3) with architecture: 3→64→128→1024→512→256→9 + BatchNorm
    - Feature T-Net (64x64) with architecture: 64→64→128→1024→512→256→4096 + BatchNorm
    - Point-wise MLP: in_channels→64→64→[feat_transform]→64→128→1024 with BatchNorm
    - Global max pooling
    - Classifier MLP: 1024→512→256→classes with BatchNorm + Dropout(0.3)

    Supports feature-augmented input (xyz + geometric features):
    - in_channels=3: xyz only (original PointNet)
    - in_channels=7: xyz(3) + features(4) for NSGA3-derived features
      Features: linearity, curvature, density_var, planarity

    The T-Net for input transform always operates on xyz (3 channels).
    Additional features are concatenated AFTER the input transform.
    """

    def __init__(
        self,
        num_points: int = 1024,  # Original PointNet uses 1024 points
        num_classes: int = 2,
        use_tnet: bool = True,
        feature_transform: bool = True,
        in_channels: int = 3,  # 3 for xyz, 7 for xyz+features
    ):
        super().__init__()

        self.num_points = num_points
        self.num_classes = num_classes
        self.use_tnet = use_tnet
        self.feature_transform = feature_transform
        self.in_channels = in_channels
        self.input_dim = num_points * in_channels

        # Input T-Net (3x3 transformation) - original architecture
        if use_tnet:
            self.input_tnet = TNet(k=3)

        # Feature T-Net (64x64 transformation)
        if feature_transform:
            self.feat_tnet = TNet(k=64)

        # Point-wise MLP with BatchNorm (original architecture)
        # Original: 64 → 64 → 64 → 128 → 1024 (5 layers)
        # First layer takes all channels (xyz + features if present)
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)  # Feature transform applied after this
        # After feature transform:
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        # Classifier MLP with BatchNorm and Dropout (original architecture)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Handle flattened input for ONNX compatibility
        if x.dim() == 2:
            x = x.view(batch_size, self.num_points, self.in_channels)

        # Separate xyz and features if using augmented input
        if self.in_channels > 3:
            xyz = x[:, :, :3]  # (batch, n_points, 3)
            extra_features = x[:, :, 3:]  # (batch, n_points, in_channels-3)
        else:
            xyz = x
            extra_features = None

        # Input T-Net: predict 3x3 transformation matrix (only for xyz)
        input_trans = None
        if self.use_tnet:
            # (batch, n_points, 3) -> (batch, 3, n_points)
            xyz_t = xyz.transpose(1, 2)
            input_trans = self.input_tnet(xyz_t)  # (batch, 3, 3)
            # Apply transformation to xyz
            xyz = torch.bmm(xyz, input_trans)  # (batch, n_points, 3)

        # Recombine xyz with extra features if present
        if extra_features is not None:
            x = torch.cat([xyz, extra_features], dim=2)  # (batch, n_points, in_channels)
        else:
            x = xyz

        # Point-wise MLP (shared weights across points)
        # (batch, n_points, in_channels) -> (batch, in_channels, n_points)
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))  # (batch, 64, n_points)
        x = F.relu(self.bn2(self.conv2(x)))  # (batch, 64, n_points)

        # Feature T-Net (64x64 transformation) - applied after conv2
        feat_trans = None
        if self.feature_transform:
            feat_trans = self.feat_tnet(x)  # (batch, 64, 64)
            x = x.transpose(1, 2)  # (batch, n_points, 64)
            x = torch.bmm(x, feat_trans)  # (batch, n_points, 64)
            x = x.transpose(1, 2)  # (batch, 64, n_points)

        x = F.relu(self.bn3(self.conv3(x)))  # (batch, 64, n_points)
        x = F.relu(self.bn4(self.conv4(x)))  # (batch, 128, n_points)
        x = F.relu(self.bn5(self.conv5(x)))  # (batch, 1024, n_points)

        # Global max pooling (symmetric function for permutation invariance)
        x = torch.max(x, dim=2)[0]  # (batch, 1024)

        # Classifier MLP
        x = F.relu(self.bn_fc1(self.fc1(x)))  # (batch, 512)
        x = self.dropout(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))  # (batch, 256)
        x = self.dropout(x)
        x = self.fc3(x)  # (batch, num_classes)

        return x

    def get_transforms(self, x: torch.Tensor):
        """Return input and feature transforms for regularization loss."""
        batch_size = x.shape[0]

        if x.dim() == 2:
            x = x.view(batch_size, self.num_points, self.in_channels)

        if self.in_channels > 3:
            xyz = x[:, :, :3]
        else:
            xyz = x

        input_trans = None
        feat_trans = None

        if self.use_tnet:
            xyz_t = xyz.transpose(1, 2)
            input_trans = self.input_tnet(xyz_t)
            xyz = torch.bmm(xyz, input_trans)

        if self.in_channels > 3:
            x = torch.cat([xyz, x[:, :, 3:]], dim=2)
        else:
            x = xyz

        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        if self.feature_transform:
            feat_trans = self.feat_tnet(x)

        return input_trans, feat_trans


class PointNetLarge(nn.Module):
    """
    PointNet Large - Versione con più parametri per verifica più robusta.

    Architettura aumentata:
    - T-Net: 3 -> 128 -> 256 -> 512 (vs 3 -> 64 -> 128 -> 256)
    - Point MLP: 3 -> 128 -> 256 -> 512 -> 1024 (vs 3 -> 64 -> 128 -> 256)
    - Classifier: 1024 -> 512 -> 256 -> 128 -> 2 (vs 256 -> 128 -> 64 -> 2)

    ~1.5M parametri (vs ~100k del modello base)
    """

    def __init__(
        self,
        num_points: int = 64,
        num_classes: int = 2,
        use_tnet: bool = True,
        pooling: str = "max",
    ):
        super().__init__()

        self.num_points = num_points
        self.num_classes = num_classes
        self.use_tnet = use_tnet
        self.pooling = pooling
        self.input_dim = num_points * 3

        if use_tnet:
            # T-Net più grande
            self.tnet_conv1 = nn.Conv1d(3, 128, 1)
            self.tnet_conv2 = nn.Conv1d(128, 256, 1)
            self.tnet_conv3 = nn.Conv1d(256, 512, 1)
            self.tnet_fc1 = nn.Linear(512, 256)
            self.tnet_fc2 = nn.Linear(256, 128)
            self.tnet_fc3 = nn.Linear(128, 9)
            self.tnet_fc3.weight.data.zero_()
            self.tnet_fc3.bias.data.copy_(torch.eye(3).view(-1))

        # Point-wise MLP più profondo
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(256, 512, 1)
        self.conv4 = nn.Conv1d(512, 1024, 1)

        # Classifier più profondo
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)

        # Dropout per regolarizzazione
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        if x.dim() == 2:
            x = x.view(batch_size, self.num_points, 3)

        # T-Net
        if self.use_tnet:
            x_t = x.transpose(1, 2)
            t = F.relu(self.tnet_conv1(x_t))
            t = F.relu(self.tnet_conv2(t))
            t = F.relu(self.tnet_conv3(t))
            if self.pooling == "mean":
                t = torch.mean(t, dim=2)
            else:
                t = torch.max(t, dim=2)[0]
            t = F.relu(self.tnet_fc1(t))
            t = F.relu(self.tnet_fc2(t))
            t = self.tnet_fc3(t)
            t = t.view(batch_size, 3, 3)
            x = torch.bmm(x, t)

        # Point-wise MLP
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Global pooling
        if self.pooling == "mean":
            x = torch.mean(x, dim=2)
        else:
            x = torch.max(x, dim=2)[0]

        # Classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


if __name__ == "__main__":
    print("PointNet Model Tests")
    print("=" * 60)

    # Test verification model
    model = PointNetForVerification(num_points=64, num_classes=2, use_tnet=True)
    x = torch.randn(4, 64, 3)
    out = model(x)
    print(f"PointNetForVerification:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test ONNX export
    print("\nTesting ONNX export...")
    model.eval()
    x_flat = torch.randn(1, 64 * 3)
    torch.onnx.export(
        model,
        x_flat,
        "/tmp/pointnet_test.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
    )
    print("ONNX export successful!")
