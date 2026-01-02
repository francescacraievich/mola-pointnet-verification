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
    PointNet for verification using original architecture.

    Based on: Qi et al., "PointNet: Deep Learning on Point Sets" (CVPR 2017)

    This uses the original PointNet architecture with:
    - T-Net for input transformation (3x3 matrix)
    - Point-wise MLP with Conv1d (kernel_size=1)
    - Global max pooling (or mean pooling for CROWN compatibility)
    - Fully connected classifier

    Simplified version without feature transform (64x64 T-Net) for verification.
    """

    def __init__(
        self,
        num_points: int = 64,
        num_classes: int = 2,
        use_tnet: bool = True,
        pooling: str = "max",  # "max" or "mean"
    ):
        super().__init__()

        self.num_points = num_points
        self.num_classes = num_classes
        self.use_tnet = use_tnet
        self.pooling = pooling
        self.input_dim = num_points * 3

        if use_tnet:
            # Input T-Net (3x3 transformation matrix)
            # Architecture from original paper
            self.tnet_conv1 = nn.Conv1d(3, 64, 1)
            self.tnet_conv2 = nn.Conv1d(64, 128, 1)
            self.tnet_conv3 = nn.Conv1d(128, 256, 1)
            self.tnet_fc1 = nn.Linear(256, 128)
            self.tnet_fc2 = nn.Linear(128, 64)
            self.tnet_fc3 = nn.Linear(64, 9)
            # Initialize to identity
            self.tnet_fc3.weight.data.zero_()
            self.tnet_fc3.bias.data.copy_(torch.eye(3).view(-1))

        # Point-wise MLP (shared weights across points)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)

        # Classifier MLP
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Handle flattened input for ONNX compatibility
        if x.dim() == 2:
            x = x.view(batch_size, self.num_points, 3)

        # Input T-Net: predict 3x3 transformation matrix
        if self.use_tnet:
            # (batch, n_points, 3) -> (batch, 3, n_points)
            x_t = x.transpose(1, 2)
            t = F.relu(self.tnet_conv1(x_t))  # (batch, 64, n_points)
            t = F.relu(self.tnet_conv2(t))    # (batch, 128, n_points)
            t = F.relu(self.tnet_conv3(t))    # (batch, 256, n_points)
            # Global pooling (mean for CROWN, max for original)
            if self.pooling == "mean":
                t = torch.mean(t, dim=2)      # (batch, 256) - CROWN compatible
            else:
                t = torch.max(t, dim=2)[0]    # (batch, 256)
            t = F.relu(self.tnet_fc1(t))      # (batch, 128)
            t = F.relu(self.tnet_fc2(t))      # (batch, 64)
            t = self.tnet_fc3(t)              # (batch, 9)
            t = t.view(batch_size, 3, 3)      # (batch, 3, 3)
            # Apply transformation to input points
            x = torch.bmm(x, t)               # (batch, n_points, 3)

        # Point-wise MLP
        # (batch, n_points, 3) -> (batch, 3, n_points)
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))             # (batch, 64, n_points)
        x = F.relu(self.conv2(x))             # (batch, 128, n_points)
        x = F.relu(self.conv3(x))             # (batch, 256, n_points)

        # Global pooling (symmetric function for permutation invariance)
        if self.pooling == "mean":
            x = torch.mean(x, dim=2)          # (batch, 256) - CROWN compatible
        else:
            x = torch.max(x, dim=2)[0]        # (batch, 256) - original PointNet

        # Classifier
        x = F.relu(self.fc1(x))               # (batch, 128)
        x = F.relu(self.fc2(x))               # (batch, 64)
        x = self.fc3(x)                       # (batch, num_classes)

        return x


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
