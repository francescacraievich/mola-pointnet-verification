"""
Train PointNet for Point Cloud Classification.

Uses GPU acceleration for fast training on groups of LiDAR points.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from pointnet_model import PointNetForVerification, PointNetClassifier, PointNetLarge

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {DEVICE}")


def load_dataset(data_path: Path):
    """Load point group dataset."""
    train_groups = np.load(data_path / "train_groups.npy")
    train_labels = np.load(data_path / "train_labels.npy")
    test_groups = np.load(data_path / "test_groups.npy")
    test_labels = np.load(data_path / "test_labels.npy")

    print(f"Train: {train_groups.shape}, Test: {test_groups.shape}")

    # Convert to tensors
    train_data = TensorDataset(
        torch.from_numpy(train_groups).float(),
        torch.from_numpy(train_labels).long()
    )
    test_data = TensorDataset(
        torch.from_numpy(test_groups).float(),
        torch.from_numpy(test_labels).long()
    )

    return train_data, test_data


def train_epoch(model, loader, criterion, optimizer):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_groups, batch_labels in loader:
        batch_groups = batch_groups.to(DEVICE)
        batch_labels = batch_labels.to(DEVICE)

        optimizer.zero_grad()

        # Forward pass
        if isinstance(model, PointNetClassifier):
            outputs, _, feat_trans = model(batch_groups)
            # Add regularization for feature transform
            loss = criterion(outputs, batch_labels)
            if feat_trans is not None:
                identity = torch.eye(64, device=DEVICE).unsqueeze(0).repeat(batch_groups.size(0), 1, 1)
                reg_loss = torch.mean(torch.norm(identity - torch.bmm(feat_trans, feat_trans.transpose(1, 2)), dim=(1, 2)))
                loss = loss + 0.001 * reg_loss
        else:
            outputs = model(batch_groups)
            loss = criterion(outputs, batch_labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_groups.size(0)
        _, predicted = outputs.max(1)
        total += batch_labels.size(0)
        correct += predicted.eq(batch_labels).sum().item()

    return total_loss / total, 100. * correct / total


def evaluate(model, loader, criterion):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_groups, batch_labels in loader:
            batch_groups = batch_groups.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            if isinstance(model, PointNetClassifier):
                outputs, _, _ = model(batch_groups)
            else:
                outputs = model(batch_groups)

            loss = criterion(outputs, batch_labels)

            total_loss += loss.item() * batch_groups.size(0)
            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()

    return total_loss / total, 100. * correct / total


def train_pointnet(
    data_path: Path,
    output_path: Path,
    n_points: int = 64,
    num_classes: int = 2,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.001,
    use_tnet: bool = True,
    model_type: str = "verification",
):
    """
    Train PointNet model.

    Args:
        data_path: Path to dataset
        output_path: Path to save model
        n_points: Points per sample
        num_classes: Number of classes
        epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        use_tnet: Use T-Net in verification model
        model_type: "verification" (smaller) or "full" (original PointNet)
    """
    print("=" * 60)
    print("Training PointNet")
    print("=" * 60)

    # Load data
    print("\n1. Loading dataset...")
    train_data, test_data = load_dataset(data_path)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Create model
    print(f"\n2. Creating model (type={model_type}, use_tnet={use_tnet})...")
    if model_type == "full":
        model = PointNetClassifier(
            num_points=n_points,
            num_classes=num_classes,
            feature_transform=True,
        )
    elif model_type == "large":
        model = PointNetLarge(
            num_points=n_points,
            num_classes=num_classes,
            use_tnet=use_tnet,
        )
    else:
        model = PointNetForVerification(
            num_points=n_points,
            num_classes=num_classes,
            use_tnet=use_tnet,
        )

    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {n_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Training loop
    print(f"\n3. Training for {epochs} epochs...")
    best_acc = 0
    best_model_state = None

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, Acc={train_acc:.1f}% | "
                  f"Test Loss={test_loss:.4f}, Acc={test_acc:.1f}%")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation
    print(f"\n4. Final evaluation...")
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"   Best Test Accuracy: {best_acc:.2f}%")

    # Save model
    print(f"\n5. Saving model...")
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "pointnet.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_points": n_points,
        "num_classes": num_classes,
        "use_tnet": use_tnet,
        "model_type": model_type,
        "test_accuracy": best_acc,
    }, model_path)
    print(f"   Saved to {model_path}")

    return model, best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/pointnet")
    parser.add_argument("--output", type=str, default="models")
    parser.add_argument("--n-points", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--use-tnet", action="store_true", default=True)
    parser.add_argument("--model-type", type=str, default="verification",
                       choices=["verification", "full", "large"])

    args = parser.parse_args()

    train_pointnet(
        data_path=Path(args.data),
        output_path=Path(args.output),
        n_points=args.n_points,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_tnet=args.use_tnet,
        model_type=args.model_type,
    )
