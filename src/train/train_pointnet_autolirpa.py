#!/usr/bin/env python3
"""
Train PointNet model compatible with auto_LiRPA verification.

This trains a smaller model (512 features, no BatchNorm) that can be
verified efficiently with auto_LiRPA/α,β-CROWN.

Uses NSGA3 dynamic labels for training.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = Path(__file__).parent.parent.parent  # mola-pointnet-verification/
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src" / "model"))

from pointnet_autolirpa_compatible import PointNetAutoLiRPA
from src.scripts.nsga3_integration import get_criticality_weights


def load_data_with_nsga3_labels(data_dir: Path, nsga3_dir: Path, run_id: int = 10):
    """Load data and compute NSGA3 dynamic labels."""

    # Load data
    train_data = np.load(data_dir / "train_groups.npy")
    test_data = np.load(data_dir / "test_groups.npy")

    # Get NSGA3 weights
    weights = get_criticality_weights(nsga3_results_dir=nsga3_dir, run_id=run_id)
    print(
        f"NSGA-III weights: linearity={weights['linearity']:.4f}, "
        f"curvature={weights['curvature']:.4f}, "
        f"density_var={weights['density_var']:.4f}, "
        f"nonplanarity={weights['nonplanarity']:.4f}"
    )

    # Compute labels for training data
    train_labels = []
    for i in range(len(train_data)):
        features = train_data[i]
        score = (
            features[:, 3].mean() * weights["linearity"]
            + features[:, 4].mean() * weights["curvature"]
            + features[:, 5].mean() * weights["density_var"]
            + (1.0 - features[:, 6].mean()) * weights["nonplanarity"]
        )
        train_labels.append(0 if score >= 0.5 else 1)
    train_labels = np.array(train_labels)

    # Compute labels for test data
    test_labels = []
    for i in range(len(test_data)):
        features = test_data[i]
        score = (
            features[:, 3].mean() * weights["linearity"]
            + features[:, 4].mean() * weights["curvature"]
            + features[:, 5].mean() * weights["density_var"]
            + (1.0 - features[:, 6].mean()) * weights["nonplanarity"]
        )
        test_labels.append(0 if score >= 0.5 else 1)
    test_labels = np.array(test_labels)

    # Extract XYZ only
    train_xyz = train_data[:, :, :3].astype(np.float32)
    test_xyz = test_data[:, :, :3].astype(np.float32)

    print(
        f"Train: {len(train_xyz)} samples, "
        f"CRITICAL: {(train_labels==0).sum()}, NON_CRITICAL: {(train_labels==1).sum()}"
    )
    print(
        f"Test: {len(test_xyz)} samples, "
        f"CRITICAL: {(test_labels==0).sum()}, NON_CRITICAL: {(test_labels==1).sum()}"
    )

    return train_xyz, train_labels, test_xyz, test_labels


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 100,
    lr: float = 0.001,
    device: str = "cuda",
):
    """Train the model."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()

        scheduler.step()

        # Evaluation
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_x)
                _, predicted = outputs.max(1)
                test_total += batch_y.size(0)
                test_correct += predicted.eq(batch_y).sum().item()

        train_acc = 100.0 * train_correct / train_total
        test_acc = 100.0 * test_correct / test_total

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:3d}/{epochs}: "
                f"Loss={train_loss/len(train_loader):.4f}, "
                f"Train Acc={train_acc:.1f}%, "
                f"Test Acc={test_acc:.1f}% "
                f"(Best: {best_acc:.1f}%)"
            )

    return best_state, best_acc


def main():
    parser = argparse.ArgumentParser(description="Train auto_LiRPA compatible PointNet")
    parser.add_argument("--data-dir", type=str, default="data/pointnet")
    parser.add_argument(
        "--nsga3-dir", type=str, default="../mola-adversarial-nsga3/src/results/runs"
    )
    parser.add_argument("--nsga3-run-id", type=int, default=10)
    parser.add_argument("--max-features", type=int, default=512, help="Max features (256 or 512)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="saved_models/pointnet_autolirpa_512.pth")

    args = parser.parse_args()

    data_dir = BASE_DIR / args.data_dir
    nsga3_dir = Path(args.nsga3_dir)
    output_path = BASE_DIR / args.output

    print("=" * 70)
    print("Training auto_LiRPA Compatible PointNet")
    print("=" * 70)
    print(f"Max features: {args.max_features}")
    print(f"BatchNorm: False (required for auto_LiRPA)")
    print(f"Pooling: torch.mean() (required for auto_LiRPA)")
    print()

    # Load data
    print("Loading data with NSGA3 labels...")
    train_xyz, train_labels, test_xyz, test_labels = load_data_with_nsga3_labels(
        data_dir, nsga3_dir, args.nsga3_run_id
    )

    # Create datasets
    train_dataset = TensorDataset(
        torch.from_numpy(train_xyz), torch.from_numpy(train_labels).long()
    )
    test_dataset = TensorDataset(torch.from_numpy(test_xyz), torch.from_numpy(test_labels).long())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    print(f"\nCreating model with {args.max_features} features...")
    model = PointNetAutoLiRPA(
        num_points=64,
        num_classes=2,
        max_features=args.max_features,
        use_batchnorm=False,  # Required for auto_LiRPA
    )
    print(model)

    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {params:,}")
    print()

    # Train
    print("Training...")
    start_time = time.time()
    best_state, best_acc = train_model(
        model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=args.device
    )
    elapsed = time.time() - start_time

    print(f"\nTraining completed in {elapsed:.1f}s")
    print(f"Best test accuracy: {best_acc:.1f}%")

    # Save model
    model.load_state_dict(best_state)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": best_state,
            "architecture": "PointNetAutoLiRPA",
            "max_features": args.max_features,
            "use_batchnorm": False,
            "pooling": "mean",
            "test_accuracy": best_acc,
            "nsga3_run_id": args.nsga3_run_id,
        },
        output_path,
    )

    print(f"\n✓ Model saved to: {output_path}")


if __name__ == "__main__":
    main()
