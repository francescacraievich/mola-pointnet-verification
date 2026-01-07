#!/usr/bin/env python3
"""
Train PointNet with 3DCertify architecture using 1024 points and 1024 features.
This matches the original 3DCertify configuration for accurate verification. 10k for training and test.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent  # mola-pointnet-verification/
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "3dcertify"))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pointnet.model import PointNet
from torch.utils.data import DataLoader, TensorDataset

# Config - full size model (3DCertify standard)
NUM_POINTS = 1024
NUM_CLASSES = 2
MAX_FEATURES = 1024
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("Training PointNet (3DCertify Architecture) - 1024 Points, 1024 Features")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"Points per sample: {NUM_POINTS}")
print(f"Max features: {MAX_FEATURES}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print("=" * 70)
print()

# Load data - use 1024-point dataset
print("Loading MOLA data (1024 points)...")
train_groups = np.load(BASE_DIR / "data/pointnet_1024/train_groups.npy")
train_labels = np.load(BASE_DIR / "data/pointnet_1024/train_labels.npy")
test_groups = np.load(BASE_DIR / "data/pointnet_1024/test_groups.npy")
test_labels = np.load(BASE_DIR / "data/pointnet_1024/test_labels.npy")

print(f"  Train: {train_groups.shape}, Labels: {np.bincount(train_labels)}")
print(f"  Test: {test_groups.shape}, Labels: {np.bincount(test_labels)}")
print()

# Extract XYZ only (first 3 channels)
print("Extracting XYZ coordinates...")
train_xyz = train_groups[:, :, :3]
test_xyz = test_groups[:, :, :3]
print(f"  Train shape: {train_xyz.shape}")
print(f"  Test shape: {test_xyz.shape}")
print()

# Create datasets
train_dataset = TensorDataset(torch.FloatTensor(train_xyz), torch.LongTensor(train_labels))
test_dataset = TensorDataset(torch.FloatTensor(test_xyz), torch.LongTensor(test_labels))

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)

# Create model - 3DCertify architecture with 1024 features
print("Creating PointNet model (3DCertify architecture, 1024 points, 1024 features)...")
model = PointNet(
    number_points=NUM_POINTS,
    num_classes=NUM_CLASSES,
    max_features=MAX_FEATURES,
    pool_function="improved_max",  # Cascaded MaxPool for better verification
    transposed_input=False,  # Input is (batch, n_points, 3)
).to(DEVICE)

print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Max features: {MAX_FEATURES}")
print(f"  Pooling: improved_max (cascaded)")
print()

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


# Training loop
def train_epoch(epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        if (batch_idx + 1) % 50 == 0:
            print(
                f"  [{batch_idx+1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}, "
                f"Acc: {100.*correct/total:.2f}%"
            )

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    accuracy = 100.0 * correct / total
    return accuracy


# Train
print("=" * 70)
print("Training...")
print("=" * 70)
print()

best_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    print(f"Epoch {epoch}/{EPOCHS}")
    print("-" * 70)

    train_loss, train_acc = train_epoch(epoch)
    test_acc = evaluate()
    scheduler.step()

    print()
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Test Acc: {test_acc:.2f}%")

    # Save best model
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "test_accuracy": test_acc,
                "train_accuracy": train_acc,
                "num_points": NUM_POINTS,
                "num_classes": NUM_CLASSES,
                "max_features": MAX_FEATURES,
                "architecture": "3DCertify",
                "pool_function": "improved_max",
            },
            BASE_DIR / "saved_models/pointnet_3dcertify_1024.pth",
        )
        print(f"  * Best model saved! (Test Acc: {test_acc:.2f}%)")

    print()

print("=" * 70)
print("Training complete!")
print(f"Best test accuracy: {best_acc:.2f}%")
print(f"Model saved to: saved_models/pointnet_3dcertify_1024.pth")
print("=" * 70)
