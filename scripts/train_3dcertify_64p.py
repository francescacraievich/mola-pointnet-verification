#!/usr/bin/env python3
"""
Train PointNet with 3DCertify architecture using 64 points (like their examples).
This matches the model used in 3DCertify paper experiments.
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, '3dcertify')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pointnet.model import PointNet

# Config - matching 3DCertify examples
NUM_POINTS = 64  # Like in verify_perturbation.py examples
NUM_CLASSES = 2
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("="*70)
print("Training PointNet (3DCertify Architecture) - 64 Points")
print("="*70)
print(f"Device: {DEVICE}")
print(f"Points per sample: {NUM_POINTS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print("="*70)
print()

# Load data
print("Loading MOLA data...")
train_groups = np.load('data/pointnet/train_groups.npy')  # (10000, 1024, 7)
train_labels = np.load('data/pointnet/train_labels.npy')
test_groups = np.load('data/pointnet/test_groups.npy')
test_labels = np.load('data/pointnet/test_labels.npy')

print(f"  Train: {train_groups.shape}, Labels: {np.bincount(train_labels)}")
print(f"  Test: {test_groups.shape}, Labels: {np.bincount(test_labels)}")
print()

# Subsample to NUM_POINTS and extract XYZ only
def subsample_points(data, n_points):
    """Subsample from original points to n_points and extract XYZ only."""
    orig_points = data.shape[1]
    indices = np.linspace(0, orig_points-1, n_points, dtype=int)
    return data[:, indices, :3]  # Only XYZ coordinates

print(f"Preprocessing: extracting XYZ and subsampling to {NUM_POINTS} points...")
train_xyz = subsample_points(train_groups, NUM_POINTS)
test_xyz = subsample_points(test_groups, NUM_POINTS)
print(f"  Train shape: {train_xyz.shape}")
print(f"  Test shape: {test_xyz.shape}")
print()

# Create datasets
train_dataset = TensorDataset(
    torch.FloatTensor(train_xyz),
    torch.LongTensor(train_labels)
)
test_dataset = TensorDataset(
    torch.FloatTensor(test_xyz),
    torch.LongTensor(test_labels)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# Create model - EXACTLY like 3DCertify
print("Creating PointNet model (3DCertify architecture)...")
model = PointNet(
    number_points=NUM_POINTS,
    num_classes=NUM_CLASSES,
    max_features=1024,
    pool_function='improved_max',  # Cascaded MaxPool for better verification
    transposed_input=False  # Input is (batch, n_points, 3)
).to(DEVICE)

print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
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
            print(f"  [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f}, "
                  f"Acc: {100.*correct/total:.2f}%")

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
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

    accuracy = 100. * correct / total
    return accuracy

# Train
print("="*70)
print("Training...")
print("="*70)
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
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'test_accuracy': test_acc,
            'train_accuracy': train_acc,
            'num_points': NUM_POINTS,
            'num_classes': NUM_CLASSES,
            'architecture': '3DCertify',
            'pool_function': 'improved_max',
        }, 'models/pointnet_3dcertify_64p.pth')
        print(f"  ✓ Best model saved! (Test Acc: {test_acc:.2f}%)")

    print()

print("="*70)
print("Training complete!")
print(f"Best test accuracy: {best_acc:.2f}%")
print(f"Model saved to: models/pointnet_3dcertify_64p.pth")
print("="*70)
