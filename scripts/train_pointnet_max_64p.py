#!/usr/bin/env python3
"""
Train PointNet with single MaxPool (pool_function='max') for auto_LiRPA verification.

This model is compatible with auto_LiRPA but may have slightly worse verification bounds
compared to improved_max. Used for comparison with ERAN.
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, '3dcertify')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

from pointnet.model import PointNet

print("="*70)
print("Training PointNet with Single MaxPool (auto_LiRPA compatible)")
print("="*70)
print()

# Configuration
TRAIN_DATA_PATH = 'data/pointnet/train_groups.npy'
TRAIN_LABELS_PATH = 'data/pointnet/train_labels.npy'
TEST_DATA_PATH = 'data/pointnet/test_groups.npy'
TEST_LABELS_PATH = 'data/pointnet/test_labels.npy'
MODEL_SAVE_PATH = 'models/pointnet_max_64p.pth'

NUM_POINTS = 64
NUM_CLASSES = 2
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Configuration:")
print(f"  Device: {DEVICE}")
print(f"  Points per sample: {NUM_POINTS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Pooling: max (single MaxPool - auto_LiRPA compatible)")
print()

# Dataset
class PointCloudDataset(Dataset):
    def __init__(self, data_path, labels_path, n_points=64):
        self.data = np.load(data_path)  # (N, 1024, 7)
        self.labels = np.load(labels_path)
        self.n_points = n_points

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Subsample to n_points
        orig_points = self.data.shape[1]
        indices = np.linspace(0, orig_points-1, self.n_points, dtype=int)
        sample = self.data[idx, indices, :3]  # Only XYZ
        label = self.labels[idx]
        return torch.FloatTensor(sample), torch.LongTensor([label])

# Load datasets
print("Loading datasets...")
train_dataset = PointCloudDataset(TRAIN_DATA_PATH, TRAIN_LABELS_PATH, NUM_POINTS)
test_dataset = PointCloudDataset(TEST_DATA_PATH, TEST_LABELS_PATH, NUM_POINTS)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"  Train samples: {len(train_dataset)}")
print(f"  Test samples: {len(test_dataset)}")
print(f"  Label distribution (train): {dict(enumerate(np.bincount(train_dataset.labels)))}")
print()

# Create model with SINGLE MaxPool (auto_LiRPA compatible)
print("Creating model...")
model = PointNet(
    number_points=NUM_POINTS,
    num_classes=NUM_CLASSES,
    max_features=1024,
    pool_function='max',  # SINGLE MaxPool for auto_LiRPA compatibility!
    disable_assertions=True,
    transposed_input=False  # Input: (batch, n_points, 3)
).to(DEVICE)

print("  ✓ Model created with pool_function='max'")
print()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# Training loop
print("="*70)
print("Starting Training")
print("="*70)
print()

best_test_acc = 0.0
best_epoch = 0

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(DEVICE)
        labels = labels.squeeze().to(DEVICE)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

    train_acc = 100.0 * train_correct / train_total
    train_loss /= len(train_loader)

    # Test
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(DEVICE)
            labels = labels.squeeze().to(DEVICE)

            outputs = model(data)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = 100.0 * test_correct / test_total
    test_loss /= len(test_loader)

    # Update learning rate
    scheduler.step()

    # Print progress
    print(f"Epoch {epoch+1:2d}/{EPOCHS}: "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    # Save best model
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_epoch = epoch + 1

        Path(MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'architecture': 'PointNet_max_single_maxpool'
        }, MODEL_SAVE_PATH)
        print(f"  → Best model saved! (Test Acc: {test_acc:.2f}%)")

print()
print("="*70)
print("Training Complete!")
print("="*70)
print(f"Best Test Accuracy: {best_test_acc:.2f}% (Epoch {best_epoch})")
print(f"Model saved to: {MODEL_SAVE_PATH}")
print()
print("This model uses pool_function='max' (single MaxPool)")
print("Compatible with auto_LiRPA for verification!")
