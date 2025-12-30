"""
Training Script for MLP LiDAR Point Classifier

Trains the MLP model on processed LiDAR point cloud data.
- Loss: CrossEntropyLoss
- Optimizer: Adam (lr=0.001)
- Target accuracy: >80% on test set
- Saves training metrics and best model checkpoint
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model import CLASS_NAMES, create_model


def load_processed_data(data_dir: Path) -> Tuple[np.ndarray, ...]:
    """Load processed data from numpy files."""
    train_points = np.load(data_dir / "train_points.npy")
    train_labels = np.load(data_dir / "train_labels.npy")
    test_points = np.load(data_dir / "test_points.npy")
    test_labels = np.load(data_dir / "test_labels.npy")

    print(f"Loaded training data: {train_points.shape}")
    print(f"Loaded test data: {test_points.shape}")

    return train_points, train_labels, test_points, test_labels


def create_dataloaders(
    train_points: np.ndarray,
    train_labels: np.ndarray,
    test_points: np.ndarray,
    test_labels: np.ndarray,
    batch_size: int = 256,
) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders from numpy arrays."""
    # Convert to tensors
    train_dataset = TensorDataset(
        torch.from_numpy(train_points).float(), torch.from_numpy(train_labels).long()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(test_points).float(), torch.from_numpy(test_labels).long()
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, test_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for points, labels in train_loader:
        points, labels = points.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(points)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * points.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module, test_loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float, Dict[int, Dict[str, float]]]:
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    # Per-class statistics
    class_correct = {0: 0, 1: 0, 2: 0}
    class_total = {0: 0, 1: 0, 2: 0}

    for points, labels in test_loader:
        points, labels = points.to(device), labels.to(device)

        outputs = model(points)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * points.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        # Per-class accuracy
        for cls in range(3):
            mask = labels == cls
            class_correct[cls] += predicted[mask].eq(labels[mask]).sum().item()
            class_total[cls] += mask.sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total

    # Per-class metrics
    per_class = {}
    for cls in range(3):
        if class_total[cls] > 0:
            per_class[cls] = {
                "accuracy": 100.0 * class_correct[cls] / class_total[cls],
                "correct": class_correct[cls],
                "total": class_total[cls],
            }

    return avg_loss, accuracy, per_class


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 0.001,
    save_dir: Path = Path("models"),
    results_dir: Path = Path("results"),
) -> Dict:
    """Full training loop."""

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "per_class_acc": [],
        "best_epoch": 0,
        "best_acc": 0.0,
    }

    best_acc = 0.0
    save_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining on {device}")
    print(f"Epochs: {epochs}, Learning rate: {lr}")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate
        test_loss, test_acc, per_class = evaluate(model, test_loader, criterion, device)

        # Update scheduler
        scheduler.step(test_acc)

        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["per_class_acc"].append(
            {CLASS_NAMES[k]: v["accuracy"] for k, v in per_class.items()}
        )

        # Print progress
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
        )

        # Print per-class accuracy every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print("  Per-class accuracy:")
            for cls, metrics in per_class.items():
                print(f"    {CLASS_NAMES[cls]}: {metrics['accuracy']:.2f}%")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            history["best_epoch"] = epoch
            history["best_acc"] = best_acc

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_acc": test_acc,
                "test_loss": test_loss,
                "input_dim": model.input_dim,
                "hidden_dims": model.hidden_dims,
                "num_classes": model.num_classes,
            }
            torch.save(checkpoint, save_dir / "mlp_lidar.pth")
            print(f"  -> New best model saved! (Acc: {best_acc:.2f}%)")

    print("=" * 60)
    print(f"Training complete!")
    print(f"Best test accuracy: {history['best_acc']:.2f}% (Epoch {history['best_epoch']})")

    # Save training log
    log_path = results_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training log saved to {log_path}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train MLP LiDAR classifier")
    parser.add_argument(
        "--data-dir", type=str, default="data/processed", help="Directory with processed data"
    )
    parser.add_argument(
        "--save-dir", type=str, default="models", help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Directory to save training logs"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data_dir = Path(args.data_dir)
    train_points, train_labels, test_points, test_labels = load_processed_data(data_dir)

    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        train_points, train_labels, test_points, test_labels, batch_size=args.batch_size
    )

    # Create model
    model = create_model()
    model = model.to(device)
    print(f"\n{model.get_architecture_summary()}\n")

    # Train
    history = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        save_dir=Path(args.save_dir),
        results_dir=Path(args.results_dir),
    )

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    # Load best model
    checkpoint = torch.load(Path(args.save_dir) / "mlp_lidar.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, per_class = evaluate(model, test_loader, criterion, device)

    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    print("\nPer-class accuracy:")
    for cls, metrics in per_class.items():
        print(
            f"  {CLASS_NAMES[cls]}: {metrics['accuracy']:.2f}% "
            f"({metrics['correct']}/{metrics['total']})"
        )

    # Check if target reached
    target_acc = 80.0
    if test_acc >= target_acc:
        print(f"\n✓ Target accuracy ({target_acc}%) REACHED!")
    else:
        print(
            f"\n✗ Target accuracy ({target_acc}%) not reached. "
            f"Consider more epochs or tuning hyperparameters."
        )


if __name__ == "__main__":
    main()
