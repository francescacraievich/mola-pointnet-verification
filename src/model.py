"""
MLP Model for LiDAR Point Classification

Architecture designed for efficient formal verification with αβ-CROWN:
- Input: 3 features (x, y, z coordinates)
- Layer 1: Linear(3 → 256) + ReLU
- Layer 2: Linear(256 → 256) + ReLU
- Layer 3: Linear(256 → 128) + ReLU
- Layer 4: Linear(128 → 3) (output logits for 3 classes)

Total parameters: ~100K
No BatchNorm or Dropout (these complicate verification)

Classes:
    0 - GROUND: drivable terrain
    1 - OBSTACLE: walls, objects (safety-critical)
    2 - OTHER: sky, far points, noise

Properties to verify:
    1. Local Robustness: ∀x' : ||x' - x₀||_∞ ≤ ε → f(x') = f(x₀)
    2. Safety: ∀x' : ||x' - x₀||_∞ ≤ ε ∧ f(x₀)=OBSTACLE → f(x') ≠ GROUND
"""

from typing import Tuple

import torch
import torch.nn as nn

# Class indices
CLASS_GROUND = 0
CLASS_OBSTACLE = 1
CLASS_OTHER = 2

# Class names for display
CLASS_NAMES = {CLASS_GROUND: "GROUND", CLASS_OBSTACLE: "OBSTACLE", CLASS_OTHER: "OTHER"}


class MLPClassifier(nn.Module):
    """
    MLP classifier for LiDAR point classification.

    Designed for formal verification:
    - Simple feedforward architecture
    - ReLU activations (piecewise linear, good for verification)
    - No BatchNorm/Dropout (complicate bound propagation)

    Architecture:
        Input(3) → Linear(256) → ReLU → Linear(256) → ReLU
                 → Linear(128) → ReLU → Linear(3)
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dims: Tuple[int, ...] = (256, 256, 128),
        num_classes: int = 3,
    ):
        """
        Initialize MLP classifier.

        Args:
            input_dim: Number of input features (default 3 for x,y,z)
            hidden_dims: Tuple of hidden layer dimensions
            num_classes: Number of output classes
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer (no activation - raw logits for CrossEntropyLoss)
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming initialization for ReLU."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3) or (batch_size, n_points, 3)

        Returns:
            Logits tensor of shape (batch_size, 3) or (batch_size, n_points, 3)
        """
        return self.network(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predicted class labels.

        Args:
            x: Input tensor

        Returns:
            Predicted class indices
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predicted class probabilities.

        Args:
            x: Input tensor

        Returns:
            Class probabilities (softmax of logits)
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_architecture_summary(self) -> str:
        """Get a string summary of the architecture."""
        lines = [
            "MLP Classifier Architecture",
            "=" * 40,
            f"Input dimension: {self.input_dim}",
            f"Hidden dimensions: {self.hidden_dims}",
            f"Output classes: {self.num_classes}",
            f"Total parameters: {self.count_parameters():,}",
            "",
            "Layer structure:",
        ]

        prev_dim = self.input_dim
        for i, hidden_dim in enumerate(self.hidden_dims):
            lines.append(f"  Layer {i+1}: Linear({prev_dim} → {hidden_dim}) + ReLU")
            prev_dim = hidden_dim
        lines.append(f"  Output: Linear({prev_dim} → {self.num_classes})")

        return "\n".join(lines)


def create_model(
    input_dim: int = 3, hidden_dims: Tuple[int, ...] = (256, 256, 128), num_classes: int = 3
) -> MLPClassifier:
    """
    Factory function to create the MLP model.

    Args:
        input_dim: Number of input features
        hidden_dims: Tuple of hidden layer sizes
        num_classes: Number of output classes

    Returns:
        MLPClassifier instance
    """
    return MLPClassifier(input_dim=input_dim, hidden_dims=hidden_dims, num_classes=num_classes)


def load_model(path: str, device: str = "cpu") -> MLPClassifier:
    """
    Load a trained model from checkpoint.

    Args:
        path: Path to the .pth file
        device: Device to load the model on

    Returns:
        Loaded MLPClassifier
    """
    checkpoint = torch.load(path, map_location=device)

    # Handle both full checkpoint and state_dict only
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        # Try to get architecture params if saved
        input_dim = checkpoint.get("input_dim", 3)
        hidden_dims = checkpoint.get("hidden_dims", (256, 256, 128))
        num_classes = checkpoint.get("num_classes", 3)
    else:
        state_dict = checkpoint
        input_dim = 3
        hidden_dims = (256, 256, 128)
        num_classes = 3

    model = create_model(input_dim, hidden_dims, num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


if __name__ == "__main__":
    # Test the model
    print("Testing MLP Classifier...")

    # Create model
    model = create_model()
    print(model.get_architecture_summary())
    print()

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 3)

    # Forward pass
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")

    # Predictions
    predictions = model.predict(x)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5].tolist()}")

    # Probabilities
    probs = model.predict_proba(x)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Sample probabilities:\n{probs[:3]}")

    # Verify architecture is verification-friendly
    print("\nVerification-friendly checks:")
    print(f"  - No BatchNorm: {not any(isinstance(m, nn.BatchNorm1d) for m in model.modules())}")
    print(f"  - No Dropout: {not any(isinstance(m, nn.Dropout) for m in model.modules())}")
    print(
        f"  - Only ReLU activations: {all(isinstance(m, (nn.Linear, nn.ReLU, nn.Sequential)) for m in model.modules())}"
    )

    print("\nModel test passed!")
