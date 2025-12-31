"""
Export PyTorch Model to ONNX Format

Exports the trained MLP model to ONNX format for use with αβ-CROWN verifier.
Includes validation to ensure the export is correct.
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from model import load_model


def export_to_onnx(
    model: torch.nn.Module,
    output_path: Path,
    input_dim: int = 3,
    opset_version: int = 12,
) -> None:
    """
    Export PyTorch model to ONNX format.

    Args:
        model: Trained PyTorch model
        output_path: Path to save the ONNX model
        input_dim: Number of input features
        opset_version: ONNX opset version (12 is good for αβ-CROWN compatibility)
    """
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, input_dim)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print(f"Model exported to {output_path}")


def validate_onnx_model(onnx_path: Path) -> bool:
    """
    Validate the exported ONNX model.

    Args:
        onnx_path: Path to the ONNX model

    Returns:
        True if validation passes
    """
    print(f"Validating ONNX model: {onnx_path}")

    # Load and check ONNX model
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    print("  ONNX model structure is valid")
    return True


def compare_outputs(
    pytorch_model: torch.nn.Module,
    onnx_path: Path,
    num_samples: int = 100,
    tolerance: float = 1e-5,
) -> bool:
    """
    Compare PyTorch and ONNX model outputs to ensure consistency.

    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to exported ONNX model
        num_samples: Number of random samples to test
        tolerance: Maximum allowed difference

    Returns:
        True if outputs match within tolerance
    """
    print(f"Comparing PyTorch and ONNX outputs ({num_samples} samples)...")

    pytorch_model.eval()

    # Create ONNX runtime session
    ort_session = ort.InferenceSession(str(onnx_path))

    # Test with random inputs
    max_diff = 0.0
    for i in range(num_samples):
        # Random input
        x = torch.randn(1, 3)

        # PyTorch output
        with torch.no_grad():
            pytorch_out = pytorch_model(x).numpy()

        # ONNX output
        onnx_out = ort_session.run(None, {"input": x.numpy()})[0]

        # Compare
        diff = np.abs(pytorch_out - onnx_out).max()
        max_diff = max(max_diff, diff)

        if diff > tolerance:
            print(f"  Sample {i}: Max difference {diff:.2e} exceeds tolerance {tolerance:.2e}")
            return False

    print(f"  Maximum difference: {max_diff:.2e}")
    print(f"  All outputs match within tolerance ({tolerance:.2e})")
    return True


def print_onnx_info(onnx_path: Path) -> None:
    """Print information about the ONNX model."""
    onnx_model = onnx.load(str(onnx_path))

    print("\nONNX Model Information:")
    print("=" * 40)
    print(f"  IR version: {onnx_model.ir_version}")
    print(f"  Opset version: {onnx_model.opset_import[0].version}")
    print(f"  Producer: {onnx_model.producer_name}")

    # Input info
    print("\n  Inputs:")
    for inp in onnx_model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"    {inp.name}: {shape}")

    # Output info
    print("\n  Outputs:")
    for out in onnx_model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"    {out.name}: {shape}")

    # Count nodes
    print(f"\n  Number of nodes: {len(onnx_model.graph.node)}")

    # List node types
    node_types = {}
    for node in onnx_model.graph.node:
        node_types[node.op_type] = node_types.get(node.op_type, 0) + 1

    print("  Node types:")
    for op_type, count in sorted(node_types.items()):
        print(f"    {op_type}: {count}")

    print("=" * 40)


def main():
    parser = argparse.ArgumentParser(description="Export MLP model to ONNX format")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/mlp_lidar.pth",
        help="Path to PyTorch model checkpoint",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="models/mlp_lidar.onnx",
        help="Path to save ONNX model",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=12,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip output comparison validation",
    )

    args = parser.parse_args()

    model_path = Path(args.model_path)
    output_path = Path(args.output_path)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load PyTorch model
    print(f"Loading PyTorch model from {model_path}...")
    model = load_model(str(model_path))
    print(f"  Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Export to ONNX
    export_to_onnx(model, output_path, opset_version=args.opset_version)

    # Validate ONNX model structure
    validate_onnx_model(output_path)

    # Compare outputs
    if not args.skip_validation:
        outputs_match = compare_outputs(model, output_path)
        if not outputs_match:
            print("\nWARNING: PyTorch and ONNX outputs do not match!")
            return 1

    # Print ONNX model info
    print_onnx_info(output_path)

    print("\nONNX export complete!")
    print(f"Model saved to: {output_path}")
    print("\nThe model is ready for αβ-CROWN verification.")

    return 0


if __name__ == "__main__":
    exit(main())
