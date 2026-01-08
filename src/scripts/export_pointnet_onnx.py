"""
Export trained PointNet to ONNX format for α,β-CROWN verification.

Creates a flattened-input model (64*3=192 inputs) compatible with α,β-CROWN.
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent  # mola-pointnet-verification/
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src" / "model"))

import onnx
import torch
from pointnet_autolirpa_compatible import PointNetAutoLiRPA


class FlattenedPointNet(torch.nn.Module):
    """Wrapper that accepts flattened input for ONNX export."""

    def __init__(self, model, n_points=64):
        super().__init__()
        self.model = model
        self.n_points = n_points

    def forward(self, x):
        # Reshape from (batch, n_points*3) to (batch, n_points, 3)
        x = x.view(-1, self.n_points, 3)
        return self.model(x)


def export_pointnet_onnx(
    model_path: Path,
    output_path: Path,
    n_points: int = 64,
    max_features: int = 512,
):
    """
    Export PointNet to ONNX with flattened input.

    Args:
        model_path: Path to trained model (.pth)
        output_path: Path for ONNX output
        n_points: Number of points (default 64)
        max_features: Max features in model (default 512)
    """
    print("=" * 60)
    print("Exporting PointNet to ONNX for α,β-CROWN")
    print("=" * 60)

    # Load checkpoint
    print("\n1. Loading model...")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)

    print(f"   n_points: {n_points}")
    print(f"   max_features: {max_features}")
    print(f"   test_accuracy: {checkpoint.get('test_accuracy', 'N/A')}%")

    # Create model
    model = PointNetAutoLiRPA(n_points, 2, max_features, use_batchnorm=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Wrap with flattened input handler
    wrapped_model = FlattenedPointNet(model, n_points)
    wrapped_model.eval()

    # Create dummy input - flattened for ONNX
    # # Input shape: (batch, n_points * 3) = (batch, 192) for 64 points

    dummy_input = torch.randn(1, n_points * 3)

    print(f"\n2. Exporting to ONNX...")
    print(f"   Input shape: {dummy_input.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapped_model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=12,
        do_constant_folding=True,
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print(f"   Saved to {output_path}")

    # Verify ONNX
    print("\n3. Validating ONNX...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("   ONNX model is valid!")

    # Print model info
    print("\n4. Model summary:")
    print(f"   Inputs: {[i.name for i in onnx_model.graph.input]}")
    print(f"   Outputs: {[o.name for o in onnx_model.graph.output]}")

    # Test inference: run the exported model on dummy data to verify
    # that the ONNX export works correctly before using it with α,β-CROWN.
    print("\n5. Testing inference...")
    import onnxruntime as ort

    session = ort.InferenceSession(str(output_path))
    test_input = dummy_input.numpy()
    outputs = session.run(None, {"input": test_input})
    print(f"   Output shape: {outputs[0].shape}")
    print(f"   Sample output: {outputs[0][0]}")

    print("\nExport complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=str(BASE_DIR / "saved_models/pointnet_autolirpa_512.pth"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(BASE_DIR / "saved_models/pointnet_autolirpa_flat.onnx"),
    )
    parser.add_argument("--n-points", type=int, default=64)
    parser.add_argument("--max-features", type=int, default=512)

    args = parser.parse_args()

    export_pointnet_onnx(
        model_path=Path(args.model),
        output_path=Path(args.output),
        n_points=args.n_points,
        max_features=args.max_features,
    )
