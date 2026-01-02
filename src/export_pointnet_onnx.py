"""
Export trained PointNet to ONNX format for αβ-CROWN verification.
"""

import argparse
from pathlib import Path

import torch
import onnx

from pointnet_model import PointNetForVerification


def export_pointnet_onnx(
    model_path: Path,
    output_path: Path,
    n_points: int = 64,
):
    """
    Export PointNet to ONNX.

    Args:
        model_path: Path to trained model (.pth)
        output_path: Path for ONNX output
    """
    print("=" * 60)
    print("Exporting PointNet to ONNX")
    print("=" * 60)

    # Load checkpoint
    print("\n1. Loading model...")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    n_points = checkpoint.get("n_points", n_points)
    num_classes = checkpoint.get("num_classes", 2)
    use_tnet = checkpoint.get("use_tnet", True)

    print(f"   n_points: {n_points}")
    print(f"   num_classes: {num_classes}")
    print(f"   use_tnet: {use_tnet}")
    print(f"   test_accuracy: {checkpoint.get('test_accuracy', 'N/A'):.2f}%")

    # Create model
    model = PointNetForVerification(
        num_points=n_points,
        num_classes=num_classes,
        use_tnet=use_tnet,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create dummy input - flattened for ONNX
    # Input shape: (batch, n_points * 3) = (batch, 192) for 64 points
    dummy_input = torch.randn(1, n_points * 3)

    print(f"\n2. Exporting to ONNX...")
    print(f"   Input shape: {dummy_input.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
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

    # Test inference
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
    parser.add_argument("--model", type=str, default="models/pointnet.pth")
    parser.add_argument("--output", type=str, default="models/pointnet.onnx")
    parser.add_argument("--n-points", type=int, default=64)

    args = parser.parse_args()

    export_pointnet_onnx(
        model_path=Path(args.model),
        output_path=Path(args.output),
        n_points=args.n_points,
    )
