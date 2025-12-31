"""Tests for ONNX export functionality."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import create_model


class TestONNXExport:
    """Test suite for ONNX export functions."""

    def test_export_creates_file(self):
        """Test that export creates an ONNX file."""
        from export_onnx import export_to_onnx

        model = create_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.onnx"
            export_to_onnx(model, output_path)
            assert output_path.exists()

    def test_exported_model_valid(self):
        """Test that exported ONNX model passes validation."""
        from export_onnx import export_to_onnx, validate_onnx_model

        model = create_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.onnx"
            export_to_onnx(model, output_path)
            assert validate_onnx_model(output_path)

    def test_pytorch_onnx_output_match(self):
        """Test that PyTorch and ONNX outputs match."""
        from export_onnx import compare_outputs, export_to_onnx

        model = create_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.onnx"
            export_to_onnx(model, output_path)
            assert compare_outputs(model, output_path, num_samples=50)

    def test_export_with_custom_opset(self):
        """Test export with different opset versions."""
        from export_onnx import export_to_onnx, validate_onnx_model

        model = create_model()

        for opset in [11, 12, 13]:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / f"test_model_opset{opset}.onnx"
                export_to_onnx(model, output_path, opset_version=opset)
                assert validate_onnx_model(output_path)


class TestONNXInfo:
    """Test ONNX model information extraction."""

    def test_print_onnx_info(self, capsys):
        """Test that print_onnx_info outputs model details."""
        from export_onnx import export_to_onnx, print_onnx_info

        model = create_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.onnx"
            export_to_onnx(model, output_path)
            print_onnx_info(output_path)

            captured = capsys.readouterr()
            assert "ONNX Model Information" in captured.out
            assert "input" in captured.out.lower()
            assert "output" in captured.out.lower()
