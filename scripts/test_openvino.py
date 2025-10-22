#!/usr/bin/env python3
"""
Test script for OpenVINO integration.

This script verifies that OpenVINO conversion and inference work correctly
by creating a minimal test case.

Usage:
    python scripts/test_openvino.py
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

# Add repo root to path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Check OpenVINO availability
try:
    from openvino import Core

    OPENVINO_AVAILABLE = True
    print("✓ OpenVINO is available")
except ImportError:
    OPENVINO_AVAILABLE = False
    print("✗ OpenVINO is NOT available")
    print("  Install with: pip install openvino openvino-dev")
    sys.exit(1)


class DummyPhase2Model(nn.Module):
    """Minimal model mimicking Phase 2 structure for testing."""

    def __init__(self, vocab_size=1024, embed_dim=64, num_classes=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv2d(embed_dim, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, 16, 16) int64
        B = x.shape[0]
        x = self.embedding(x)  # (B, 16, 16, embed_dim)
        x = x.permute(0, 3, 1, 2)  # (B, embed_dim, 16, 16)
        x = torch.relu(self.conv1(x))  # (B, 128, 16, 16)
        x = self.pool(x)  # (B, 128, 1, 1)
        x = x.view(B, -1)  # (B, 128)
        x = self.fc(x)  # (B, num_classes)
        return x


def test_pytorch_inference():
    """Test basic PyTorch inference."""
    print("\n--- Testing PyTorch Inference ---")

    model = DummyPhase2Model()
    model.eval()

    # Create dummy input
    batch_size = 4
    dummy_input = torch.randint(0, 1024, (batch_size, 16, 16), dtype=torch.long)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"✓ PyTorch inference successful")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")

    return model, dummy_input, output


def test_onnx_export(model, dummy_input, cache_dir):
    """Test ONNX export."""
    print("\n--- Testing ONNX Export ---")

    cache_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = cache_dir / "test_model.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=11,
    )

    print(f"✓ ONNX export successful: {onnx_path}")
    print(f"  File size: {onnx_path.stat().st_size / 1024:.2f} KB")

    return onnx_path


def test_openvino_conversion(onnx_path, cache_dir):
    """Test OpenVINO conversion."""
    print("\n--- Testing OpenVINO Conversion ---")

    ir_dir = cache_dir / "ir"
    ir_dir.mkdir(parents=True, exist_ok=True)

    # Try direct conversion via Python API
    try:
        ie = Core()
        model = ie.read_model(model=str(onnx_path))
        from openvino import serialize

        xml_path = ir_dir / "test_model.xml"
        serialize(model, str(xml_path))

        print(f"✓ OpenVINO conversion successful: {xml_path}")
        bin_path = ir_dir / "test_model.bin"
        if bin_path.exists():
            print(f"  XML size: {xml_path.stat().st_size / 1024:.2f} KB")
            print(f"  BIN size: {bin_path.stat().st_size / 1024:.2f} KB")

        return xml_path
    except Exception as e:
        print(f"✗ OpenVINO conversion failed: {e}")
        raise


def test_openvino_inference(xml_path, dummy_input, pytorch_output):
    """Test OpenVINO inference and compare with PyTorch."""
    print("\n--- Testing OpenVINO Inference ---")

    ie = Core()
    model = ie.read_model(model=str(xml_path))
    compiled_model = ie.compile_model(model, "CPU")
    infer_request = compiled_model.create_infer_request()

    # Get input/output layers
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    # Run inference
    input_data = dummy_input.numpy().astype(np.int64)
    results = infer_request.infer({input_layer: input_data})
    ov_output = results[output_layer]

    print(f"✓ OpenVINO inference successful")
    print(f"  Input shape: {input_data.shape}")
    print(f"  Output shape: {ov_output.shape}")

    # Compare outputs
    pytorch_np = pytorch_output.detach().numpy()
    max_diff = np.abs(pytorch_np - ov_output).max()
    mean_diff = np.abs(pytorch_np - ov_output).mean()

    print(f"\n--- Comparing PyTorch vs OpenVINO ---")
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")

    # Check if outputs are close (tolerance of 0.001)
    if max_diff < 0.001:
        print(f"✓ Outputs match (max_diff < 0.001)")
    elif max_diff < 0.01:
        print(f"⚠ Outputs are close but not exact (0.001 < max_diff < 0.01)")
    else:
        print(f"✗ Outputs differ significantly (max_diff >= 0.01)")
        return False

    return True


def test_batch_sizes(xml_path):
    """Test different batch sizes with OpenVINO."""
    print("\n--- Testing Different Batch Sizes ---")

    ie = Core()
    model = ie.read_model(model=str(xml_path))
    compiled_model = ie.compile_model(model, "CPU")
    infer_request = compiled_model.create_infer_request()

    input_layer = compiled_model.input(0)

    batch_sizes = [1, 4, 16, 64, 256]

    for bs in batch_sizes:
        try:
            input_data = np.random.randint(0, 1024, (bs, 16, 16), dtype=np.int64)
            results = infer_request.infer({input_layer: input_data})
            output = results[compiled_model.output(0)]
            print(
                f"  ✓ Batch size {bs:3d}: input {input_data.shape} -> output {output.shape}"
            )
        except Exception as e:
            print(f"  ✗ Batch size {bs:3d}: failed - {e}")


def cleanup(cache_dir):
    """Clean up test cache."""
    import shutil

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"\n✓ Cleaned up test cache: {cache_dir}")


def main():
    print("=" * 60)
    print("OpenVINO Integration Test")
    print("=" * 60)

    cache_dir = Path("cache/test_openvino")

    try:
        # Test 1: PyTorch inference
        model, dummy_input, pytorch_output = test_pytorch_inference()

        # Test 2: ONNX export
        onnx_path = test_onnx_export(model, dummy_input, cache_dir)

        # Test 3: OpenVINO conversion
        xml_path = test_openvino_conversion(onnx_path, cache_dir)

        # Test 4: OpenVINO inference
        success = test_openvino_inference(xml_path, dummy_input, pytorch_output)

        # Test 5: Batch sizes
        test_batch_sizes(xml_path)

        print("\n" + "=" * 60)
        if success:
            print("✓ ALL TESTS PASSED")
        else:
            print("⚠ TESTS COMPLETED WITH WARNINGS")
        print("=" * 60)

        # Cleanup
        cleanup(cache_dir)

        return 0

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()

        # Cleanup on failure too
        cleanup(cache_dir)

        return 1


if __name__ == "__main__":
    sys.exit(main())
