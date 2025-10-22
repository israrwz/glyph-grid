# OpenVINO Acceleration for Phase 2 Inference

This document describes how to use OpenVINO to accelerate Phase 2 glyph classification inference.

## Overview

OpenVINO is Intel's toolkit for optimizing and deploying deep learning models. It can significantly improve inference speed on Intel CPUs through various optimizations including:

- Graph optimizations
- Low-precision inference
- CPU-specific kernel optimizations
- Multi-threading

## Installation

Install OpenVINO and its development tools:

```bash
pip install openvino openvino-dev
```

## Usage

### Basic Usage

To enable OpenVINO inference, add the `--use-openvino` flag to your inference command:

```bash
python scripts/infer_chain.py \
    --rasters_dir data/rasters \
    --phase1_ckpt checkpoints/phase1/best.pt \
    --phase2_ckpt checkpoints/phase2/best.pt \
    --label_map data/grids_memmap/label_map.json \
    --chars_csv data/chars.csv \
    --batch_size 256 \
    --use-openvino
```

### Advanced Options

#### Device Selection

Specify the target device for OpenVINO inference (default is CPU):

```bash
python scripts/infer_chain.py \
    ... \
    --use-openvino \
    --openvino-device CPU
```

Available devices:
- `CPU` - Intel CPU (default)
- `GPU` - Intel integrated GPU (requires additional setup)
- `AUTO` - Automatic device selection

#### Cache Directory

OpenVINO models are cached after first conversion. Specify a custom cache directory:

```bash
python scripts/infer_chain.py \
    ... \
    --use-openvino \
    --openvino-cache-dir /path/to/cache
```

Default cache location: `cache/openvino/`

## How It Works

1. **First Run**: 
   - The PyTorch Phase 2 model is exported to ONNX format
   - ONNX model is converted to OpenVINO IR (Intermediate Representation)
   - Both are cached for future use
   - This adds ~5-10 seconds to the first run

2. **Subsequent Runs**:
   - Cached OpenVINO model is loaded directly
   - Inference uses optimized OpenVINO runtime
   - No conversion overhead

## Performance

Expected speedup depends on hardware and model size:

- **Intel CPUs**: 1.5x - 3x faster than PyTorch
- **Batch sizes**: Works best with batch_size >= 64
- **Model size**: Larger models benefit more from optimization

Example timing (on Intel Core i7):
```
Without OpenVINO: ~17ms per glyph (batch_size=256)
With OpenVINO:    ~8-12ms per glyph (batch_size=256)
```

## Troubleshooting

### Installation Issues

If you encounter import errors:

```bash
# Try upgrading pip first
pip install --upgrade pip

# Install with specific versions
pip install openvino==2023.2.0 openvino-dev==2023.2.0
```

### Conversion Failures

If ONNX or OpenVINO conversion fails:

1. Clear the cache: `rm -rf cache/openvino/`
2. Try without OpenVINO first to verify the model works
3. Check that your PyTorch model doesn't use unsupported operations

### Performance Issues

If OpenVINO is slower than expected:

1. Ensure you're using a reasonable batch size (64-256)
2. Check CPU utilization - OpenVINO should use multiple cores
3. Verify you're running on Intel hardware (OpenVINO is optimized for Intel)
4. Try different batch sizes to find optimal performance

## Limitations

- **GPU Support**: Requires Intel integrated or discrete GPU with additional drivers
- **Model Types**: Currently supports CNN-based Phase 2 models
- **Dynamic Shapes**: Batch size is dynamic, but grid size (16x16) is fixed
- **Platform**: Optimized for Intel CPUs; may not provide speedup on AMD/ARM

## Cache Management

The OpenVINO cache contains:
- `phase2_model.onnx` - Exported ONNX model
- `openvino_ir/phase2_model.xml` - OpenVINO model graph
- `openvino_ir/phase2_model.bin` - OpenVINO model weights

To force re-conversion (e.g., after model updates):
```bash
rm -rf cache/openvino/
```

## Comparison with Other Optimizations

| Method | Speedup | Compatibility | Setup |
|--------|---------|---------------|-------|
| Batching | 3x | ✅ All models | None |
| OpenVINO | 1.5-3x | ✅ Most models | pip install |
| TorchScript JIT | 1.1x | ⚠️ Limited | None |
| ONNX Runtime | 1.2x | ✅ Most models | pip install |

**Recommendation**: Use batching (already enabled) + OpenVINO for best performance.

## References

- [OpenVINO Documentation](https://docs.openvino.ai/)
- [Model Optimizer Guide](https://docs.openvino.ai/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
- [OpenVINO Python API](https://docs.openvino.ai/latest/api/ie_python_api/api.html)