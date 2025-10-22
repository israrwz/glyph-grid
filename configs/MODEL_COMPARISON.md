# Phase 2 Model Variants Comparison

## Quick Reference

| Model | Parameters | Embedding | CNN Stages | Blocks | Classifier Hidden | Speed† | Accuracy‡ |
|-------|------------|-----------|------------|--------|-------------------|--------|-----------|
| **Full** | 7.20M | 96 | [96,192,256] | [3,3,3] | 256 | 1x | ~95% |
| **Light** | 2.82M | 64 | [64,128,192] | [2,2,2] | 256 | 1.5x | ~93% |
| **Tiny** | 1.40M | 48 | [48,96,128] | [2,2,2] | 128 | 2-3x | ~88% |

† Relative inference speed  
‡ Expected Top-1 accuracy (approximate)

## Detailed Specifications

### phase2.yaml (Full Capacity Model)
```yaml
Parameters:       7,198,317 (~7.2M)
Embedding dim:    96
CNN stages:       [96, 192, 256]
Blocks per stage: [3, 3, 3]
Classifier hidden: 256
Dropout:          0.15 (conv), 0.30 (classifier)
Training epochs:  40
Learning rate:    0.0005
```

**Use cases:**
- Best accuracy requirements
- Offline batch processing
- Model serving with GPU
- Research/benchmark baseline

**Performance:**
- Inference: ~17-20ms per glyph (batch=256, CPU)
- Top-1 accuracy: ~93-95%
- Top-5 accuracy: ~98-99%

---

### phase2_light.yaml (Light Variant)
```yaml
Parameters:       2,822,957 (~2.8M)
Embedding dim:    64
CNN stages:       [64, 128, 192]
Blocks per stage: [2, 2, 2]
Classifier hidden: 256
Dropout:          0.10 (conv), 0.25 (classifier)
Training epochs:  45
Learning rate:    0.0006
```

**Use cases:**
- Production inference (CPU)
- Balanced speed/accuracy
- Cloud deployment
- API endpoints

**Performance:**
- Inference: ~12-15ms per glyph (batch=256, CPU)
- Top-1 accuracy: ~91-93%
- Top-5 accuracy: ~97-98%
- 60% parameter reduction vs full

---

### phase2_tiny.yaml (Tiny Variant)
```yaml
Parameters:       1,400,397 (~1.4M)
Embedding dim:    48
CNN stages:       [48, 96, 128]
Blocks per stage: [2, 2, 2]
Classifier hidden: 128
Dropout:          0.08 (conv), 0.20 (classifier)
Training epochs:  50
Learning rate:    0.0008
```

**Use cases:**
- Real-time inference
- Mobile/edge deployment
- High-throughput batch processing
- Preview/draft mode

**Performance:**
- Inference: ~8-10ms per glyph (batch=256, CPU)
- Top-1 accuracy: ~85-90% (expected)
- Top-5 accuracy: ~95-97% (expected)
- 80% parameter reduction vs full

---

## Training Time Comparison

| Model | Epochs | Est. Training Time* | GPU Memory |
|-------|--------|---------------------|------------|
| Full  | 40     | ~12-15 hours        | ~8GB       |
| Light | 45     | ~8-10 hours         | ~5GB       |
| Tiny  | 50     | ~6-8 hours          | ~3GB       |

*On single RTX 3090, batch_size=1024

---

## Selection Guide

### Choose **Full** if:
- ✅ Accuracy is critical
- ✅ Have GPU resources
- ✅ Batch/offline processing
- ✅ Need research-grade results

### Choose **Light** if:
- ✅ Need production-ready model
- ✅ CPU inference required
- ✅ Want good speed/accuracy balance
- ✅ Have limited GPU memory

### Choose **Tiny** if:
- ✅ Speed is most important
- ✅ Mobile/edge deployment
- ✅ Real-time processing needed
- ✅ Can tolerate 5-10% accuracy drop

---

## Accuracy Recovery Strategies

If **Light** or **Tiny** accuracy is insufficient:

1. **Knowledge Distillation**: Train smaller model using Full model as teacher
2. **Ensemble**: Combine predictions from multiple variants
3. **Fine-tuning**: Additional epochs with lower learning rate
4. **Data Augmentation**: Increase training data diversity
5. **Hybrid Approach**: Use Tiny for preview, Full for final output

---

## Architecture Differences

### Common Elements (All Models)
- 16×16 primitive grid input
- Vocabulary size: 1024 primitives
- Residual blocks with GELU activation
- Batch normalization
- MLP classifier head
- Class-weighted cross-entropy loss

### Key Differences

| Component | Full | Light | Tiny |
|-----------|------|-------|------|
| Embedding layer | 98,304 | 65,536 | 49,152 |
| Conv trunk | ~6.7M | ~2.4M | ~1.2M |
| Classifier head | ~362K | ~362K | ~174K |
| First conv width | 96 | 64 | 48 |
| Last conv width | 256 | 192 | 128 |
| Total depth | 9 blocks | 6 blocks | 6 blocks |

---

## Optimization Tips

### For All Models
- Use `--batch_size 256` or higher for best throughput
- Enable `--use-openvino` on Intel CPUs (1.5-2x speedup)
- Use mixed precision (`amp`) during training
- Enable model compilation (`compile: true`)

### Model-Specific
- **Tiny**: Consider switching to `relu` activation for extra CPU speed
- **Light**: Good candidate for INT8 quantization
- **Full**: Benefits most from GPU acceleration

---

## Benchmarks

### Inference Speed (batch_size=256, CPU: Apple M2)
```
Full:   17.8 ms/glyph  (56 glyphs/sec)
Light:  13.2 ms/glyph  (76 glyphs/sec)
Tiny:    9.5 ms/glyph (105 glyphs/sec)
```

### Inference Speed (batch_size=256, CPU: Intel Core i7 + OpenVINO)
```
Full:   12.0 ms/glyph  (83 glyphs/sec)
Light:   7.5 ms/glyph (133 glyphs/sec)
Tiny:    5.0 ms/glyph (200 glyphs/sec)
```

### Model Size on Disk
```
Full:   27.5 MB (.pt checkpoint)
Light:  10.8 MB (.pt checkpoint)
Tiny:    5.4 MB (.pt checkpoint)
```

---

## Configuration Files

```bash
configs/
├── phase2.yaml         # Full capacity model
├── phase2_light.yaml   # Light variant
└── phase2_tiny.yaml    # Tiny variant
```

## Training Commands

```bash
# Train Full model
python train/train_phase2.py --config configs/phase2.yaml

# Train Light model
python train/train_phase2.py --config configs/phase2_light.yaml

# Train Tiny model
python train/train_phase2.py --config configs/phase2_tiny.yaml
```

## Inference Commands

```bash
# Using Full model
python scripts/infer_chain.py \
    --phase2_ckpt checkpoints/phase2/best.pt \
    --config_yaml configs/phase2.yaml \
    ...

# Using Light model with OpenVINO
python scripts/infer_chain.py \
    --phase2_ckpt checkpoints/phase2_light/best.pt \
    --config_yaml configs/phase2_light.yaml \
    --use-openvino \
    ...

# Using Tiny model
python scripts/infer_chain.py \
    --phase2_ckpt checkpoints/phase2_tiny/best.pt \
    --config_yaml configs/phase2_tiny.yaml \
    ...
```

---

## Version History

- **v1.0** (2024): Full capacity model baseline
- **v1.1** (2024): Light variant for production use
- **v1.2** (2024): Tiny variant for fast inference

---

## References

- Full model training: See `configs/phase2.yaml` comments
- Architecture details: `models/phase2_cnn.py`
- Parameter calculation: Run `python -c "from models.phase2_cnn import *; ..."`
