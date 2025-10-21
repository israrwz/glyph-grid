# Unified MLP Glyph Recognition Plan (`mlp-plan.md`)
## Fresh Kaggle Notebook Approach

## 1. Motivation & Pivot
Phase 2 CNN inference (~7s per font) is still too slow vs Phase 1 (~1s). We pivot to a single unified model that:
- Directly consumes the 128×128 raster.
- Internally learns primitive-like cell features (auxiliary supervision).
- Uses a lightweight MLP mixer (no deep convolutional stack after stem).
- **Uses PyTorch Lightning for minimal boilerplate** (typical Kaggle pattern).
Goals: <2s per font end‑to‑end, ≤3M params, maintain or improve overall Unicode glyph accuracy, keep primitive interpretability for debugging.

**Key Simplification**: Use PyTorch Lightning to reduce training code from 1000+ lines to ~100 lines, matching typical Kaggle notebook patterns.

## 2. Available Data (Clean Kaggle Start)
We assume a minimal fresh repository with only:
- `dataset/rasters/*.png` : 128×128 glyph raster images (binary or thresholdable grayscale).
- `dataset/chars.csv` : Rows with `label,base_unicode` (and optionally other metadata columns).
- `dataset/glyphs.db` : SQLite (or similar) containing glyph metadata (label, glyph_id, splits if present).
- `data/rasterize.py` : (Optional) retained script to regenerate rasters if needed.

Removed assumptions:
- No precomputed primitive memmaps.
- No phase1/phase2 checkpoints, centroids, or prior label_map.json.
- No existing primitive vocab artifacts (we will bootstrap primitives on‑the‑fly or treat them as latent).

Implication: The unified model must derive (or optionally learn) primitive-like representations directly from rasters without relying on earlier pipeline outputs. We will generate an internal label_map dynamically from chars.csv (ordered by first occurrence of label) and store it in-memory; optional export to `artifacts/label_map.json` for checkpoint reproducibility.

## 3. Objectives & Constraints
- Accuracy: Unicode top‑1 ≥ current Phase 2 light (baseline target); primitive auxiliary ≥90% accuracy.
- Latency: Single forward pass; batch-friendly; avoid large per-glyph loops.
- Memory: Model ≤3M params; optional INT8 dynamic quantization for CPU.
- Debuggability: Ability to visualize predicted per‑cell primitive IDs overlayed on raster.
- Persistence: Training should resume quickly using a hybrid in‑memory + memmap cache for grids.

## 4. High-Level Architecture Overview
Pipeline (forward):
1. Input: `[B,1,128,128]` glyph raster (uint8 -> float32 normalized to [0,1]).
2. Stem CNN: Minimal downsampling to align with 16×16 cell grid.
3. Cell Feature Tensor: `[B,C,16,16]` (C ≈ 64).
4. Primitive Head (auxiliary): Per-cell linear classifier → primitive logits `[B,256,1024]`.
5. Primitive ID Extraction: Argmax (optionally Gumbel-soft or Straight‑Through during train).
6. Token Embedding: Primitive IDs → embedding dimension `E` (e.g. 32 or 48).
7. Positional Encoding: Absolute 2D + optional learned bias.
8. MLP Mixer Blocks (Hourglass-style or token-mixing + channel-mixing):
   - Several blocks transform `[B,256,E]`.
9. Pooling: Mean or learned CLS token → `[B,E]`.
10. Unicode Classification Head: MLP → logits `[B,num_unicode_classes]`.
11. Optional: Multi-label diacritic / subset head (future extension).

Debug mode: Skip token embedding/mixer; return primitive grid (IDs/logits).

## 5. Detailed Layer Specification
Stem CNN (targeting spatial 16×16):
- Conv(1→32, k=3, stride=2, pad=1) → 64×64
- MaxPool(2) → 32×32
- Conv(32→64, k=3, stride=2, pad=1) → 16×16
- Optional light residual block (64→64) ×1 (adds ~30K params if used)
Feature Flatten:
- Rearrange `[B,64,16,16]` → `[B,256,64]` (256 cells).
Primitive Head:
- Linear(64→1024) applied per cell → `[B,256,1024]` primitive_logits
Token Embedding:
- Embedding(1024, E) (E=32 or 48)
Positional Encoding:
- Fixed sinusoidal 2D or learned `[256,E]` non-trainable vs trainable toggle.
MLP Mixer Block (Hourglass):
- Expand: Linear(E→W) + SiLU (e.g. W=256)
- Bottleneck: Linear(W→N) + SiLU (N=64)
- Project: Linear(N→E) + LayerNorm
- Relative Position Bias: small learned embedding aggregated into token features
Repeat 2–4 blocks based on accuracy/speed tradeoff.
Head:
- Linear(E→256) + activation + Dropout
- Linear(256→1216) (unicode classes)

## 6. Parameter & FLOPs Estimates (Example E=32, W=256, N=64)
- Stem Conv/BN/Pool: ≈ (1*32*3*3) + (32*64*3*3) + BN params ≈ 20K
- Primitive Head: 256 * (64*1024 + 1024 bias) ≈ 16.8M logits computed but not all stored simultaneously (memory shaped by batch) — parameters only 64*1024 + 1024 ≈ 65,600.
- Embedding: 1024 * 32 = 32,768
- Mixer Block (per block):
  - Expand: 32*256 + 256 bias ≈ 8,448
  - Bottleneck: 256*64 + 64 bias ≈ 16,448
  - Project: 64*32 + 32 bias ≈ 2,080
  - LayerNorm: 64 params (gamma+beta per token embedding dimension? Actually LN over E=32 => 64 total)
  - Rel Pos Emb: (buckets^2 * E) small (e.g. 8*8*32 = 2,048)
  Total ≈ ~29K per block.
- Two blocks: ≈ 58K
- Head: (32*256 + 256) + (256*1216 + 1216) ≈ 8,448 + 312,832 ≈ 321,280
Total Trainable Params ≈ Stem(20K) + Primitive Head(65.6K) + Embedding(32.8K) + Mixer(58K) + Head(321K) + Residual optional (~30K) ≈ ~528K base.
NOTE: Primitive logits layer counts once; heavy compute but parameter-light. If we extend blocks or width, params can approach 2–3M (acceptable ceiling). This is leaner than 7.2M CNN while leveraging large softmax only at primitive head.

## 7. Primitive Auxiliary Supervision
- Primary Unicode loss: Cross-Entropy on glyph logits.
- Auxiliary primitive loss: Cross-Entropy on per-cell primitive logits.
- Total: `L = L_unicode + α * L_primitive` (α schedule: 1.0 first N epochs, then decay to 0.3).
- Optional temperature smoothing for primitive predictions to reduce argmax brittleness during token embedding (soft → hard).

## 8. Data Handling & Lightweight Caching (Minimal Environment)
Given only raw rasters and optional glyph metadata, we simplify (no large artifact persistence):
- Primitive supervision: optional. If unavailable, treat primitive head as self-supervised (entropy regularization + online k-means mini-batch clustering over cell features to form transient prototype IDs).
- On-the-fly cell extraction: Slice each 128×128 raster into 16×16 × 8×8 cells in the data loader (no stored grids; computed per batch).
- Lightweight in-memory cache (optional): Store recent N glyph tensors (5–10% of dataset) to mitigate disk PNG decode cost; eviction FIFO; no memmap flush logic.
- Resume: Reload model weights and rebuild caches lazily; regenerate label_map from chars.csv to ensure determinism.
- No memmap or flush intervals needed; avoid filesystem writes exceeding Kaggle limits (only checkpoints + optional label_map export).
- Optional future extension: Write a simple `grids_cache.npy` if session length permits, but not required in initial notebook.

This keeps the notebook portable and avoids large artifact management overhead.

## 9. Kaggle Notebook Structure (Simple PyTorch Lightning)
Instead of complex training scripts, use a single compact notebook with PyTorch Lightning:

```python
# Cell 1: Installs & Imports (5-10 lines)
!pip install lightning -q
import lightning as L
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from pathlib import Path
from PIL import Image

# Cell 2: Dataset (20-30 lines)
class GlyphDataset(Dataset):
    def __init__(self, rasters_dir, chars_csv):
        self.chars = pd.read_csv(chars_csv)
        self.label_to_idx = {lbl: i for i, lbl in enumerate(self.chars['label'].unique())}
        # Load raster paths and map labels
    def __getitem__(self, idx): ...
    def __len__(self): ...

# Cell 3: LightningModule (40-60 lines)
class GlyphModel(L.LightningModule):
    def __init__(self, num_classes=1216):
        super().__init__()
        self.stem = nn.Sequential(...)  # Stem CNN
        self.primitive_head = nn.Linear(64, 1024)
        self.embed = nn.Embedding(1024, 32)
        self.mixer = nn.Sequential(...)  # MLP blocks
        self.head = nn.Linear(32, num_classes)
    
    def forward(self, x, debug_mode=False): ...
    
    def training_step(self, batch, batch_idx):
        img, label = batch
        unicode_logits, prim_logits = self(img)
        loss = F.cross_entropy(unicode_logits, label)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        img, label = batch
        unicode_logits, _ = self(img)
        acc = (unicode_logits.argmax(1) == label).float().mean()
        self.log('val_acc', acc)
    
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=4e-4)

# Cell 4: Train (5-10 lines)
train_ds = GlyphDataset('dataset/rasters', 'dataset/chars.csv')
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
model = GlyphModel()
trainer = L.Trainer(max_epochs=30, accelerator='auto', precision='16-mixed')
trainer.fit(model, train_loader)
```

**Total: ~100 lines vs 1000+ in previous approach.**

## 10. Configuration (Inline or Simple YAML)
For Kaggle notebooks, avoid heavy config files. Use inline hyperparameters or a minimal dict:
```python
# Simple inline config (Kaggle style)
config = {
    'batch_size': 128,
    'epochs': 30,
    'lr': 4e-4,
    'embedding_dim': 32,
    'num_primitives': 1024,
    'num_classes': 1216,
    'primitive_alpha': 0.3,  # Weight for auxiliary loss
}

# Or minimal YAML (optional)
# config.yaml:
# batch_size: 128
# lr: 0.0004
# embedding_dim: 32
```

Lightning Trainer handles most config automatically (checkpointing, logging, precision, etc.).

## 11. Loss Functions
- Unicode CE (with optional label smoothing 0.05).
- Primitive CE (optionally focal if imbalance observed).
- KL distillation (optional) from previous Phase 2 CNN: `KL( softmax(old_logits/T) || softmax(new_logits/T) ) * λ`.
- Total: `L_total = CE_unicode + α * CE_primitive + λ * KL_distill (conditional)`.

## 12. Evaluation & Script Integration
- Adapt existing overlay script workflow:
  - Replace Phase 1 cell classifier + Phase 2 chain with single `mlp_infer.py`.
  - Debug primitive overlay: call `model(..., debug_mode=True)` -> primitive IDs.
  - Provide CLI to output JSONL similar to `infer_chain.py`.
- Metrics:
  - Unicode top‑1/top‑5.
  - Primitive per-class accuracy (sample subset to reduce overhead).
  - Diacritic subset accuracy (filter labels with `_diacritic` or known codepoint list).
- Batch evaluation on test split derived from existing splits or regeneration.

## 13. Performance & Speed Strategies
- Use AMP for primitive head large softmax.
- Reduce embedding_dim to 24 if latency still high (retest accuracy).
- JIT compile / torch.compile.
- Quantize Linear layers dynamically for CPU deployment.
- Optional caching of primitive logits (if stable after early epochs) to skip recomputing token embedding path for curriculum—likely unnecessary initially.

## 14. Distillation Plan
Stage 1: Train unified MLP from scratch with strong primitive supervision.
Stage 2: Load Phase 2 (capacity) checkpoint; run inference to obtain soft unicode logits per glyph; store them in a sidecar memmap or on-the-fly; add KL term with temperature T=3.
Stage 3: Reduce α for primitive loss; focus on unicode logits alignment.
Stage 4: Optional fine-tune with primitives turned off (α=0) for final classification polishing.

## 15. Incremental Migration Path
1. Implement minimal model (no rel pos bias, 1 mixer block).
2. Validate shapes, training loop (primitive supervision only).
3. Add unicode head.
4. Enable joint loss + logging.
5. Add positional encoding + second mixer block.
6. Integrate caching & resume mechanics.
7. Distillation integration.
8. Quantization and deployment refinements.

## 16. Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Primitive loss dominates, hurting unicode accuracy | Schedule α decay; gradient norm monitoring |
| Overfitting due to small mixer depth | Early stopping + dropout regularization |
| Memmap flush race conditions | Atomic flush (write temp then rename), periodic checksum |
| Argmax discreteness harms gradients | Optional Gumbel-soft during first epochs for smoother primitive embedding |
| Large primitive head softmax cost | Use AMP + potential hierarchical softmax (future) |

## 17. High-Level TODO List (Kaggle Notebook Approach)
**Phase 1: Minimal Working Notebook (~2 hours)**
- [ ] Cell 1: Setup (installs, imports, paths) - 10 lines
- [ ] Cell 2: Data loading (GlyphDataset, label_map from chars.csv) - 30 lines
- [ ] Cell 3: Model (LightningModule with stem + primitive head + mixer) - 50 lines
- [ ] Cell 4: Train (Trainer + fit) - 10 lines
- [ ] Cell 5: Quick inference test - 10 lines

**Phase 2: Add Features (~1-2 hours each)**
- [ ] Primitive visualization (debug_mode overlay) - add 20 lines
- [ ] Validation split & metrics - modify training_step/validation_step
- [ ] Data augmentation (torchvision transforms) - 5 lines in Dataset
- [ ] Save/load best checkpoint - Lightning default, add callback if needed

**Phase 3: Optimization (optional)**
- [ ] Mixed precision (already included with `precision='16-mixed'`)
- [ ] Learning rate scheduler (add to configure_optimizers)
- [ ] Primitive loss scheduling (add alpha decay logic)
- [ ] Export to TorchScript for deployment

**No separate scripts needed** - everything in one notebook, ~150 lines total.

## 18. Kaggle-Specific Best Practices
1. **Use Lightning Callbacks** for advanced features (not custom code):
   ```python
   from lightning.callbacks import ModelCheckpoint, EarlyStopping
   trainer = L.Trainer(
       callbacks=[
           ModelCheckpoint(monitor='val_acc', mode='max'),
           EarlyStopping(monitor='val_acc', patience=5)
       ]
   )
   ```

2. **Kaggle Paths** (auto-detected):
   - Data: `/kaggle/input/your-dataset/`
   - Output: `/kaggle/working/` (auto-saved)
   - Logs: Automatically available via TensorBoard in notebook

3. **GPU Utilization**:
   ```python
   trainer = L.Trainer(accelerator='auto')  # Automatically uses GPU if available
   ```

4. **Quick Debugging**:
   ```python
   trainer = L.Trainer(fast_dev_run=True)  # Run 1 batch to test
   trainer = L.Trainer(overfit_batches=10)  # Overfit check
   ```

5. **Avoid Heavy Dependencies**: Lightning includes everything needed (no separate config parsers, custom loggers, manual checkpoint management).

## 19. Example Kaggle Notebook Outline
```
Notebook: glyph-recognition-unified-mlp.ipynb

[Cell 1] Setup & Installs
[Cell 2] Load Data & Create Label Map
[Cell 3] Define GlyphDataset
[Cell 4] Define GlyphModel (LightningModule)
[Cell 5] Train
[Cell 6] Visualize Primitives (Debug Mode)
[Cell 7] Inference & Submission

Total: 7 cells, ~150 lines
```

---

End of `mlp-plan.md`
