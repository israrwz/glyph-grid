# Hierarchical Glyph Recognition Project Plan
Independent Architecture & Implementation Specification

## 1. Purpose & Scope
Build a high‑accuracy, font‑invariant recognizer for ~1.3K Arabic (and related) glyph classes (isolated forms, contextual variants, diacritics, ligatures). Emphasis:
- Robust generalization across thousands of fonts.
- Interpretability via a two‑phase “primitive grid” pipeline.
- Extensibility to retrieval / similarity and downstream layout analysis.

This document is self‑contained: it defines data assumptions, rasterization policy, model phases, evaluation, and extension roadmap with no dependency on any prior codebase.

---

## 2. Core Design Decisions (Summary)
| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Input Representation | 128×128 single‑channel raster, binary foreground (1) on background (0) | Compact, sufficient for stroke topology |
| Alignment | Top-left anchored (not centered) | Stable spatial prior; cells always index from same origin |
| Scaling Policy | Tight fit for main glyphs; diacritics upscaled to 50% canvas | Preserve detail while normalizing size variance; restore salience of tiny marks |
| Diacritic Detection | Size heuristic: major dimension < 25% of EM vertical span | Simple, deterministic |
| Hole Preservation | Orientation-based winding fill | Keeps counters (holes) with minimal complexity |
| Tokenization | 16×16 grid of 8×8 “cells” | Balance between local detail and manageable sequence length |
| Phase 1 Output | Hard 1024 primitive IDs (categorical) | Interpretability & efficiency; can switch to embeddings later |
| Phase 2 Input | Sequence / map of primitive IDs + positional encoding | Discrete abstraction improves invariance |
| Architecture Style | Two-phase (cell classifier → glyph classifier) | Decomposes complexity; explicit error accounting |
| Filtering | Exclude labels with <5 samples; exclude shaped variants if undesired | Reduce noise and unstable classes |

---

## 3. Data Model & Storage

### 3.0. Project Directory Conventions
To keep the new project self‑contained and reproducible, all inputs and generated artifacts follow a strict location policy:

- Input Database (read‑only): `dataset/glyphs.db`
  The SQLite file containing `fonts` and `glyphs` tables (schema defined below). No process mutates this file; any schema migrations produce a new versioned copy (e.g. `dataset/glyphs_v2.db`).

- Generated Outputs (write targets): `outputs/`
  - `outputs/rasters/` : Per‑glyph raster binaries (e.g. `glyph_<id>.u8` or batched `.npz`)
  - `outputs/grids/`   : Primitive ID grids (e.g. `grid_<id>.u16`)
  - `outputs/meta/`    : Metadata JSONL (`meta.jsonl`), label maps, font split manifests
  - `outputs/primitives/` : Clustering centroids (`primitive_centroids.npy`), prototype thumbnails
  - `outputs/checkpoints/phase1/` : Phase 1 model weights
  - `outputs/checkpoints/phase2/` : Phase 2 model weights
  - `outputs/logs/` : Training logs, scalar metrics, confusion matrices
  - `outputs/analysis/` : Error reports, attention maps, ablation summaries

Policy:
1. Nothing writes back into `dataset/`.
2. Any destructive or large intermediate cache (e.g., LMDB) is placed under `outputs/cache/` with clear size estimates.
3. Reproducibility metadata (git commit, config hashes, centroid hash) appended as a JSON line in `outputs/meta/run_history.jsonl`.
4. Temporary working files (e.g., partial clustering shards) go to `outputs/tmp/` and are safe to delete after successful pipeline completion.

These conventions ensure clean separation of source data and derived artifacts, enabling easy cleanup (`rm -rf outputs/*`) without risking original data loss.

### 3.1. Glyph Database Schema (Proposed)
SQLite (or equivalent relational store). The fields below are exhaustive for what the pipeline expects; any additional columns should be additive and ignored by default.

```
TABLE fonts (
  file_hash      TEXT PRIMARY KEY,   -- Stable hash / identifier of the font file
  upem           INTEGER,            -- Units per EM (scaling base)
  ascent         INTEGER,            -- Font ascent (font units)
  descent        INTEGER,            -- Font descent (font units; typically negative or 0-based offset)
  typo_ascent    INTEGER,            -- Typographic ascent (preferred over ascent when present)
  typo_descent   INTEGER,            -- Typographic descent
  x_height       INTEGER,            -- x-height if available (0 if absent)
  cap_height     INTEGER,            -- Cap height if available
  line_gap       INTEGER             -- Additional line gap
  -- (Optional future fields: family_name TEXT, style TEXT, weight INT)
);

TABLE glyphs (
  glyph_id       INTEGER PRIMARY KEY,  -- Internal glyph surrogate key
  -- Some legacy scripts may reference 'id' instead of 'glyph_id'; treat them interchangeably if both exist.
  f_id           TEXT NOT NULL,        -- Foreign key to fonts.file_hash
  label          TEXT NOT NULL,        -- Canonical classification label (post-filter)
  contours       TEXT NOT NULL,        -- JSON array encoding vector outline commands
  joining_group  TEXT,                 -- Script joining group (Arabic shaping attribute)
  char_class     TEXT,                 -- High-level category (latin, diacritic, isol, init, medi, final, punctuation, etc.)
  -- Optional / tolerated (if present):
  width          INTEGER,              -- Advance width (not required; ignored if absent)
  lsb            INTEGER,              -- Left side bearing (ignored if absent)
  rsb            INTEGER,              -- Right side bearing (ignored if absent)
  alt_label      TEXT,                 -- Alternate or raw source label (not used in classification)
  FOREIGN KEY(f_id) REFERENCES fonts(file_hash)
);
```

**Derived / Query-Time Conditions (Not Physical Columns)**
- `length(contours) > 0` is used in queries to exclude empty contour strings.
- Diacritic detection uses computed bounding box vs. EM height (not stored; recomputed).
- Label frequency (for min-count filtering) is computed via a `GROUP BY label` aggregation; frequency itself is not persisted unless an auxiliary summary table is introduced.

**Field Usage Summary**
| Table  | Field         | Required | Used For                                      |
|--------|--------------|----------|-----------------------------------------------|
| fonts  | file_hash     | Yes      | Join key                                      |
| fonts  | upem          | Preferred| Scale normalization, diacritic size heuristic |
| fonts  | ascent/descent| Fallback | EM height if typo_* missing                   |
| fonts  | typo_ascent/descent | Preferred | EM height (primary)                  |
| fonts  | x_height      | Optional | Future size-aware features                    |
| fonts  | cap_height    | Optional | Future size-aware features                    |
| fonts  | line_gap      | Optional | Documentation / potential metrics             |
| glyphs | glyph_id      | Yes      | Primary glyph reference                       |
| glyphs | f_id          | Yes      | Font linkage                                  |
| glyphs | label         | Yes      | Target class (after filtering)                |
| glyphs | contours      | Yes      | Vector outline source                         |
| glyphs | joining_group | Optional | Grouping / auxiliary supervision              |
| glyphs | char_class    | Optional | Filtering, analysis (exclude shaped, etc.)    |
| glyphs | width/lsb/rsb | Optional | Reserved for future kerning / spacing tasks   |
| glyphs | alt_label     | Optional | Provenance / auditing                         |

**Integrity / Validation Checks**
1. Ensure every glyph row has a matching font row (`INNER JOIN` cardinality test).
2. Reject glyphs where `contours` parses to zero closed paths.
3. After filtering by label frequency (≥5), re-map `label -> class_index` deterministically (sorted lexicographically).
4. Log orphan font hashes or glyph rows failing parsing for audit.

**Indexing Recommendations**
```
CREATE INDEX IF NOT EXISTS idx_glyphs_fid ON glyphs(f_id);
CREATE INDEX IF NOT EXISTS idx_glyphs_label ON glyphs(label);
CREATE INDEX IF NOT EXISTS idx_glyphs_joining_group ON glyphs(joining_group);
```
(Helps stratified sampling and class frequency scans.)

**Rationale for Including Optional Metrics**
Even if not immediately consumed, ascent/descent/x_height/cap_height provide hooks for future relative-size modeling, composite glyph detection, or diacritic anchoring heuristics without schema migration.


### 3.2. Vector Contours JSON
Example canonical command format (per subpath):
```
[
  ["moveTo", [x,y]],
  ["lineTo", [x,y]],
  ["qCurveTo", [[ctrl_or_mid?], [x,y]]],  // Quadratic with midpoint interpolation
  ["cubicTo", [[c1x,c1y],[c2x,c2y],[x,y]]],
  ["closePath", null]
]
```
Multiple subpaths → list of lists.

### 3.3. Label Filtering
1. Aggregate frequency per distinct `label`.
2. Keep labels where count ≥ 5 (threshold configurable in preprocessing script).
3. Optionally drop “shaped” variants (e.g., suffix `_init`, `_medi`, `_final`) if target taxonomy wants only abstract forms.
4. Produce mapping: `label -> class_index` in stable sorted order.

---

## 4. Rasterization Specification

### 4.1. Coordinate & EM Metrics
- EM vertical span = `used_ascent - used_descent`. Prefer `typo_ascent/typo_descent`; fallback to `ascent/descent`.
- Parse contours into polylines; sample cubic Bézier curves uniformly (e.g., 8 subdivisions per segment).

### 4.2. Bounding Box & Diacritic Heuristic
```
bbox_w = max_x - min_x
bbox_h = max_y - min_y
major_dim = max(bbox_w, bbox_h)
em_height = (used_ascent - used_descent)
ratio = major_dim / em_height
is_diacritic = (ratio < 0.25)
```

### 4.3. Scaling & Placement
| Case | Target Largest Dimension | Placement |
|------|--------------------------|-----------|
| Main glyph | 128 px | Top-left (post-scale shift min to 0,0) |
| Diacritic | 64 px (50% canvas side) | Top-left (no center) |

Scale factor:
```
scale = target_largest_dimension / major_dim
```
Apply scale to all points. Then translate so new min_x = 0, min_y = 0.

### 4.4. Rendering Algorithm
1. Supersample canvas: `render_size = 2 * 128 = 256`.
2. Draw filled polygons:
   - Determine dominant orientation by summing absolute signed areas with sign to decide fill vs. hole.
   - Fill solids (255), carve holes (0).
3. Optional stroke (not needed initially).
4. Downsample to 128×128 with bicubic, threshold at 0.5 → binary mask.
5. Store as uint8 (0 or 255), or float32 in [0,1].

### 4.5. Metadata Captured
```
glyph_id
label
font_hash
is_diacritic (bool)
original_bbox = [min_x, min_y, max_x, max_y] (font units)
scale_factor
major_dim_ratio = ratio
target_dim (64 or 128)
```

---

## 5. Cell Grid Extraction
- Partition raster into 16×16 grid (each cell 8×8).
- Maintain deterministic ordering (row-major).
- Cells with all zeros → primitive ID 0 (EMPTY).
- For diacritic glyphs, trailing empty region remains zeros (no special cropping).

---

## 6. Phase 1: Primitive Cell Classification

### 6.1. Primitive Vocabulary Initialization
- Sample ~1M non-empty cells (random across fonts).
- Flatten each 8×8 mask → 64D vector.
- K-Means (k=1023) ignoring empty; assign cluster IDs 1..1023.
- Class 0 reserved for empty cell.
- Save prototype centroids for visualization.

### 6.2. Model Architecture (Baseline CNN)
```
Input: (B,1,8,8)
Conv(32,3,pad=1) + ReLU + BN
MaxPool(2) -> (B,32,4,4)
Conv(64,3,pad=1) + ReLU + BN
MaxPool(2) -> (B,64,2,2)
Flatten -> 256
FC 128 + ReLU + Dropout(0.2)
FC 1024 logits
Softmax/CrossEntropy
```
Parameters: ~50–60K.

### 6.3. Training Protocol
| Setting | Value |
|---------|-------|
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | Cosine or OneCycle |
| Batch Size | 1024 |
| Epochs | 50–80 (early stop on val acc plateau) |
| Loss | CrossEntropy (optional class weights) |
| Augmentations | On full raster before slicing (translation ≤2px, blur σ≤0.6, mild gamma); per-cell augmentation withheld to preserve mutual consistency |

### 6.4. Metrics
- Accuracy overall & by primitive frequency bucket.
- Confusion matrix (especially near-diagonals).
- Prototype visualization (average cell per cluster).

### 6.5. Optional Future Variant
Switch from hard IDs to 64D learned embedding with contrastive loss + product quantization fallback.

---

## 7. Phase 2: Glyph Classification (Primitive Grid → Label)

### 7.1. Input Representation
- Grid indices: shape (16,16) of ints (0..1023).
- Embed each primitive ID: `Embedding(1024, 64)` → (16,16,64).
- Add 2D positional encoding (sinusoidal or learnable).

### 7.2. Two Backbone Options

#### A. Lightweight Transformer
- Flatten to 256 tokens (optional patch grouping: 4×4 blocks -> 64 tokens of dim 256).
- Encoder: 4–6 layers, d_model=256, heads=8, MLP=512, dropout=0.1.
- Global average or CLS token → 256D → FC (num_classes).

#### B. CNN Over Embedding Maps
- Reshape to channels-first (64,16,16).
- Apply a small residual CNN (e.g., 6 residual blocks with width 128).
- Global average pool → FC.

Start with Transformer variant for sequence interpretability.

### 7.3. Loss & Training
| Component | Description |
|-----------|-------------|
| Primary Loss | CrossEntropy on glyph label |
| Aux (Optional) | Binary flags for presence of diacritics subtype categories (weight 0.1) |
| Optimizer | AdamW (lr=5e-4) |
| Batch Size | 128 (glyph-level) |
| Epochs | 30–50 |
| Scheduler | ReduceLROnPlateau (monitor val acc) or Cosine |
| Regularization | Dropout 0.1, label smoothing 0.05 |

Use noisy targets during training by feeding predicted Phase 1 grids (not ground-truth prototypes) after Phase 1 converges to simulate deployment noise.

### 7.4. Joint Fine-Tuning (Optional)
After Phase 2 stabilizes:
1. Unfreeze Phase 1 final FC layer (or entire Phase 1) with smaller lr (1e-4).
2. Train combined loss for ~5 epochs for holistic alignment.

---

## 8. Evaluation

### 8.1. Splits
- Fonts partitioned: 80% train, 10% validation, 10% test (disjoint font sets).
- Record list of font hashes per split (YAML/JSON).

### 8.2. Metrics
| Level | Metrics |
|-------|---------|
| Phase 1 | Primitive accuracy, per-frequency bucket accuracy, top confusions |
| Phase 2 | Top-1 / Top-5 glyph accuracy, macro & weighted F1 |
| End-to-End | Accuracy vs. direct single-stage baseline, per-font variance |
| Robustness | Accuracy on diacritics subset, ligatures subset |
| Embedding (if added) | Retrieval top-k, intra/inter similarity diff, effect size |

### 8.3. Ablations
1. Grid resolution: 8×8 cells (coarser) vs 32×32 (finer).
2. Primitive vocabulary size: 512, 1024, 2048.
3. Diacritic retention method: current scaling vs center anchoring (control).
4. Hole fill strategy (orientation vs even-odd) – optional verification.
5. One-stage baseline: Direct CNN (e.g., MobileNetV3) on 128×128.

### 8.4. Error Analysis
- Heatmap overlay: Map Phase 2 attention weights back onto cells.
- Cluster misclassified glyph embeddings to identify systematic confusions.
- Compare predicted vs. prototype cell distributions to see if failure due to Phase 1 noise or global composition.

---

## 9. Data Pipeline Implementation Outline

### 9.1. Preprocessing Script Tasks
1. Load fonts & glyphs table.
2. Aggregate label counts; build whitelist.
3. Filter glyph rows (contours not null, whitelist).
4. Rasterize each glyph with unified policy (store tensor + metadata).
5. Slice cells; store cell tensors in a chunked store (LMDB or memory-mapped arrays).
6. Run primitive K-Means; assign IDs; rewrite grid indices.
7. Persist:
   - `rasters/<glyph_id>.u8` (optional)
   - `grids/<glyph_id>.u16` (16×16)
   - `meta.jsonl`
   - `primitive_centroids.npy`
   - `label_map.json`

### 9.2. Storage Recommendations
| Resource | Format | Justification |
|----------|--------|---------------|
| Rasters | Compressed NPZ / LMDB | Fast random access |
| Grids | NPZ / LMDB | Small fixed-size records |
| Metadata | JSONL | Stream-friendly |
| Primitives | NPY | Direct load to torch/tf tensor |

### 9.3. Determinism
- Fixed random seed for K-Means initialization.
- Save permutation of fonts used for splitting.
- Hash of primitive centroids for reproducibility tags.

---

## 10. Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|-----------|
| Label noise (rare variants mislabeled) | Caps accuracy | Minimum frequency filter; manual inspection of low-confidence classes |
| Over-abundance of empty cells | Class imbalance hurting Phase 1 | Class weights or downsample empties |
| Diacritic misclassification due to top-left anchoring | Reduced discriminative context | Consider adding relative position embedding of non-empty region bounding box |
| Primitive vocabulary under-expressive | High Phase 1 error | Increase k (2048) or switch to learned embeddings |
| Tight scaling removing relative size cues | Hard to differentiate some classes | Diacritic heuristic preserves part of size signal; future: hybrid size channel |
| Transformer overfitting | Poor generalization | Dropout, early stop, alternative CNN path |
| Performance constraints (IO bottleneck) | Slow training | LMDB + prefetch, pin memory, larger batches |

---

## 11. Extension Roadmap
| Phase | Feature | Description |
|-------|---------|-------------|
| 1 | Baseline pipeline | Implement rasterizer, Phase 1 hard IDs, Phase 2 transformer |
| 2 | Contrastive Phase 1 | Replace hard IDs with learnable embeddings + clustering fallback |
| 3 | Multi-channel raster | Add distance transform & skeleton channels |
| 4 | Retrieval API | Nearest-neighbor glyph similarity service |
| 5 | Compound sequence modeling | From glyphs to word shapes (ligature detection) |
| 6 | Active learning loop | Flag high-entropy glyphs for relabeling |
| 7 | Quantization / Distillation | Optimize for edge deployment |

---

## 12. Pseudocode Excerpts

### 12.1. Rasterization (Simplified)
```python
def rasterize(glyph, font_metrics):
    polys = parse_contours(glyph.contours)
    min_x,min_y,max_x,max_y = bbox(polys)
    major = max(max_x-min_x, max_y-min_y)
    em_h = (font_metrics.ascent - font_metrics.descent) or (max_y - min_y)
    ratio = major / em_h
    is_diac = ratio < 0.25
    target = 64 if is_diac else 128
    scale = target / major if major > 0 else 1.0
    polys_scaled = [ [( (x-min_x)*scale, (y-min_y)*scale ) for (x,y) in p] for p in polys ]
    canvas = render_orientation_fill(polys_scaled, supersample=2, size=128)
    canvas = downsample_and_binarize(canvas)
    return canvas, { "scale_factor": scale, "is_diacritic": is_diac, "ratio": ratio }
```

### 12.2. Primitive Assignment
```python
# After clustering centroids C (1023 x 64)
def assign_primitive(cell):
    if cell.sum() == 0: return 0
    v = cell.flatten().astype(np.float32)
    idx = np.argmin(((C - v)**2).sum(axis=1))
    return idx + 1
```

---

## 13. Baseline Performance Targets (Aspirational)
| Component | Metric | Target |
|-----------|--------|--------|
| Phase 1 Primitives | Accuracy | ≥ 90% |
| Phase 1 Empty Class | False Positive Rate | < 2% |
| Phase 2 Top-1 | Accuracy | ≥ 55% (baseline) |
| Phase 2 Top-1 (with enhancements) | Accuracy | ≥ 70% |
| End-to-End Retrieval (top10) | Mean Accuracy | ≥ 60% (after embedding upgrade) |

*Note:* These targets acknowledge complexity of ~1.3K glyph classes; achieving >90% may require multi-channel or contextual modeling.

---

## 14. Engineering Blueprint

### 14.1. Proposed Directory Structure
```
data/
  prepare.py
  rasterize.py
  primitives.py
  split.py
models/
  phase1_cnn.py
  phase2_transformer.py
  losses.py
train/
  train_phase1.py
  train_phase2.py
  joint_tune.py
eval/
  eval_phase1.py
  eval_phase2.py
  analyze_errors.py
assets/
  prototypes/
  centroids/
docs/
  NEW_PLAN.md
configs/
  rasterizer.yaml
  phase1.yaml
  phase2.yaml
```

### 14.2. Config Example (phase1.yaml)
```yaml
seed: 42
primitive_count: 1024
batch_size: 1024
lr: 0.001
weight_decay: 1e-4
epochs: 60
early_stop_patience: 8
augment:
  translate_px: 2
  blur_sigma_max: 0.6
  gamma_jitter: 0.05
```

---

## 15. Validation & QA Checklist
| Step | Check | Tool |
|------|-------|------|
| Rasterization | Holes preserved (e.g., “م”, “ن” inner counters) | Visual grid |
| Diacritic Scaling | All diacritics ≈ 64px major dim | Scripted assertion |
| Primitive Distribution | Class histogram not hyper-skewed | Histogram plot |
| Phase 1 Overfit Probe | Train on 5K cells → 99% acc quickly | Sanity run |
| Phase 2 Noise Robustness | Train using predicted vs true primitives | Ablation |
| Split Integrity | No font hash overlap | Hash set diff |

---

## 16. Open Questions (To Revisit)
| Question | Notes |
|----------|-------|
| Are size cues beyond diacritic heuristic beneficial? | Consider adding relative size scalar channel later |
| Multi-diacritic stacking behavior? | Might require composite glyph decomposition if future tasks need sequence modeling |
| Should we compress primitives via product quantization? | Evaluate once memory > threshold |
| Would a direct MobileNet baseline surpass hierarchical quickly? | Maintain baseline for comparison |

---

## 17. Initial Execution Timeline (Indicative)
| Week | Milestone |
|------|-----------|
| 1 | Implement rasterizer + diacritic heuristic + dataset export |
| 2 | Primitive clustering + Phase 1 training (baseline) |
| 3 | Phase 2 transformer prototype + baseline evaluation |
| 4 | Error analysis + improvement iteration (embedding variant, margin losses) |
| 5 | Multi-channel & advanced augmentation experiments |
| 6 | Documentation, packaging, optional retrieval service prototype |

---

## 18. Summary
This plan establishes a structured, interpretable, and extensible framework for large-vocabulary glyph recognition using a tight + diacritic-aware raster pipeline and a two-phase modeling approach. It encodes lessons learned from earlier explorations: avoid excessive mode complexity, preserve essential size cues for diacritics, and provide clear hooks for analysis and future embedding enhancements. During project implementation and feature expansion, instead of creating command line arguments, create config entries in the yaml file with default values.

## 19. Keras Notebook Workflow & Environment Strategy

### 19.1. Goals for Notebook Usage
Notebook sessions (e.g., Kaggle, Colab, local Jupyter) will be used for:
- Rapid prototyping of Phase 1 (primitive classifier) and Phase 2 (glyph classifier).
- Visualization of rasterization, diacritic scaling correctness, and primitive prototype quality.
- Sanity checks on small subsets before long GPU runs.

Core training (full dataset) should still migrate to scripted runs for determinism, but notebooks accelerate iteration on:
- Raster policy tweaks (diacritic enlargement ratio).
- Primitive vocabulary size adjustments (e.g., 512 vs 1024).
- Margin / embedding weight schedules.

### 19.2. Directory & Data Conventions in Notebooks
- Mount / copy the read‑only database at `dataset/glyphs.db`.
- All notebook-generated artifacts (even transient) go under `outputs/`:
  - `outputs/notebooks/phase1_preview/` for sampled rasters & cell grids.
  - `outputs/notebooks/plots/` for distribution charts.
  - `outputs/notebooks/checkpoints/` for quick, disposable weights (distinct from main `outputs/checkpoints/` to avoid confusion).
- Never write into `dataset/`.

### 19.3. Phase 1 (Primitives) – Keras Notebook Flow
1. Load a *small* glyph subset (e.g., 8–16 fonts, or 20k glyphs) for CPU sanity on a laptop.
2. Rasterize on-the-fly (or load pre-raster cache if previously generated).
3. Extract 16×16 grid → flatten cells → filter out empties for clustering sample.
4. Run mini K-Means (e.g., k=128) just to verify pipeline speed & cluster “look”.
5. Train tiny CNN (k=128 classes) for 5–10 epochs; verify:
   - Rapid overfit on small sample (accuracy >95%) → confirms model capacity.
   - Empty cell FPR is low (<2–3%).
6. Scale up:
   - Switch k to 1024 using full sampled clustering (offline or new cell sample).
   - Train with larger batches on GPU (Colab/Kaggle) watching primitive accuracy curve.

Gotchas:
- Excessive ORDER BY RANDOM() in SQLite for large limits is slow; precompute a shuffled id list or sample IDs once, then IN (...) batches.
- Class imbalance: empty cells dominate; mitigate via class weighting or sample cap.

### 19.4. Phase 2 (Glyph Classification) – Keras Notebook Flow
1. Load saved primitive grids (or generate online by running Phase 1 inference).
2. Build embedding layer: `Embedding(1024, 64)` + positional encoding.
3. Prototype both:
   - A lightweight Transformer encoder.
   - A simple residual CNN over (64,16,16) embedding map.
4. Train baseline CE only (no margin) first:
   - Confirm val accuracy rises quickly above random (≈ 0.08% for 1.3K classes).
5. Introduce ArcFace / CosFace margin with **warmup**:
   - Start margin=0 for first few epochs; linearly ramp to target (e.g., 0.15) over epochs 5–10.
   - Avoid using raw linear logits for validation loss when margin is active—either:
     a) Reconstruct normalized + scaled cosine logits in validation, or
     b) Accept that val_loss is inflated and rely on val_acc (documented caveat).
6. Inject primitive noise simulation:
   - Randomly replace 2–5% of grid tokens with random non-empty primitives to model Phase 1 error bound.
7. Add diacritic-focused slice evaluation: filter glyphs flagged `is_diacritic` or those whose grid has ≤ N non-empty cells.

Gotchas:
- Validation loss mismatch: If margin path differs between train/eval, CE can spike unrealistically while accuracy is correct; *trust accuracy* or harmonize evaluation logits.
- Large batch smoothing: Using very large batches can flatten early accuracy growth—smaller batch (e.g., 512–768) may improve convergence dynamics.
- Primitive noise: Training purely on ground-truth primitives inflates unrealistic performance; always add some predicted or synthetic noise once Phase 1 stabilizes.
- Over-anchoring diacritics: Top-left anchoring is deliberate; do not random-shift tiny diacritics during augmentation (can destroy learned positional prior).

### 19.5. CPU vs GPU Workflow
| Mode | Use Case | Settings |
|------|----------|----------|
| Local Laptop (CPU) | Sanity: pipeline integrity, diacritic scaling, clustering small k | Subset fonts (≤10), k=128, batch_size ≤256 |
| Cloud GPU (Notebook) | Full primitive training & Phase 2 baseline | k=1024, batch_size 768–1024, mixed precision |
| Scripted GPU Runs | Long training, ablations, reproducibility | Log deterministic config & hash centroids |

Performance tips:
- Cache preprocessed rasters locally in NPZ or LMDB to avoid repeated contour parsing.
- Use mixed precision (if TensorFlow: `tf.keras.mixed_precision.set_global_policy('mixed_float16')`) for Phase 2 speedup.
- Avoid eager re-clustering; persist centroids and reuse unless raster policy changed.

### 19.6. Reproducibility in Notebooks
- Log: random seed, k-means centroid SHA1, font split manifest, primitive count, margin schedule.
- Save a `run_manifest.json` per notebook execution into `outputs/notebooks/`.
- For partial reruns, assert centroid hash matches previously stored hash to prevent silent drift.

### 19.7. Common Pitfalls & Mitigations (Collected From Prior Exploration)
| Pitfall | Symptom | Mitigation |
|---------|---------|-----------|
| Ignoring margin mismatch | Huge val CE loss, stable accuracy | Recompute cosine logits at eval or ignore val_loss |
| Over-scaling diacritics | Uniform blobs lose distinctive traits | Cap target dimension at exactly 64 px; no further dilation |
| Excess empty cells | Primitive classifier biases | Downweight empty or cap per-batch empty ratio |
| Single augmentation path | Overfitting subtle stroke micro-noise | Add mild blur + gamma + token noise simulation |
| Late margin + high embed weight | Plateau at low accuracy | Warm margin + moderate embed weight ramp (decoupled) |

### 19.8. Minimal Keras Phase 1 Sketch (Illustrative)
```python
inputs = tf.keras.Input((8,8,1))
x = tf.keras.layers.Conv2D(32,3,padding='same',activation='relu')(inputs)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv2D(64,3,padding='same',activation='relu')(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128,activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1024,activation='softmax')(x)
phase1_model = tf.keras.Model(inputs, outputs)
phase1_model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
```

### 19.9. Minimal Keras Phase 2 Skeleton (Using Primitives)
```python
primitive_vocab = 1024
embed_dim = 64
seq_h = seq_w = 16

ids_in = tf.keras.Input((seq_h, seq_w), dtype='int32')
x = tf.keras.layers.Embedding(primitive_vocab, embed_dim)(ids_in)
# Positional encoding (simple learnable)
pos = tf.keras.layers.Embedding(seq_h*seq_w, embed_dim)(
    tf.range(seq_h*seq_w)[None,:])
pos = tf.reshape(pos, (1, seq_h, seq_w, embed_dim))
x = x + pos
# Flatten to sequence
x = tf.reshape(x, (-1, seq_h*seq_w, embed_dim))
# Simple transformer block (1-2 layers for notebook demo)
attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=embed_dim)(x, x)
x = tf.keras.layers.LayerNormalization()(x + attn)
ff = tf.keras.layers.Dense(embed_dim*4, activation='relu')(x)
ff = tf.keras.layers.Dense(embed_dim)(ff)
x = tf.keras.layers.LayerNormalization()(x + ff)
x = tf.reduce_mean(x, axis=1)
logits = tf.keras.layers.Dense(NUM_CLASSES)(x)
phase2_model = tf.keras.Model(ids_in, logits)
phase2_model.compile(optimizer=tf.keras.optimizers.AdamW(5e-4),
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])
```

### 19.10. Transition From Notebook to Production
1. Export final primitive centroids + label map → `outputs/primitives/`.
2. Export Phase 1 weights; freeze architecture contract (input size, class count).
3. Regenerate Phase 2 training dataset using Phase 1 inference (ensures alignment).
4. Migrate all hyperparameters to config YAML; record git commit + environment (Python & framework versions).

---

End of specification.
