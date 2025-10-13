# Hierarchical Glyph Recognition Project - Implementation Progress

_Last updated: <!-- timestamp placeholder -->_

## 1. Scope Recap (from NEW_PLAN.md)
Phased pipeline:
1. Rasterize vector glyphs (uniform scaling, diacritic heuristic) → 128×128 binary.
2. Extract 16×16 grid (8×8 cells) → primitive cell dataset.
3. Phase 1: K-Means over ~1M non-empty cells → 1023 primitive clusters (+ empty).
4. Phase 1 CNN primitive classifier (8×8 -> 1024 classes).
5. Phase 2 transformer over primitive ID grid -> glyph label.
6. Evaluation, error analysis, ablations.

## 2. Completed / Implemented

### 2.1 Repository Structure & Configs
- Created canonical directory layout (`data/`, `models/`, `train/`, `configs/`, `docs/`, `assets/`).
- Added `rasterizer.yaml`, `phase1.yaml`, `phase2.yaml` aligned tightly with plan (later refined).
- Copied original plan to `docs/NEW_PLAN.md`.

### 2.2 Rasterization Pipeline (data/rasterize.py)
- Font DB ingestion (SQLite `glyphs` + `fonts` join).
- Diacritic scaling rule (ratio < 0.25 => target largest dimension 64 else 128).
- Supersampling (2×) and downsampling to 128×128 with thresholding.
- Contour parsing:
  - CFF/command style: `moveTo`, `lineTo`, `qCurveTo`, `cubicTo/curveTo`, `closePath`.
  - Multi-point quadratic handling (TrueType-style qCurve implicit on-curve insertion).
  - Orientation & hole strategies implemented: `orientation`, `even-odd`, `winding`.
  - Hybrid fallback: for single-contour winding under-fill → parity fallback.
- Metadata produced (`metadata.jsonl`) with per-glyph scaling + classification fields.
- Skips empty / degenerate glyphs (no subpaths after parsing).
- Ability to limit glyphs (`--limit-glyphs`) for debug runs.

### 2.3 Fill Rule Enhancements
- Added debug infrastructure (optional) to dump masks.
- Integrated improved polygon classification logic with parity / winding comparison.
- Diagnosed hole rendering behavior; validated that certain “missing holes” are absent in original font outlines (not raster bug).
- Implemented hybrid winding fallback for self-overlap under-fill scenarios.

### 2.4 Smoke & Debug Tooling
- `data/smoke_test.py`: shape, binary value, diacritic heuristic invariants.
- Ephemeral debug scripts (executed externally) to:
  - Inspect glyph outlines directly via fontTools (CFF charstrings).
  - Compare parity vs winding pixel counts for suspect glyph IDs.
- Verified improved handling for previously problematic glyph sets.

### 2.5 Primitive Vocabulary Scaffolding
- `data/primitives.py`: sampling, MiniBatchKMeans (k=1023), centroid save (index 0 reserved for empty).
- Assignment logic: nearest centroid → primitive IDs (1..1023).
- Consolidation script `data/consolidate_cells.py` for large-scale cell corpus + optional reservoir sample.

### 2.6 Configuration Evolution
- Raster config trimmed to plan-only parameters (later extended with debug flags & fill rule).
- Phase 1 & Phase 2 configs aligned with plan’s enumerated hyperparameters (embedding dims, transformer depth, etc.).

### 2.7 Proven Execution
- Rasterized first 1K glyph subset multiple times with different fill rules.
- Verified diacritic detection, metadata integrity, and absence of false “empty” glyphs after parser fixes.
- Confirmed correct counter rendering in cleaned font set (after removal of problematic font).

## 3. In Progress / Partially Done
- Phase 1 model code (CNN) not yet implemented in `models/phase1_cnn.py`.
- Training loops (`train/train_phase1.py`, `train/train_phase2.py`) pending.
- Joint fine-tuning pipeline placeholder only.
- No automated label frequency bucketing metrics yet (planned for evaluation).
- No cell-level LMDB / memory-mapped optimization (current NumPy path only).
- Advanced error analysis scripts not implemented (attention heatmaps etc. stubbed conceptually).
- No adaptive curve flattening (uniform subdivision = 8); may affect fine curvature fidelity.

## 4. Known Technical Gaps / Issues
| Area | Current State | Impact | Planned Remedy |
|------|---------------|--------|----------------|
| Outline fidelity | Uniform 8-step subdivision | Minor curvature artifacts for tight curves | Adaptive flatness or FreeType raster |
| Hole semantics | Hybrid winding+parity fallback | Rare incorrect inclusions in complex strokes | Switch to authoritative raster (FreeType) |
| Self-overlap stroke artifacts | Heuristic fallback | Potential misclassification of interior void | Use native scan-conversion (non-zero winding) |
| Performance scaling | Single-process Python | Longer runtime for full corpus | Parallel chunking or C-backed raster |
| Primitive sampling | Reservoir + full read loop | I/O overhead on large sets | Cell shards + memory map / LMDB |
| Diacritic heuristic | Ratio-based only | Edge cases for near-threshold shapes | Add margin band / optional metadata channel |
| Metadata QA | Basic fields only | Harder to audit outline anomalies | Add contour_count, hole_count from DB & effective fill rule |
| Phase 1 dataset | Not materialized | Blocks primitive CNN training | Implement dataset + dataloader soon |

## 5. Decision Log (Highlights)
- Adopt binary 0/255 storage for rasters (plan alignment, compression friendly).
- Selected winding as the baseline fill rule due to closer match with actual font engines.
- Added parity fallback only for single-contour severe under-fill (temporary until engine-based raster present).
- Deferred complex curve error metrics until after baseline model accuracy established.

## 6. Next Immediate TODOs

### 6.1 Rasterization & Rendering
1. (B) Integrate FreeType (freetype-py) for authoritative non-zero winding raster:
   - Load each glyph outline via FreeType (CFF or glyf seamlessly).
   - Render to high-res bitmap (e.g., 4× oversample), downsample with area averaging.
   - Replace current Python polygon fill path behind a feature flag (`engine: freetype|python`).
   - Record fallback metrics: compare pixel difference vs current implementation for a sample.
2. Add geometric nesting normalization (if Python engine retained as fallback).
3. Implement adaptive subdivision (only if FreeType integration is delayed).

### 6.2 Primitive Pipeline
4. Run full corpus rasterization (post FreeType) & consolidate all cells.
5. Execute K-Means (k=1023) on ~1M sampled non-empty cells; store centroids, population stats, frequency buckets.
6. Build Phase 1 training dataset (index mapping cell_id → primitive_id, optionally balancing empty vs non-empty).

### 6.3 Phase 1 Model
7. Implement `models/phase1_cnn.py` exactly per plan (2 conv blocks + FC 128 + dropout).
8. Implement `train/train_phase1.py`:
   - Config loader.
   - Torch dataset & augment (\*only on full-raster pre-slicing path, not per-cell).
   - Metrics: top1, top5, macro-F1, frequency bucket accuracy, confusion matrix.
9. Early training run & baseline metric capture (target ≥92% top-1).

### 6.4 Phase 2 Preparation
10. Build glyph-level grid + label mapping dataset artifacts.
11. Implement transformer model (`models/phase2_transformer.py`) + training script (Phase 2).
12. Integrate primitive prototype embedding initialization.

### 6.5 Evaluation & QA
13. Add evaluation scripts (`eval/eval_phase1.py`, `eval/eval_phase2.py`).
14. Implement attention heatmap to cell overlay (Phase 2 interpretability).
15. Hole & fill audit script (counts of fallback vs non-fallback glyphs; diff mask stats).

### 6.6 Performance / Robustness
16. Parallelize rasterization (multiprocessing pool with chunk commit).
17. Add caching layer keyed by font hash + raster config hash.
18. Introduce determinism checks (hash of first N rasters stable over runs).

## 7. Future (Backlog / Extensions)
- Contrastive / embedding-based primitive learning (Phase 1 variant).
- Multi-channel rasters (distance transform, skeleton).
- End-to-end word-shape / ligature modeling.
- Active learning loop (entropy sampling).
- Quantization / distillation for deployment.

## 8. Risk Review (Current)
| Risk | Status | Mitigation |
|------|--------|------------|
| Hole mis-rendering | Reduced but engine parity differences remain | Integrate FreeType (B) |
| Curve fidelity | Acceptable for baseline | Engine raster + adaptive sampling |
| Primitive imbalance (empty dominance) | Pending measurement | Downsample empties or class weighting |
| Large dataset I/O | Not optimized yet | Consolidation + mmap/LMDB |
| Label noise in rare classes | Unknown until stats | Frequency filter (≥5) + manual inspection |

## 9. Metrics to Capture Soon
- Raster difference (Python vs FreeType): mean pixel error, percent differing pixels.
- Primitive class frequency distribution (top 50 / tail).
- Phase 1 confusion pairs (top misassignments).
- Phase 2 macro vs weighted F1.

## 10. Action Summary (Short List)
Immediate (next session):
1. Implement FreeType integration (B).
2. Re-raster sample subset (e.g., 500 glyphs) compare current vs FreeType for regression.
3. Consolidate full cell corpus post-approval of raster differences.
4. Run K-Means + save centroids + stats.
5. Phase 1 CNN implementation.

---

_This document will be updated after FreeType integration and the first primitive clustering run._