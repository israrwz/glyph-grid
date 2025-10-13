# Hierarchical Glyph Recognition Project - Implementation Progress

_Last updated: 2024-01-09 (Cairo integration + hybrid diacritic detection completed)_

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

### 2.7 Cairo Integration (✓ COMPLETED)
- Integrated **pycairo** as authoritative rendering engine for non-zero winding rule.
- Added `engine` config option: `python` (legacy) or `cairo` (authoritative).
- Cairo implementation:
  - Loads parsed contour data (no font files needed).
  - Applies proper non-zero winding fill rule via `cairo.FILL_RULE_WINDING`.
  - Supports even-odd rule via `cairo.FILL_RULE_EVEN_ODD`.
  - Uses `ANTIALIAS_BEST` for smooth edges.
  - Supersampling at 4× (512px) with LANCZOS downsampling to 128px.
- Fixes self-overlapping stroke artifacts that plagued Python implementation.
- Config default set to `engine: cairo`.
- Verified with smoke tests and sample rasterization runs.

### 2.8 Hybrid Diacritic Detection (✓ COMPLETED)
- Analyzed all 524,364 glyphs to optimize diacritic detection accuracy.
- **Hybrid rule implemented:** `(advance_width < 100) OR (ratio < 0.15)` achieves **97.98% accuracy**.
- Key findings:
  - Diacritic median advance_width: 0 (combining marks)
  - Non-diacritic median advance_width: 891
  - Simple ratio threshold (0.25) only achieved 94% accuracy
  - Hybrid rule catches 89.1% of diacritics with 98.4% precision
- **Outlier filtering:** Excludes 2,336 large calligraphic diacritics (10.9% of diacritics) from training:
  - Diacritics with `advance_width >= 100 AND ratio >= 0.15`
  - Includes both 0x610-0x615 (sallalallah) and standard marks (0x064B-0x065F) in ornamental fonts
  - These are legitimate outliers (e.g., Quranic fonts with large decorative forms)
- Verified contour overflow is NOT a good classifier (only 62-80% accuracy)
- Config parameters added: `diacritic_advance_threshold`, `exclude_large_diacritics`

### 2.9 Proven Execution
- Rasterized first 1K glyph subset multiple times with different fill rules.
- Verified diacritic detection, metadata integrity, and absence of false "empty" glyphs after parser fixes.
- Confirmed correct counter rendering in cleaned font set (after removal of problematic font).
- Cairo engine produces correct winding rule output without self-overlap artifacts.

## 3. In Progress / Partially Done
- Phase 1 model code (CNN) not yet implemented in `models/phase1_cnn.py`.
- Training loops (`train/train_phase1.py`, `train/train_phase2.py`) pending.
- Joint fine-tuning pipeline placeholder only.
- No automated label frequency bucketing metrics yet (planned for evaluation).
- No cell-level LMDB / memory-mapped optimization (current NumPy path only).
- Advanced error analysis scripts not implemented (attention heatmaps etc. stubbed conceptually).

## 4. Known Technical Gaps / Issues
| Area | Current State | Impact | Planned Remedy |
|------|---------------|--------|----------------|
| ~~Outline fidelity~~ | ✓ Cairo handles curves natively | None | ~~Resolved via Cairo~~ |
| ~~Hole semantics~~ | ✓ Cairo winding rule | None | ~~Resolved via Cairo~~ |
| ~~Self-overlap stroke artifacts~~ | ✓ Cairo winding rule | None | ~~Resolved via Cairo~~ |
| ~~Diacritic detection~~ | ✓ Hybrid rule (97.98% acc) | None | ~~Resolved via advance_width + ratio~~ |
| Performance scaling | Single-process Python | Longer runtime for full corpus | Parallel chunking (Cairo is C-backed) |
| Primitive sampling | Reservoir + full read loop | I/O overhead on large sets | Cell shards + memory map / LMDB |
| Metadata QA | Basic fields + engine tag | N/A | Current state adequate |
| Phase 1 dataset | Not materialized | Blocks primitive CNN training | Implement dataset + dataloader soon |

## 5. Decision Log (Highlights)
- Adopt binary 0/255 storage for rasters (plan alignment, compression friendly).
- Selected winding as the baseline fill rule due to closer match with actual font engines.
- **Integrated Cairo (pycairo) for authoritative winding rule** instead of FreeType (no font files needed).
- Cairo works directly with parsed contour data already in database.
- Removed Python winding heuristics; Cairo is now default engine.
- Supersample factor increased to 4× (512px render) with LANCZOS downsampling for quality.
- **Hybrid diacritic detection:** Use `advance_width < 100` OR `ratio < 0.15` (97.98% accuracy).
- **Exclude 2,336 large diacritic outliers** (calligraphic forms in Quranic/decorative fonts) from training.
- Contour overflow tested but rejected as classifier (poor accuracy vs hybrid rule).
- Deferred complex curve error metrics until after baseline model accuracy established.

## 6. Next Immediate TODOs

### 6.1 Rasterization & Rendering ✓ COMPLETED
1. ~~(B) Integrate authoritative non-zero winding raster~~ ✓ Done via Cairo
2. ~~Add geometric nesting normalization~~ ✓ Not needed (Cairo handles it)
3. ~~Implement adaptive subdivision~~ ✓ Not needed (Cairo native curves)

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
| ~~Hole mis-rendering~~ | ✓ Resolved via Cairo | ~~Cairo winding rule~~ |
| ~~Curve fidelity~~ | ✓ Resolved via Cairo | ~~Cairo native curves~~ |
| Primitive imbalance (empty dominance) | Pending measurement | Downsample empties or class weighting |
| Large dataset I/O | Not optimized yet | Consolidation + mmap/LMDB |
| Label noise in rare classes | Unknown until stats | Frequency filter (≥5) + manual inspection |

## 9. Metrics to Capture Soon
- ~~Raster difference (Python vs Cairo)~~ ✓ Cairo is now default (Python deprecated).
- ~~Diacritic detection accuracy~~ ✓ Hybrid rule achieves 97.98% (advance_width + ratio).
- Primitive class frequency distribution (top 50 / tail).
- Phase 1 confusion pairs (top misassignments).
- Phase 2 macro vs weighted F1.

## 10. Action Summary (Short List)
Immediate (next session):
1. ~~Implement authoritative rendering~~ ✓ Done via Cairo
2. ~~Optimize diacritic detection~~ ✓ Done via hybrid rule (97.98% accuracy)
3. Run full corpus rasterization with Cairo engine + outlier filtering.
4. Consolidate full cell corpus.
5. Run K-Means (k=1023) + save centroids + stats.
6. Phase 1 CNN implementation.

---

_This document was updated after Cairo integration and hybrid diacritic detection. Next: full corpus raster + primitive clustering._