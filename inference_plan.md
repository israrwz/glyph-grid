# Glyph Unicode Inference Engine (Rust + Candle) — Revised Plan

## 0. Purpose

Given a font (TTF/OTF) missing `cmap`/`gsub`, infer Unicode codepoints for each glyph using the two-stage trained models:

1. Rasterize glyph outlines to a 128×128 binary bitmap (no supersampling).
2. Slice into a 16×16 grid of 8×8 cell bitmaps.
3. Run Phase 1 CNN to classify each cell into primitive IDs (0 = empty).
4. Assemble a 16×16 primitive ID grid.
5. Run Phase 2 CNN on primitive grids to obtain glyph-level Unicode predictions.
6. Output per glyph: `glyph_id -> top1_char [top5 chars] prob=<p>` (optional debug raster images).

Zero-area glyphs are skipped entirely (ignored).

## 1. Scope Changes (Compared to Prior Draft)

- Font parsing: Use `owned_ttf_parser` through `ab_glyph` only (avoid separate `ttf-parser` dependency).
- Supersampling removed: Direct rasterization at target 128×128 resolution.
- Occupancy grid & primitive centroids removed (we do not perform nearest-centroid; strictly use Phase 1 CNN).
- Caching mention retained but minimal.
- Non-printables not considered (training label set excludes them).
- Add Python script details for converting `.pt` checkpoints to `.safetensors`.
- Add `--debug-render` option to emit per-glyph raster images under `output/{font_name}/`.
- CLI arguments minimized to `--font` and optional `--debug-render`; all other tunables are compile-time or module constants.

## 2. High-Level Pipeline

```
Font -> Outline Extraction -> Scale/Normalize -> Rasterize 128x128 (binary) 
    -> Slice 256 cells (8x8) -> Phase1 CNN -> Primitive Grid 16x16 (u16)
    -> Phase2 CNN -> Top-K Unicode predictions -> Output
```

Zero-area glyph detection occurs right after outline extraction (skip path).

## 3. Constants (Set in main.rs)

Recommended compile-time constants (no CLI args besides `--font`, `--debug-render`):

```
CANVAS_SIZE: usize = 128
CELL_SIZE: usize = 8
GRID_DIM: usize = 16
PRIMITIVE_CLASSES: usize = 1024 // (0..1023)
DIACRITIC_RATIO_THRESHOLD: f32 = 0.15
DIACRITIC_ADVANCE_THRESHOLD: u32 = 100
BINARIZE_THRESHOLD: f32 = 0.5
EMPTY_THRESHOLD: f32 = 0.0      // Phase1 post-normalization sum cutoff
TOPK: usize = 5
TOP1_ACCEPT_PROB: f32 = 0.55
TOP1_FALLBACK_PROB: f32 = 0.30
PHASE1_WEIGHT_PATH: &str = "data/phase1.safetensors"
PHASE2_WEIGHT_PATH: &str = "data/phase2.safetensors"
LABEL_MAP_PATH: &str = "data/label_map.json"
CHARS_CSV_PATH: &str = "data/chars.csv"
```

(Adjust paths as needed.)

## 4. Font & Outline Handling (owned_ttf_parser via ab_glyph)

Use `ab_glyph::FontRef` (or `FontArc` if owning). Steps:

1. Load font bytes (`std::fs::read`).
2. Construct `FontArc::try_from_slice(&bytes)` (owned).
3. Enumerate glyph IDs: 0..font.number_of_glyphs() (some may be empty).
4. For each glyph:
   - Obtain glyph `GlyphId` and scaled outline:
     - Acquire unscaled outline points (need to traverse TrueType curves).
     - If the font provides a glyph bounding box with zero width/height → skip.
5. Flatten quadratic & cubic Béziers:
   - `ab_glyph_rasterizer::Rasterizer` supports `draw_quad` / `draw_cubic`.
   - Convert any lines & curves directly; minimal adaptive subdivision typically handled by your own logic:
     - For quadratic: recursive split until distance between curve and chord < 0.25px OR constant segmentation (e.g. 8 segments).
     - For cubic: similar approach.

Note: `ab_glyph_rasterizer` does not automatically scale to 128×128—scale coordinates before drawing.

## 5. Diacritic Heuristic

From training:

```
is_diacritic = (advance_width < DIACRITIC_ADVANCE_THRESHOLD) 
               OR (major_dim / em_height) < DIACRITIC_RATIO_THRESHOLD
```

Where:
- `major_dim = max(bbox_w, bbox_h)` of original outline.
- `em_height = ascent - descent` (fallback to bbox_h if metrics missing).
- `advance_width` from font metrics; if unavailable treat as large (so ratio decides).
If `is_diacritic` → scale so major_dim == 64; else major_dim == 128.

Scaling:
- Compute scale = target_dim / major_dim (guard major_dim > 0).
- Translate min_x/min_y → origin (0,0).
- Flip Y axis: after scaling, y' = scaled_height - (y - min_y).

Zero-area glyph (major_dim == 0) -> skip pipeline.

## 6. Rasterization (No Supersampling)

Procedure:

1. Allocate `Rasterizer::new(CANVAS_SIZE, CANVAS_SIZE)`.
2. For each flattened segment:
   - Use `draw_line`, `draw_quad`, `draw_cubic` relative to scaled & flipped coordinates.
3. After drawing, iterate pixels using `for_each_pixel(|idx, alpha| {...})`.
   - `alpha` ∈ [0.0,1.0]; threshold: `alpha >= BINARIZE_THRESHOLD` → 255 else 0.
4. Store result in `Box<[u8; 128*128]>`.

Fill Rule:
- `ab_glyph_rasterizer` internally handles coverage; for non-zero winding parity fallback logic from training was Cairo-specific. With rasterizer:
  - Accept coverage result directly (differences expected to be minimal).
  - If later mismatches appear on complex self-intersecting glyphs, implement manual winding vs parity comparison (optional future improvement).
  
We do NOT implement supersampling now. Note aliasing differences; acceptable if downstream CNN robust (based on training variance).

## 7. Cell Extraction

Row-major slicing:

```
for gy in 0..GRID_DIM:
  for gx in 0..GRID_DIM:
    cell_pixels[gy*GRID_DIM + gx] = &bitmap[(gy*CELL_SIZE)..][(gx*CELL_SIZE)..]
```

Store contiguous interleaved cell data for batch inference:
- Vector length: `glyph_count * GRID_DIM * GRID_DIM * CELL_SIZE * CELL_SIZE`.

Normalization:
- Phase 1 expects float32 normalized to [0,1]; divide by 255.0.

Empty flag pre-computed per cell (all zeros). Keep a `Vec<bool>` for post-prediction masking.

## 8. Phase 1 CNN (Architecture Recap)

Layer order:

1. Conv2d(1→32, kernel=3, stride=1, padding=1) + BatchNorm + ReLU
2. MaxPool2d(kernel=2)  // 8x8 -> 4x4
3. Conv2d(32→64, kernel=3, stride=1, padding=1) + BatchNorm + ReLU
4. MaxPool2d(kernel=2)  // 4x4 -> 2x2
5. Flatten (64 * 2 * 2 = 256)
6. Linear(256→128) + ReLU + Dropout(0.2)
7. Linear(128→1024)

Output shape: (total_cells, 1024). Argmax → primitive ID. Empty cells forced to 0 regardless of prediction.

Candle Implementation:
- Build layer modules manually; ensure identical weight names for loading.
- BN parameters: `weight`, `bias`, `running_mean`, `running_var`.

## 9. Primitive Grid Assembly

Regroup predictions:
```
primitive_grid[glyph_index][gy][gx] = primitive_id
```
Represent per glyph as `[u16; 256]` or `[[u16;16];16]`.

## 10. Phase 2 CNN (Architecture Recap)

Steps:

1. Embedding layer: `Embedding(primitive_vocab=1024, embed_dim=96)` → tensor (G, 16,16,96).
2. Add deterministic 2D sinusoidal positional encoding (same dims).
   - Precompute once; stored in constant.
3. Permute to (G, 96,16,16).

Stage layout (per config):
- Stage 1: 3 × [Conv(96→96, k3,p1), GELU, (optional BN)]
- Downsample: Conv stride=2 (96→192)
- Stage 2: 3 × [Conv(192→192, k3,p1), GELU, BN]
- Downsample: Conv stride=2 (192→256)
- Stage 3: 3 × [Conv(256→256, k3,p1), GELU, BN]

After Stage 3:
- Global average pool → (G, 256)
- Classifier MLP:
  - Linear(256→256) + GELU + Dropout(0.30)
  - Linear(256→num_labels)

Softmax for probabilities; select top K.

CLS token note:
- Training used `use_cls_token: true` but CNN path may effectively treat the pooled representation; if checkpoint includes an extra learned token, incorporate by:
  - Adding a learned `cls_vec` after pooling concatenation then another linear projection.
- If checkpoint weights do not contain CLS-specific parameters (verify), skip; else integrate to remain consistent.

## 11. Unicode Mapping

Two sources:
- `label_map.json`: label string → index.
- `chars.csv`: rows with `label,base_unicode` (unicode char as single codepoint string).

At startup:
1. Load `label_map.json` to build inverse: `index -> label`.
2. Load `chars.csv` into `HashMap<label, char>`.
3. Compose `Vec<char>` aligned by index for O(1) mapping.

Output uses these characters directly.

## 12. Confidence Thresholds

Apply:

```
if top1_prob >= TOP1_ACCEPT_PROB:
    flags = ["OK"]
elif top1_prob >= TOP1_FALLBACK_PROB:
    flags = ["LOW_CONF"]
else:
    flags = ["LOW_CONF","VERY_LOW"]
```

Optional ambiguity:
- If `top1_prob - top2_prob < 0.05` add `AMBIGUOUS`.

These flags included in output line (not required for functional inference, purely diagnostic).

## 13. Debug Rendering (`--debug-render`)

If enabled:
- Create `output/{font_stem}/`.
- For each glyph rasterized (including skipped zero-area? We skip zero-area entirely so do not render them):
  - Save PNG: `glyph_{gid}.png`.
- Optionally annotate with predicted top1 (future enhancement; not required now).

Implementation:
- After rasterization but before inference, write bitmap.
- Use `image` crate or manual PNG encoding (e.g. `png` crate).

Directory creation is idempotent.

## 14. Checkpoint Conversion to Safetensors

### Rationale
Candle prefers `safetensors` for zero-copy and security. We need a script to convert `.pt` Phase 1 & Phase 2 checkpoints.

### Python Script Example

```/dev/null/convert_to_safetensors.py#L1-200
import argparse, torch, json, re
from safetensors.torch import save_file
from pathlib import Path

PREFIX_PATTERN = re.compile(r"^_orig_mod\\.")

def sanitize_state(state_dict):
    sanitized = {}
    for k,v in state_dict.items():
        k2 = PREFIX_PATTERN.sub("", k)
        sanitized[k2] = v
    return sanitized

def convert(ckpt_path: Path, out_path: Path, config_out: Path | None):
    payload = torch.load(str(ckpt_path), map_location="cpu")
    # Phase 2 checkpoints may store model_state within payload
    state = payload.get("model_state", payload)
    state = sanitize_state(state)
    # Write safetensors
    save_file(state, str(out_path))
    # Optional: dump minimal config
    if config_out:
        cfg = payload.get("config", {})
        config_out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", required=True, type=Path, help="Input .pt checkpoint")
    ap.add_argument("--out", required=True, type=Path, help="Output .safetensors path")
    ap.add_argument("--config-out", type=Path, help="Optional JSON config export")
    args = ap.parse_args()
    convert(args.in, args.out, args.config_out)

if __name__ == "__main__":
    main()
```

Usage:

```
python convert_to_safetensors.py --in checkpoints/phase1/best.pt --out data/phase1.safetensors
python convert_to_safetensors.py --in checkpoints/phase2/best.pt --out data/phase2.safetensors --config-out data/phase2_config.json
```

### Loading in Rust (Conceptual)

```/dev/null/inference_phase1.rs#L1-80
use candle_core::{Device, Tensor};
use candle_nn::{Conv2d, Linear, VarBuilder};

pub struct Phase1Model {
    // store layers
}

impl Phase1Model {
    pub fn load(path: &str, device: &Device) -> candle_core::Result<Self> {
        let vb = VarBuilder::from_safetensors(std::fs::read(path)?, device);
        // create layers using vb.get(...)
        // conv1_weight = vb.get(("conv1.weight"), shape)?
        // ...
        Ok(Self { /* layers */ })
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // apply layers
        Ok(logits)
    }
}
```

(Real implementation will enumerate actual parameter names found in safetensors file.)

## 15. Model Key Alignment

Ensure Python conversion preserves names expected by Rust implementation:

Phase 1 typical PyTorch naming:
```
conv_blocks.0.conv.weight
conv_blocks.0.conv.bias
conv_blocks.0.bn.weight
conv_blocks.0.bn.bias
conv_blocks.0.bn.running_mean
conv_blocks.0.bn.running_var
...
fc1.weight
fc1.bias
fc2.weight
fc2.bias
```

Rename strategy:
- Keep same to avoid mapping complexity.
- In Candle, treat nested modules with dot-separated names using `VarBuilder::get` with exact keys.

Phase 2:
Include embedding:
```
embedding.weight
cnn.stages.0.blocks.0.conv.weight
...
classifier.fc1.weight
classifier.fc2.weight
```

Confirm actual prefix by inspecting exported `config_phase2.json` if available.

## 16. Memory & Batch Strategy

- Rasterization: parallel (Rayon) per glyph.
- Phase 1 inference: single batch of all cells (or chunk if memory limited).
- Phase 2 inference: single batch of all glyph grids.
- Memory reuse: preallocate vectors; avoid per-glyph allocations.

## 17. Skipping Zero-Area Glyphs

Condition:
- If bounding box width == 0 OR height == 0 after extraction: skip rasterization and inference.
- Not represented in output; optionally log `[SKIP gid] zero-area`.

## 18. Output Format

Example line:
```
42 -> é [é, e, ė, ê, ë] prob=0.87 flags=OK
```
If low confidence:
```
57 -> . [., ·, : , ؛ , …] prob=0.41 flags=LOW_CONF
```

## 19. Caching (Future)

Not implemented now; simple note:
- Key = BLAKE3 hash of quantized contour points + diacritic flag + scale + model version hashes.
- Value = serialized `GlyphPrediction`.
- Integrate under feature `cache`.

## 20. Error Handling

- Font load failure: exit with error message.
- Missing weights: abort early; indicate required conversion step.
- Mismatched tensor shapes: report which key failed.
- Partial glyph failures: skip glyph; continue.

## 21. Alignment With `infer_chain.py`

Key verified behaviors from existing script:
- Expect raster shape (128,128), enforce.
- Normalize cells dividing by 255.0.
- Empty cell detection: raw max == 0 OR normalized sum <= empty_threshold (we keep threshold = 0.0 default).
- Phase 1 argmax predictions used directly.
- Phase 2 top-K via softmax; probability list emitted.
- Label mapping via JSON label_map + chars.csv.

All preserved. Differences:
- No supersampling; may produce slight difference in edge smoothness (acceptable unless accuracy regression observed).
- Fill rule parity fallback omitted initially (we rely on rasterizer coverage); can be added if mismatch analysis later.

## 22. Implementation Milestones

1. Load font & enumerate glyphs.
2. Outline extraction & zero-area filtering.
3. Scale + diacritic heuristic + coordinate transform.
4. Rasterize with `ab_glyph_rasterizer`.
5. Cell slicing & batch preparation.
6. Load Phase 1 Candle model; run inference.
7. Assemble primitive grids.
8. Load Phase 2 Candle model; run inference.
9. Map indices to Unicode chars; apply confidence flags.
10. Debug rendering (if enabled).
11. Final output.

## 23. Risks & Mitigations (Updated)

| Risk | Mitigation |
|------|------------|
| Aliasing differences (no supersampling) reduce Phase 1 accuracy | If regression observed, reintroduce optional 2× supersampling compile-time feature. |
| Fill rule mismatch for complex self-intersections | Add parity fallback detection if discrepancy threshold discovered. |
| CLS token handling divergence | Inspect Phase 2 state dict for CLS-specific parameters; replicate if present. |
| Memory spike for huge fonts | Batch glyphs (streaming) if `glyph_count * 256` cells exceed threshold. |

## 24. Minimal CLI

Command:
```
glyph-infer --font path/to/font.otf [--debug-render]
```

Help text:
- `--font`: path to font file (required).
- `--debug-render`: write per-glyph raster PNGs for visual inspection.

## 25. Example Rust Skeleton

```/dev/null/main.rs#L1-120
fn main() -> anyhow::Result<()> {
    let args = Cli::parse();
    let font_bytes = std::fs::read(&args.font)?;
    let font = FontArc::try_from_slice(&font_bytes)
        .map_err(|_| anyhow::anyhow!("Failed to parse font"))?;
    let outlines = extract_outlines(&font); // Vec<GlyphOutline>
    let mut rasters = Vec::new();
    outlines.par_iter().for_each(|g| {
        if let Some(r) = rasterize_glyph(g) {
            if args.debug_render {
                save_png(&r.bitmap, format!("output/{}/glyph_{}.png", args.font_stem, g.glyph_id));
            }
            rasters.push(r);
        }
    });
    let (cell_batch, glyph_cell_ranges, empty_mask) = build_cell_batch(&rasters);
    let phase1 = Phase1Model::load(PHASE1_WEIGHT_PATH, &Device::Cpu)?;
    let cell_logits = phase1.forward(&cell_batch)?;
    let primitive_grids = regroup_primitives(cell_logits, glyph_cell_ranges, empty_mask);
    let phase2 = Phase2Model::load(PHASE2_WEIGHT_PATH, &Device::Cpu)?;
    let glyph_logits = phase2.forward(&primitive_grids)?;
    let predictions = postprocess_glyph_logits(glyph_logits, TOPK);
    for p in predictions {
        println!("{}", format_prediction_line(&p));
    }
    Ok(())
}
```

## 26. Summary

This revised plan:
- Uses only `ab_glyph` + `ab_glyph_rasterizer` (via owned_ttf_parser).
- Eliminates supersampling & centroid logic.
- Provides precise diacritic scaling heuristic consistent with training.
- Details safetensors conversion script for both phases.
- Adds debug-render facility.
- Minimizes CLI arguments while retaining configurability through constants.
- Aligns model input/output behaviors with `infer_chain.py`.

Ready to proceed with implementation under this specification.
