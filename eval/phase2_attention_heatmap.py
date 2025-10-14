#!/usr/bin/env python3
"""
Phase 2 Attention Heatmap Visualization
=======================================

Generates attention heatmaps from the Phase 2 glyph classification transformer
(model: primitive ID grid → glyph label).

Primary Outputs (per requested glyph):
  - Aggregated saliency heatmap (CLS → spatial tokens) merged over:
       * heads (mean)
       * optionally layers (mean or per-layer if --per-layer)
  - (Optional) Per-layer heatmaps (if --per-layer) showing CLS attention distribution.
  - JSON sidecar with raw attention statistics (min/max/mean) for reproducibility.

Usage (example):
  python -m eval.phase2_attention_heatmap \
      --config configs/phase2.yaml \
      --checkpoint-dir checkpoints/phase2 \
      --grids-dir data/processed/grids \
      --label-map data/processed/label_map.json \
      --glyph-ids 1234 5678 9012 \
      --out-dir output/phase2_attn \
      --per-layer

Sampling from existing splits (mutually exclusive with --glyph-ids):
  python -m eval.phase2_attention_heatmap \
      --config configs/phase2.yaml \
      --checkpoint-dir checkpoints/phase2 \
      --grids-dir data/processed/grids \
      --label-map data/processed/label_map.json \
      --split-file data/processed/splits/phase2_val_ids.txt \
      --sample 16 \
      --out-dir output/phase2_attn_val

Key Assumptions:
  - Grids stored as .u16 raw (16x16) or .npy; loader searches <id>.u16 then <id>.npy.
  - Phase 2 checkpoint (.pt) contains either:
        {'model_state': state_dict}
        {'model_state_dict': state_dict}
        direct state_dict
  - Config YAML matches schema in configs/phase2.yaml (sections: input, model, etc.).
  - Transformer built via models.phase2_transformer.build_phase2_model (already added).

Attention Extraction Semantics:
  - The model forward(..., return_attn=True) returns list[Tensor] of per-layer attention
    weights. For nn.MultiheadAttention(batch_first=True), shape is (B, heads, T, T).
  - We aggregate:
        layer_mean = mean over heads -> (B, T, T)
        If CLS token used:
           CLS-to-spatial map = layer_mean[:, 0, 1:]  (exclude CLS itself)
           Reshape to (16,16)
        Else:
           We approximate saliency per token by mean over sources: mean(layer_mean[:, i, :])
           producing a (T,) vector → reshape (16,16).
  - Final aggregated heatmap (if not per-layer) is mean over layers.

Color Mapping:
  - Uses matplotlib if available; otherwise a simple red colormap (value → (val,0,0)).
  - Normalization: by default min-max per heatmap; can fix global range with --global-scale.

Outputs:
  out_dir/
    glyph_<id>_heatmap.png
    glyph_<id>_layer<L>.png (if --per-layer)
    glyph_<id>_attn.json

Limitations / Future:
  - No integration of original raster overlay (could add with optional --rasters-dir).
  - No patch grouping aware reshaping (if patch grouping was enabled, indexing adjusts
    automatically only for 16x16 baseline). Will warn if seq length unexpected.
  - No attention rollout; only direct CLS attention.

Dependencies:
  - torch, numpy, pyyaml, pillow
  - (optional) matplotlib for better colormap

License: Project root license.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

try:
    import torch
except ImportError as e:
    raise RuntimeError("PyTorch is required for this script.") from e

# Optional libs
try:
    import yaml
except ImportError as e:
    raise RuntimeError("pyyaml required (pip install pyyaml).") from e

try:
    import matplotlib.pyplot as plt  # type: ignore

    _HAVE_MPL = True
except Exception:  # pragma: no cover
    _HAVE_MPL = False

# ---------------------------------------------------------------------------
# Model Import
# ---------------------------------------------------------------------------
try:
    from models.phase2_transformer import build_phase2_model
except ImportError as e:
    raise RuntimeError(
        "Failed importing phase2 model. Ensure PYTHONPATH includes project root."
    ) from e


# ---------------------------------------------------------------------------
# CLI / Config
# ---------------------------------------------------------------------------


@dataclass
class Args:
    config: Path
    checkpoint_dir: Path
    checkpoint_file: Optional[Path]
    grids_dir: Path
    label_map: Path
    glyph_ids: List[int]
    split_file: Optional[Path]
    sample: Optional[int]
    out_dir: Path
    per_layer: bool
    global_scale: bool
    cmap: str
    format: str
    verbose: bool
    device: str
    centroid_file: Optional[Path]


def parse_args(argv: Optional[Sequence[str]] = None) -> Args:
    p = argparse.ArgumentParser(
        description="Phase 2 transformer attention heatmap visualization."
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Phase 2 YAML config (for model architecture).",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/phase2"),
        help="Directory containing phase 2 checkpoints (*.pt).",
    )
    p.add_argument(
        "--checkpoint-file",
        type=Path,
        default=None,
        help="Explicit checkpoint file (overrides --checkpoint-dir).",
    )
    p.add_argument(
        "--grids-dir",
        type=Path,
        required=True,
        help="Directory containing per-glyph grids (<id>.u16 or <id>.npy).",
    )
    p.add_argument(
        "--label-map",
        type=Path,
        required=True,
        help="label_map.json mapping label string → class index (used only to infer num_labels).",
    )
    p.add_argument(
        "--glyph-ids",
        type=int,
        nargs="*",
        default=[],
        help="Explicit glyph ids to visualize (mutually exclusive with --split-file/--sample).",
    )
    p.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Optional file listing glyph ids for sampling (e.g. phase2_val_ids.txt).",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        help="If provided with --split-file (or without explicit glyph ids), randomly sample N glyphs.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/phase2_attn"),
        help="Output directory for heatmaps and JSON stats.",
    )
    p.add_argument(
        "--per-layer",
        action="store_true",
        help="Write per-layer heatmaps instead of (or in addition to) aggregated mean.",
    )
    p.add_argument(
        "--global-scale",
        action="store_true",
        help="Normalize heatmaps using global min/max across all selected glyphs (default: per-image).",
    )
    p.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap (fallback grayscale/red if MPL missing).",
    )
    p.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "jpg"],
        help="Image output format.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="'auto' | 'cpu' | 'cuda' (if available).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument(
        "--centroid-file",
        type=Path,
        default=None,
        help="Optional primitive_centroids.npy for embedding init (overrides config path).",
    )
    ns = p.parse_args(argv)

    return Args(
        config=ns.config,
        checkpoint_dir=ns.checkpoint_dir,
        checkpoint_file=ns.checkpoint_file,
        grids_dir=ns.grids_dir,
        label_map=ns.label_map,
        glyph_ids=list(ns.glyph_ids),
        split_file=ns.split_file,
        sample=ns.sample,
        out_dir=ns.out_dir,
        per_layer=ns.per_layer,
        global_scale=ns.global_scale,
        cmap=ns.cmap,
        format=ns.format,
        verbose=ns.verbose,
        device=ns.device,
        centroid_file=ns.centroid_file,
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def log(msg: str, *, verbose: bool):
    if verbose:
        print(f"[info] {msg}", file=sys.stderr)


def find_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    if not ckpt_dir.exists():
        return None
    cands = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_label_map(path: Path) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def choose_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def load_centroids_if_requested(
    cfg: Dict, override_path: Optional[Path], verbose=False
):
    path = override_path
    if path is None:
        # Look in config
        model_init = (cfg.get("model") or {}).get("init") or {}
        if model_init.get("embedding_from_centroids", True):
            # Try config.data.primitive_centroids first, then typical default
            data_cfg = cfg.get("data") or {}
            pc = data_cfg.get("primitive_centroids")
            if pc:
                path = Path(pc)
    if path and path.exists():
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[1] != 64:
            raise ValueError(f"Centroids shape unexpected: {arr.shape}")
        if verbose:
            print(
                f"[info] Loaded centroids from {path} shape={arr.shape}",
                file=sys.stderr,
            )
        return torch.from_numpy(arr.astype(np.float32))
    return None


def load_checkpoint(model: torch.nn.Module, path: Path, strict: bool = True):
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict):
        for key in ("model_state", "model_state_dict"):
            if key in raw and isinstance(raw[key], dict):
                model.load_state_dict(raw[key], strict=strict)
                return
        # Maybe direct state_dict
        param_keys = [k for k in raw.keys() if isinstance(raw[k], torch.Tensor)]
        if param_keys:
            model.load_state_dict(raw, strict=strict)
            return
    raise RuntimeError(f"Could not interpret checkpoint file: {path}")


def read_glyph_ids_from_split(path: Path) -> List[int]:
    ids: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.append(int(line))
            except ValueError:
                pass
    return ids


def pick_glyph_ids(args: Args) -> List[int]:
    if args.glyph_ids:
        return args.glyph_ids
    candidate_ids: List[int] = []
    if args.split_file and args.split_file.exists():
        candidate_ids = read_glyph_ids_from_split(args.split_file)
    else:
        # Fallback: all IDs with grid files traversal (WARNING: may be large).
        # Strategy: list .u16 only (avoid enumerating .npy duplicates).
        candidate_ids = [
            int(p.stem) for p in args.grids_dir.glob("*.u16") if p.stem.isdigit()
        ]
    if args.sample and args.sample < len(candidate_ids):
        import random

        random.seed(42)
        candidate_ids = random.sample(candidate_ids, args.sample)
    return candidate_ids


def load_grid(grids_dir: Path, gid: int) -> np.ndarray:
    raw_path = grids_dir / f"{gid}.u16"
    if raw_path.exists():
        arr = np.fromfile(raw_path, dtype=np.uint16)
        if arr.size != 256:
            raise ValueError(f"Grid length mismatch for {gid}: {arr.size}")
        return arr.reshape(16, 16)
    npy_path = grids_dir / f"{gid}.npy"
    if npy_path.exists():
        arr = np.load(npy_path)
        if arr.shape != (16, 16):
            raise ValueError(f"Grid shape mismatch for {gid}: {arr.shape}")
        return arr.astype(np.uint16)
    raise FileNotFoundError(f"No grid file found for glyph_id={gid} (.u16 or .npy).")


def normalize_heatmap(
    hmap: np.ndarray, global_min: float, global_max: float
) -> np.ndarray:
    denom = (global_max - global_min) if (global_max - global_min) > 1e-12 else 1.0
    out = (hmap - global_min) / denom
    return np.clip(out, 0.0, 1.0)


def to_colormap(hmap: np.ndarray, cmap: str) -> Image.Image:
    """
    Convert normalized (0..1) 2D array to RGBA image.
    """
    if _HAVE_MPL:
        cmap_obj = plt.get_cmap(cmap)
        colored = cmap_obj(hmap)  # (H,W,4) float RGBA
        img = (colored * 255).astype(np.uint8)
        return Image.fromarray(img, mode="RGBA")
    # Fallback: simple red gradient
    h, w = hmap.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    val = (hmap * 255).astype(np.uint8)
    rgba[..., 0] = val
    rgba[..., 3] = 255
    return Image.fromarray(rgba, mode="RGBA")


def save_image(img: Image.Image, path: Path, fmt: str = "png"):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path.with_suffix(f".{fmt}"), format=fmt.upper())


# ---------------------------------------------------------------------------
# Attention Processing
# ---------------------------------------------------------------------------


def extract_cls_heatmaps(
    attn_list: List[torch.Tensor], use_cls: bool, verbose=False
) -> List[np.ndarray]:
    """
    attn_list: list of (B, heads, T, T) attention tensors (raw softmax outputs).
    Returns: list of numpy arrays shape (B, 16, 16) per layer containing CLS->token attention
             aggregated over heads. If no CLS, returns token saliency via mean inbound attention.
    """
    heatmaps: List[np.ndarray] = []
    for layer_idx, attn in enumerate(attn_list):
        if attn is None:
            continue
        if attn.dim() != 4:
            raise ValueError(
                f"Expected attention shape (B,H,T,T); got {tuple(attn.shape)}"
            )
        B, H, T, T2 = attn.shape
        if T != T2:
            raise ValueError("Attention must be square: got T != T2.")
        # Determine spatial token count
        if use_cls:
            spatial_tokens = T - 1
        else:
            spatial_tokens = T
        # Expected baseline 16*16=256; allow patch grouping gracefully by sqrt heuristic
        spatial_dim = int(spatial_tokens)
        side = int(math.sqrt(spatial_dim))
        if side * side != spatial_dim:
            if verbose:
                print(
                    f"[warn] Non-square spatial token count {spatial_tokens}; skipping layer {layer_idx}",
                    file=sys.stderr,
                )
            continue
        # Aggregate heads
        layer_mean = attn.mean(dim=1)  # (B,T,T)
        if use_cls:
            # CLS attends to others: take row 0
            cls_row = layer_mean[:, 0, 1:]  # (B, spatial_tokens)
            maps = cls_row.view(B, side, side).detach().cpu().numpy()
        else:
            # Mean attention "received" by each token (column-wise)
            token_importance = layer_mean.mean(dim=1)  # mean over source dimension
            maps = token_importance.view(B, side, side).detach().cpu().numpy()
        heatmaps.append(maps)
    return heatmaps  # list length = n_layers each shape (B,side,side)


def aggregate_heatmaps(layer_maps: List[np.ndarray]) -> np.ndarray:
    """
    Mean over layers: Input list of (B,H,W) → output (B,H,W)
    """
    if not layer_maps:
        return np.zeros((0, 16, 16), dtype=np.float32)
    stacked = np.stack(layer_maps, axis=0)  # (L,B,H,W)
    return stacked.mean(axis=0)


# ---------------------------------------------------------------------------
# Main Logic
# ---------------------------------------------------------------------------


def run(args: Args) -> int:
    log("Loading config...", verbose=args.verbose)
    cfg = load_yaml(args.config)
    label_map = load_label_map(args.label_map)
    num_labels = len(label_map)

    device = choose_device(args.device)
    log(f"Using device {device}", verbose=args.verbose)

    centroids_tensor = load_centroids_if_requested(
        cfg, args.centroid_file, verbose=args.verbose
    )

    # Build model
    model = build_phase2_model(
        cfg, num_labels=num_labels, primitive_centroids=centroids_tensor
    )
    model.to(device)
    model.eval()

    # Checkpoint
    ckpt_path = args.checkpoint_file
    if ckpt_path is None:
        ckpt_path = find_latest_checkpoint(args.checkpoint_dir)
    if ckpt_path is None or not ckpt_path.exists():
        print(
            f"[error] No checkpoint found (dir={args.checkpoint_dir}).", file=sys.stderr
        )
        return 2
    log(f"Loading checkpoint: {ckpt_path}", verbose=args.verbose)
    load_checkpoint(model, ckpt_path, strict=True)

    # Glyph selection
    glyph_ids = pick_glyph_ids(args)
    if not glyph_ids:
        print("[error] No glyph ids selected.", file=sys.stderr)
        return 3

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_agg_maps: List[np.ndarray] = []
    glyph_index: List[int] = []

    per_layer_store: List[List[np.ndarray]] = []  # per glyph: list[layer] heatmap (H,W)

    with torch.no_grad():
        for gid in glyph_ids:
            try:
                grid = load_grid(args.grids_dir, gid)  # (16,16)
            except Exception as e:
                print(f"[warn] skip gid={gid}: {e}", file=sys.stderr)
                continue
            grid_tensor = (
                torch.from_numpy(grid.astype(np.int64)).unsqueeze(0).to(device)
            )  # (1,16,16)

            logits, attn_list = model(
                grid_tensor, return_attn=True
            )  # attn_list list[(B,H,T,T)]
            if not isinstance(attn_list, list) or not attn_list:
                print(f"[warn] No attention returned for gid={gid}", file=sys.stderr)
                continue

            use_cls = getattr(model.cfg, "use_cls_token", True)
            layer_maps = extract_cls_heatmaps(
                attn_list, use_cls=use_cls, verbose=args.verbose
            )  # list[(B,side,side)]
            if not layer_maps:
                print(f"[warn] Empty layer maps for gid={gid}", file=sys.stderr)
                continue
            agg = aggregate_heatmaps(layer_maps)  # (B,side,side)
            agg_map = agg[0]  # first batch
            all_agg_maps.append(agg_map)
            glyph_index.append(gid)

            if args.per_layer:
                per_layer_maps = [lm[0] for lm in layer_maps]  # each (side,side)
                per_layer_store.append(per_layer_maps)

        if not all_agg_maps:
            print("[error] No attention maps produced.", file=sys.stderr)
            return 4

    # Global normalization if requested
    if args.global_scale:
        global_min = float(min(m.min() for m in all_agg_maps))
        global_max = float(max(m.max() for m in all_agg_maps))
    else:
        global_min = global_max = 0.0  # dummy; per-image branch handles normalization

    # Write outputs
    stats_summary = []
    for idx, gid in enumerate(glyph_index):
        hmap = all_agg_maps[idx]
        if args.global_scale:
            norm = normalize_heatmap(hmap, global_min, global_max)
        else:
            norm = normalize_heatmap(hmap, float(hmap.min()), float(hmap.max()))
        img = to_colormap(norm, args.cmap)
        save_image(img, out_dir / f"glyph_{gid}_heatmap", fmt=args.format)

        layer_json_records = []
        if args.per_layer:
            pl_maps = per_layer_store[idx]
            for li, lm in enumerate(pl_maps):
                if args.global_scale:
                    lnorm = normalize_heatmap(lm, global_min, global_max)
                else:
                    lnorm = normalize_heatmap(lm, float(lm.min()), float(lm.max()))
                limg = to_colormap(lnorm, args.cmap)
                save_image(
                    limg, out_dir / f"glyph_{gid}_layer{li:02d}", fmt=args.format
                )
                layer_json_records.append(
                    {
                        "layer": li,
                        "min": float(lm.min()),
                        "max": float(lm.max()),
                        "mean": float(lm.mean()),
                        "std": float(lm.std()),
                    }
                )

        # JSON metadata per glyph
        meta = {
            "glyph_id": gid,
            "agg": {
                "min": float(hmap.min()),
                "max": float(hmap.max()),
                "mean": float(hmap.mean()),
                "std": float(hmap.std()),
            },
            "layers": layer_json_records,
        }
        with open(out_dir / f"glyph_{gid}_attn.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        stats_summary.append(meta)

    # Master index
    with open(out_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump({"glyphs": stats_summary}, f, indent=2)

    log(f"Wrote {len(glyph_index)} glyph heatmaps to {out_dir}", verbose=args.verbose)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except Exception as e:
        print(f"[fatal] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
