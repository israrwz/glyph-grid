#!/usr/bin/env python3
"""
Phase 1 Unseen Glyph Overlay Script

Purpose:
  1. Query the glyph database for the most recent 500 glyph rows (descending id) using
     the provided SQL (with shaped variants filtered out).
  2. Randomly sample N (default 20) "unseen" glyphs from that result set.
  3. Rasterize each glyph via the existing rasterization pipeline (data/rasterize.py)
     to ensure consistency with training data (Cairo / diacritic heuristic etc.).
  4. Run the Phase 1 primitive classifier over the 16x16 grid of 8x8 cells.
  5. Overlay the matched primitive centroid bitmaps (semi‑transparent red, alpha configurable)
     onto the original 128x128 raster.
  6. Save:
       - Per‑glyph overlay images
       - (Optional) a combined panel image
       - A JSON index summarizing glyph metadata and predicted primitive grids

Dependencies:
  - numpy
  - pillow
  - torch
  - sqlite3 (stdlib)
  - pyyaml (for rasterizer config)
  - Existing project modules:
       * data.rasterize : load_config, parse_contours, rasterize_glyph
       * eval.phase1_inference_visualize : load_centroids, load_model, overlay_centroids
    (Falls back to internal minimal re-implementations if imports fail.)

Usage:
  python -m eval.phase1_unseen_overlay \
      --db dataset/glyphs.db \
      --raster-config configs/rasterizer.yaml \
      --centroids assets/centroids/primitive_centroids.npy \
      --checkpoint-dir checkpoints/phase1 \
      --out-dir output/unseen_overlays \
      --panel-out output/unseen_overlays/panel.png \
      --sample 20 \
      --seed 123

Notes:
  - The SQL query is fixed (with optional override via --sql-file).
  - You can point --checkpoint-file to force a specific model instead of newest in dir.
  - Centroid file must include EMPTY row (index 0). If your stored centroids exclude it,
    prepend an all‑zeros row externally before using this script.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise RuntimeError("PyTorch is required (pip install torch).") from e

# ---------------------------------------------------------------------------
# Imports from existing modules (with guarded fallbacks)
# ---------------------------------------------------------------------------
try:
    from data.rasterize import load_config as load_raster_config
    from data.rasterize import parse_contours, rasterize_glyph
except ImportError:
    load_raster_config = None  # type: ignore
    parse_contours = None  # type: ignore
    rasterize_glyph = None  # type: ignore

try:
    from eval.phase1_inference_visualize import (
        load_centroids,
        load_model,
        overlay_centroids,
        cell_to_tensor,
    )
except ImportError:
    load_centroids = None  # type: ignore
    load_model = None  # type: ignore
    overlay_centroids = None  # type: ignore
    cell_to_tensor = None  # type: ignore


# Fallback minimal implementations (only used if original imports failed)
def _fallback_cell_to_tensor(cell: np.ndarray):
    t = torch.from_numpy((cell > 0).astype("float32")).unsqueeze(0)
    return t


def _fallback_overlay_centroids(
    raster: np.ndarray,
    centroid_ids: np.ndarray,
    centroids: np.ndarray,
    alpha: int = 218,
) -> Image.Image:
    base = Image.fromarray(raster, mode="L").convert("RGBA")
    overlay = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    unique_ids = np.unique(centroid_ids)
    bitmaps = {}
    for cid in unique_ids:
        if cid == 0:
            continue
        if cid < 0 or cid >= len(centroids):
            continue
        vec = centroids[cid]
        bmp = (vec.reshape(8, 8) >= 0.5).astype(np.uint8)
        bitmaps[cid] = bmp
    for r in range(16):
        for c in range(16):
            cid = centroid_ids[r, c]
            if cid == 0:
                continue
            bmp = bitmaps.get(cid)
            if bmp is None:
                continue
            rgba = np.zeros((8, 8, 4), dtype=np.uint8)
            rgba[..., 0] = bmp * 255
            rgba[..., 3] = bmp * alpha
            patch = Image.fromarray(rgba, mode="RGBA")
            overlay.paste(patch, (c * 8, r * 8), patch)
    return Image.alpha_composite(base, overlay)


# ---------------------------------------------------------------------------
# SQL (canonical query)
# ---------------------------------------------------------------------------

CANONICAL_SQL = """
SELECT
  g.id       AS id,
  g.glyph_id AS glyph_id,
  g.label    AS label,
  g.contours AS contours_json,
  g.char_class AS char_class,
  g.advance_width AS advance_width,
  COALESCE(f.typo_ascent, f.ascent)  AS used_ascent,
  COALESCE(f.typo_descent, f.descent) AS used_descent,
  f.file_hash AS font_hash,
  g.bounds AS bounds,
  g.width  AS width,
  g.height AS height
FROM glyphs g
JOIN fonts f ON f.file_hash = g.f_id
WHERE g.contours IS NOT NULL
  AND g.has_contours != 0
  AND label NOT LIKE '%_shaped'
ORDER BY g.id DESC
LIMIT 500
"""


# ---------------------------------------------------------------------------
# Data classes / arguments
# ---------------------------------------------------------------------------


@dataclass
class Args:
    db: Path
    raster_config: Path
    centroids: Path
    checkpoint_dir: Path
    checkpoint_file: Optional[Path]
    out_dir: Path
    panel_out: Optional[Path]
    sample: int
    seed: int
    alpha: int
    device: str
    sql_file: Optional[Path]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str]) -> Args:
    ap = argparse.ArgumentParser(
        description="Sample unseen glyphs and overlay Phase 1 primitives."
    )
    ap.add_argument("--db", type=Path, required=True, help="Path to glyph SQLite DB.")
    ap.add_argument(
        "--raster-config", type=Path, required=True, help="Path to rasterizer.yaml."
    )
    ap.add_argument(
        "--centroids",
        type=Path,
        required=True,
        help="Centroid .npy (with EMPTY row 0).",
    )
    ap.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/phase1"))
    ap.add_argument(
        "--checkpoint-file",
        type=Path,
        default=None,
        help="Explicit checkpoint file (overrides dir scan).",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("output/unseen_overlays"))
    ap.add_argument(
        "--panel-out",
        type=Path,
        default=None,
        help="Optional combined panel image path.",
    )
    ap.add_argument(
        "--sample",
        type=int,
        default=20,
        help="Number of glyphs to sample from query result.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=int, default=128, help="Overlay alpha (0-255).")
    ap.add_argument("--device", type=str, default="auto", help="cuda|cpu|auto")
    ap.add_argument(
        "--sql-file",
        type=Path,
        default=None,
        help="Optional custom SQL file to override canonical query.",
    )
    ns = ap.parse_args(argv)
    return Args(
        db=ns.db,
        raster_config=ns.raster_config,
        centroids=ns.centroids,
        checkpoint_dir=ns.checkpoint_dir,
        checkpoint_file=ns.checkpoint_file,
        out_dir=ns.out_dir,
        panel_out=ns.panel_out,
        sample=ns.sample,
        seed=ns.seed,
        alpha=ns.alpha,
        device=ns.device,
        sql_file=ns.sql_file,
    )


def find_latest_checkpoint(dir_path: Path) -> Optional[Path]:
    if not dir_path.exists():
        return None
    files = sorted(dir_path.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def batched_predict_cells(model, device, raster: np.ndarray) -> np.ndarray:
    """
    Split raster into 16x16 cells, run model on all 256 cells in a single batch.
    Returns (16,16) int predicted primitive IDs.
    """
    patches = []
    for r in range(0, 128, 8):
        for c in range(0, 128, 8):
            patch = raster[r : r + 8, c : c + 8]
            patches.append(patch)
    arr = np.stack(patches, axis=0)  # (256,8,8)
    # Convert to tensor (256,1,8,8)
    if cell_to_tensor is not None:
        tensors = torch.from_numpy((arr > 0).astype("float32")).unsqueeze(1)
    else:
        tensors = torch.from_numpy((arr > 0).astype("float32")).unsqueeze(1)
    tensors = tensors.to(device)
    with torch.no_grad():
        logits = model(tensors)
        preds = torch.argmax(logits, dim=1).cpu().numpy()  # (256,)
    grid = preds.reshape(16, 16)
    return grid


def load_and_rasterize_row(row: sqlite3.Row, cfg, engine: str = "cairo"):
    """
    Parse contours -> subpaths -> rasterize via selected engine in cfg.
    """
    if parse_contours is None or rasterize_glyph is None:
        raise RuntimeError(
            "Rasterization components not available; ensure project modules import correctly."
        )
    contours_json = row["contours_json"]
    try:
        subpaths = parse_contours(contours_json) if contours_json else []
    except Exception:
        subpaths = []
    used_ascent = row["used_ascent"]
    used_descent = row["used_descent"]
    adv = row["advance_width"]
    # Override engine dynamically (non-destructive)
    prev_engine = cfg.raster.engine
    cfg.raster.engine = engine
    bitmap, meta = rasterize_glyph(
        subpaths=subpaths,
        used_ascent=used_ascent,
        used_descent=used_descent,
        cfg=cfg,
        advance_width=adv,
    )
    cfg.raster.engine = prev_engine
    return bitmap, meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    random.seed(args.seed)
    np.random.seed(args.seed)

    if load_raster_config is None:
        raise RuntimeError(
            "Cannot import data.rasterize.load_config; adjust PYTHONPATH."
        )

    # Load rasterization config
    cfg = load_raster_config(args.raster_config)
    # Force engine to cairo if available (user expects authoritative)
    if getattr(cfg.raster, "engine", "python") != "cairo":
        cfg.raster.engine = "cairo"

    # Connect DB + fetch rows
    if not args.db.exists():
        raise FileNotFoundError(f"DB not found: {args.db}")
    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    sql = CANONICAL_SQL
    if args.sql_file:
        if not args.sql_file.exists():
            raise FileNotFoundError(f"SQL file not found: {args.sql_file}")
        sql = args.sql_file.read_text(encoding="utf-8")
    rows = con.execute(sql).fetchall()
    con.close()
    if not rows:
        print("[warn] Query returned no rows.", file=sys.stderr)
        return 1
    sample_n = min(args.sample, len(rows))
    chosen = random.sample(rows, sample_n) if sample_n < len(rows) else rows

    # Load centroids + model
    if load_centroids is None:
        raise RuntimeError("Cannot import load_centroids; ensure eval module path.")
    centroids = load_centroids(args.centroids)
    if load_model is None:
        raise RuntimeError("Cannot import load_model from eval module.")
    if args.checkpoint_file:
        ckpt_file = args.checkpoint_file
    else:
        ckpt_file = find_latest_checkpoint(args.checkpoint_dir)
    if ckpt_file is None:
        raise RuntimeError("No checkpoint found.")
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    model = load_model(ckpt_file, num_classes=centroids.shape[0]).to(device).eval()

    # Prepare output
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata_index = []

    overlay_fn = overlay_centroids if overlay_centroids else _fallback_overlay_centroids

    for row in chosen:
        glyph_id = row["glyph_id"]
        label = row["label"]
        try:
            raster, meta = load_and_rasterize_row(row, cfg)
        except Exception as e:
            print(
                f"[warn] Rasterization failed for glyph_id={glyph_id}: {e}",
                file=sys.stderr,
            )
            continue
        if raster.shape != (128, 128):
            print(
                f"[warn] Unexpected raster shape for {glyph_id}: {raster.shape}",
                file=sys.stderr,
            )
            continue

        grid_pred = batched_predict_cells(model, device, raster)
        composite = overlay_fn(raster, grid_pred, centroids, alpha=args.alpha)
        out_path = args.out_dir / f"glyph_{glyph_id}_overlay.png"
        composite.save(out_path)

        metadata_index.append(
            {
                "glyph_id": int(glyph_id),
                "label": label,
                "overlay_image": out_path.as_posix(),
                "primitive_grid": grid_pred.tolist(),
                "meta": meta,
            }
        )

    # Save JSON index
    index_path = args.out_dir / "unseen_overlay_index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(metadata_index, f, ensure_ascii=False, indent=2)
    print(
        f"[info] Wrote index: {index_path} (entries={len(metadata_index)})",
        file=sys.stderr,
    )

    # Optional panel assembly
    if args.panel_out and metadata_index:
        imgs = []
        for rec in metadata_index:
            try:
                imgs.append(Image.open(rec["overlay_image"]).convert("RGBA"))
            except Exception:
                pass
        if imgs:
            cols = min(5, len(imgs))
            w, h = imgs[0].size
            rows_needed = (len(imgs) + cols - 1) // cols
            panel = Image.new("RGBA", (w * cols, h * rows_needed), (0, 0, 0, 0))
            for i, im in enumerate(imgs):
                r = i // cols
                c = i % cols
                panel.paste(im, (c * w, r * h))
            args.panel_out.parent.mkdir(parents=True, exist_ok=True)
            panel.save(args.panel_out)
            print(f"[info] Wrote panel: {args.panel_out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
