#!/usr/bin/env python3
"""
Rasterization & Primitive Vocabulary Preparation (Plan-Aligned Scaffold)

Implements Steps (from NEW_PLAN.md):
  - Section 4.*  Rasterization of glyph vector contours to 128x128 binary masks.
  - Section 5    16x16 grid (8x8 cells) extraction.
  - Section 6.1  Primitive vocabulary initialization with K-Means (k=1023) on non-empty cells.

Data Flow (intended):
  1. Source glyph records from SQLite DB: dataset/glyphs.db
     Expected table (illustrative schema; adapt to real DB):
       glyphs(
         glyph_id INTEGER PRIMARY KEY,
         label TEXT,
         contours_json TEXT,        -- JSON array-of-arrays of path commands (Section 3.2 format)
         used_ascent REAL,          -- vertical metric used for EM span
         used_descent REAL,
         font_hash TEXT
       )
     (You can extend schema with additional fields; the scaffold only uses the above.)

  2. Label filtering (Section 3.3):
       - Count frequency per distinct label.
       - Keep labels with frequency >= min_count (config: label_filter.min_count).
       - Optionally drop shaped variants (config currently false).

  3. Rasterize each glyph:
       - Parse contours JSON into subpaths.
       - Compute bbox, major dimension, ratio = major_dim / em_height.
       - If ratio < diacritic_threshold => diacritic ⇒ target_dim = 64 else 128.
       - Uniformly scale so largest dimension == target_dim.
       - Translate so (min_x, min_y) => (0,0).
       - Supersample canvas: size = 2 * 128 = 256 (fixed supersample factor = 2).
       - Flatten curves: 8 subdivisions per cubic (Section 4.1).
       - Fill polygons using orientation (outer positive area = fill, negative = hole carve-out).
       - Downsample to 128 using bicubic, threshold at 0.5 -> binary (uint8 {0,255}).

  4. Extract 16x16 grid: slice 8x8 blocks (row-major).
       - All-zero block = EMPTY cell (used later as class 0).

  5. (Optional after rasterization) Run K-Means over up to 1,000,000 sampled non-empty cells:
       - Flatten each 8x8 to 64D vector.
       - Use k=1023 clusters (IDs 1..1023). ID 0 reserved for empty.
       - Save centroids to assets/centroids/primitive_centroids.npy
       - Assign each non-empty cell cluster ID+1; empty => 0.
       - Persist per-glyph primitive ID grids (<glyph_id>_ids.u16 .npy) or a single packed store.

Determinism:
  - Fixed random seed for sampling & K-Means init.
  - Order of glyph processing is stable if DB query ORDER BY glyph_id.

This file is a scaffold: functions are implemented with clean, documented code,
but you may need to adapt DB field names, error handling, and performance
optimizations (e.g., multiprocessing, LMDB storage) as the project matures.

Dependencies (suggested minimal):
  - pillow (PIL)
  - numpy
  - pyyaml
  - scikit-learn (for efficient MiniBatchKMeans; falls back to a naive KMeans if absent)

Usage Examples:
  Rasterize & build grids + metadata:
    python data/rasterize.py rasterize --config configs/rasterizer.yaml

  Build primitive vocabulary (K-Means) after rasters/grids exist:
    python data/rasterize.py kmeans --config configs/rasterizer.yaml --max-cells 1000000

  Do both sequentially:
    python data/rasterize.py full --config configs/rasterizer.yaml --max-cells 1000000

Outputs (matching current rasterizer.yaml expectations):
  - data/rasters/<glyph_id>.png   (binary 128x128)
  - data/grids/<glyph_id>.npy     (uint8 16x16 occupancy grid: 0 or 1)  -- pre-primitive IDs
  - data/rasters/metadata.jsonl   (one JSON per glyph)
  - assets/centroids/primitive_centroids.npy  (k=1023, 64D float32)
  - data/grids/<glyph_id>_ids.npy (uint16 16x16 primitive ID grid) AFTER K-Means

IMPORTANT: This scaffold avoids introducing config keys not present in
rasterizer.yaml (except derived output file naming).

NOTE ON EMPTY / LEGACY CONTOURS:
Some database rows may have an empty 'contours' / 'contours_json' field (empty string or '[]'),
or may already store flattened polylines (e.g. [[x0,y0],[x1,y1],...]) without explicit command
tuples. The parser now:
  - Returns [] immediately for falsy / empty / '[]' strings.
  - Detects the “already polyline” case (top-level list of numeric pairs) and wraps it
    as a single subpath.
  - Gracefully skips malformed entries instead of raising.
This allows the pipeline to progress even when many glyphs currently have no vector data.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: pyyaml. Install with `pip install pyyaml`."
    ) from e

import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: pillow. Install with `pip install pillow`."
    ) from e

# Try scikit-learn for MiniBatchKMeans; fallback to naive KMeans if unavailable.
try:
    from sklearn.cluster import MiniBatchKMeans  # type: ignore

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# Try cairo for authoritative winding rule rendering
try:
    import cairo

    CAIRO_AVAILABLE = True
except ImportError:
    CAIRO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration Dataclasses (mapped from rasterizer.yaml without expanding scope)
# ---------------------------------------------------------------------------


@dataclass
class RasterConfig:
    canvas_size: int
    supersample_factor: int
    curve_subdivisions: int
    binarize_threshold: float
    main_target_dim: int
    diacritic_target_dim: int
    diacritic_ratio_threshold: float
    diacritic_advance_threshold: int
    exclude_large_diacritics: bool
    fill_rule: str
    store_mode: str
    engine: str = "python"  # "python" or "cairo"


@dataclass
class GridConfig:
    rows: int
    cols: int
    cell_px: int
    empty_primitive_id: int
    export_mode: str


@dataclass
class PathsConfig:
    glyph_db: Path
    rasters_dir: Path
    grids_dir: Path
    metadata_path: Path


@dataclass
class LabelFilterConfig:
    min_count: int
    drop_shaped_variants: bool


@dataclass
class FullConfig:
    seed: int
    deterministic: bool
    raster: RasterConfig
    grid: GridConfig
    paths: PathsConfig
    label_filter: LabelFilterConfig


# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> FullConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Minimal validation: rely only on keys that exist in rasterizer.yaml
    # (We map nested sections to dataclasses.)
    sources = raw.get("sources", {})
    outputs = raw.get("outputs", {})
    raster = raw.get("raster", {})
    grid = raw.get("grid", {})
    label_filter = raw.get("label_filter", {})

    cfg = FullConfig(
        seed=int(raw.get("seed", 42)),
        deterministic=bool(raw.get("deterministic", True)),
        raster=RasterConfig(
            canvas_size=int(raster["canvas_size"]),
            supersample_factor=int(raster.get("supersample_factor", 2)),
            curve_subdivisions=int(raster.get("curve_subdivisions", 8)),
            binarize_threshold=float(raster.get("binarize_threshold", 0.5)),
            main_target_dim=int(raster["main_target_dim"]),
            diacritic_target_dim=int(raster["diacritic_target_dim"]),
            diacritic_ratio_threshold=float(raster["diacritic_ratio_threshold"]),
            diacritic_advance_threshold=int(
                raster.get("diacritic_advance_threshold", 100)
            ),
            exclude_large_diacritics=bool(raster.get("exclude_large_diacritics", True)),
            fill_rule=str(raster.get("fill_rule", "orientation")),
            store_mode=str(raster.get("store_mode", "binary_uint8")),
            engine=str(raster.get("engine", "python")),
        ),
        grid=GridConfig(
            rows=int(grid["rows"]),
            cols=int(grid["cols"]),
            cell_px=int(grid["cell_px"]),
            empty_primitive_id=int(grid["empty_primitive_id"]),
            export_mode=str(grid.get("export_mode", "occupancy")),
        ),
        paths=PathsConfig(
            glyph_db=Path(sources["glyph_db"]),
            rasters_dir=Path(outputs["rasters_dir"]),
            grids_dir=Path(outputs["grids_dir"]),
            metadata_path=Path(outputs["metadata_path"]),
        ),
        label_filter=LabelFilterConfig(
            min_count=int(label_filter.get("min_count", 5)),
            drop_shaped_variants=bool(label_filter.get("drop_shaped_variants", False)),
        ),
    )
    return cfg


# ---------------------------------------------------------------------------
# Database Access
# ---------------------------------------------------------------------------


def iter_glyph_rows(db_path: Path) -> Iterator[Dict[str, Any]]:
    """
    Yields glyph rows as dicts after joining font metrics (fonts table) and adapting to
    the actual schema discovered in glyphs.db.

    Actual schema (glyphs):
      id, unicode_codepoint, f_id, glyph_id, glyph_name, unicode_string,
      joining_group, char_class, label, contours, contour_count, hole_count,
      orientation, has_contours, advance_width, left_side_bearing, bounds,
      width, height

    Fonts table (fonts):
      file_hash (PK), ascent, descent, typo_ascent, typo_descent, units_per_em, ...

    Mapping to plan fields:
      contours_json  <- glyphs.contours          (stored JSON path list)
      used_ascent    <- COALESCE(fonts.typo_ascent, fonts.ascent)
      used_descent   <- COALESCE(fonts.typo_descent, fonts.descent)
      font_hash      <- fonts.file_hash
      original bbox  <- parse from glyphs.bounds (JSON "[min_x,min_y,max_x,max_y]") later if needed

    We only yield rows where contours IS NOT NULL and has_contours != 0.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            """
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
            ORDER BY g.id
            """
        )
        # GROUP BY g.glyph_id
        for row in cur:
            yield dict(row)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Label Filtering
# ---------------------------------------------------------------------------


def build_label_whitelist(
    rows: Iterable[Dict[str, Any]], min_count: int, drop_shaped_variants: bool
) -> set[str]:
    freq: Dict[str, int] = {}
    # buffered = []
    for r in rows:
        # buffered.append(r)
        label = r["label"]
        freq[label] = freq.get(label, 0) + 1

    def is_shaped(label: str) -> bool:
        # Example suffixes (init/medi/final) - adapt as needed
        return label.endswith(("_init", "_medi", "_final"))

    whitelist = {l for l, c in freq.items() if c >= min_count}
    return whitelist


# ---------------------------------------------------------------------------
# Contour Parsing & Curve Flattening
# ---------------------------------------------------------------------------


def parse_contours(contours_json: str) -> List[List[Tuple[float, float]]]:
    """
    Enhanced contour parser supporting mixed TrueType-style commands, nested subpaths and
    robust quadratic handling.

    Supported input structures (auto-detected):
      1. Command lists (plan format): ["moveTo",[x,y]], ["lineTo",[x,y]],
         ["qCurveTo", [[(optional_mid_or_ctrl)...], [x,y]]],
         ["cubicTo"/"curveTo", [[c1x,c1y],[c2x,c2y],[x,y]]], ["closePath", null]
      2. Already flattened polyline: [[x0,y0],[x1,y1],...]
      3. List of polylines: [ [[...]], [[...]], ... ]
      4. List of subpaths each as a command list (glyphs.db format):
         [ [ ["moveTo",...], ["curveTo",...], ... ], [ ... ] ]
      5. Empty / missing / "[]": returns [].

    Improvements over prior version:
      - True quadratic TrueType sequences: when a qCurveTo payload contains >2 points,
        treat intermediate points as successive implicit on-curve points (TTF logic):
          * Given P0 (current point) and payload [P1, P2, ..., Pn]
            - For consecutive control points C1, C2:
                implied_on = (C1 + C2)/2 used as end of the quadratic with C1 as control.
            - Final point Pn is explicit on-curve.
      - Optional uniform subdivision parameter (fixed at 8 for now) still applied
        after resolving implied on-curve points.
      - curveTo treated as cubic alias (already handled).
      - Defensive filtering of degenerate subpaths (<3 points).

    Returns:
        List of polyline subpaths (each list of (x,y)).

    NOTE: This parser intentionally produces dense polylines (no duplicate collapse).
          Future optimization could add point reduction while preserving fidelity.
    """
    if not contours_json:
        return []
    cj = contours_json.strip()
    if cj in ("", "[]"):
        return []
    try:
        data = json.loads(cj)
    except json.JSONDecodeError:
        return []

    def is_point(pt) -> bool:
        return (
            isinstance(pt, (list, tuple))
            and len(pt) == 2
            and all(isinstance(v, (int, float)) for v in pt)
        )

    # Flat polyline
    if isinstance(data, list) and data and all(is_point(p) for p in data):
        poly = [(float(p[0]), float(p[1])) for p in data]
        return [poly] if len(poly) >= 3 else []

    # List of polylines
    if (
        isinstance(data, list)
        and data
        and all(
            isinstance(poly, (list, tuple)) and poly and all(is_point(p) for p in poly)
            for poly in data
        )
    ):
        polys = [
            [(float(p[0]), float(p[1])) for p in poly]
            for poly in data
            if len(poly) >= 3
        ]
        return polys

    # List of subpath command lists
    if (
        isinstance(data, list)
        and data
        and all(
            isinstance(sub, list)
            and sub
            and all(
                isinstance(cmd, (list, tuple))
                and len(cmd) == 2
                and isinstance(cmd[0], str)
                for cmd in sub
            )
            for sub in data
        )
    ):
        out: List[List[Tuple[float, float]]] = []
        for sub in data:
            out.extend(parse_contours(json.dumps(sub)))
        return [sp for sp in out if len(sp) >= 3]

    # Assume single command list
    subpaths: List[List[Tuple[float, float]]] = []
    current: List[Tuple[float, float]] = []

    def finish_current():
        nonlocal current
        if current and len(current) >= 3:
            subpaths.append(current)
        current = []

    for cmd in data:
        if not isinstance(cmd, (list, tuple)) or len(cmd) != 2:
            continue
        op, payload = cmd

        if op == "moveTo":
            finish_current()
            if is_point(payload):
                current.append((float(payload[0]), float(payload[1])))

        elif op == "lineTo":
            if is_point(payload):
                current.append((float(payload[0]), float(payload[1])))

        elif op == "qCurveTo":
            # TrueType quadratic handling: payload is a list of points (control/on-curve).
            if isinstance(payload, list) and current and payload:
                pts = [tuple(map(float, p)) for p in payload if is_point(p)]
                if not pts:
                    continue
                P0 = current[-1]
                # If only one point given, treat as simple quadratic via midpoint ctrl
                if len(pts) == 1:
                    end = pts[0]
                    ctrl = ((P0[0] + end[0]) * 0.5, (P0[1] + end[1]) * 0.5)
                    _append_quadratic_polyline(current, P0, ctrl, end)
                else:
                    # Multiple points: implicit on-curve between consecutive off-curve controls
                    controls = pts[:-1]
                    final_on = pts[-1]
                    expanded: List[Tuple[float, float]] = []
                    # Walk pairs
                    prev = P0
                    for i in range(len(controls) - 1):
                        c1 = controls[i]
                        c2 = controls[i + 1]
                        on_implied = ((c1[0] + c2[0]) * 0.5, (c1[1] + c2[1]) * 0.5)
                        _append_quadratic_polyline(current, prev, c1, on_implied)
                        prev = on_implied
                    # Last segment to final_on
                    _append_quadratic_polyline(current, prev, controls[-1], final_on)

        elif op in ("cubicTo", "curveTo"):
            if isinstance(payload, list) and len(payload) == 3 and current:
                try:
                    c1 = tuple(map(float, payload[0]))
                    c2 = tuple(map(float, payload[1]))
                    p1 = tuple(map(float, payload[2]))
                except Exception:
                    continue
                p0 = current[-1]
                subdivisions = 8
                for i in range(1, subdivisions + 1):
                    t = i / subdivisions
                    mt = 1 - t
                    x = (
                        mt**3 * p0[0]
                        + 3 * mt**2 * t * c1[0]
                        + 3 * mt * t**2 * c2[0]
                        + t**3 * p1[0]
                    )
                    y = (
                        mt**3 * p0[1]
                        + 3 * mt**2 * t * c1[1]
                        + 3 * mt * t**2 * c2[1]
                        + t**3 * p1[1]
                    )
                    current.append((x, y))

        elif op == "closePath":
            if current and current[0] != current[-1]:
                current.append(current[0])
            finish_current()

    finish_current()
    return [sp for sp in subpaths if len(sp) >= 3]


def _append_quadratic_polyline(
    store: List[Tuple[float, float]],
    p0: Tuple[float, float],
    ctrl: Tuple[float, float],
    p1: Tuple[float, float],
    subdivisions: int = 8,
) -> None:
    """
    Subdivide a quadratic Bezier into 'subdivisions' segments and append points (excluding p0).
    """
    for i in range(1, subdivisions + 1):
        t = i / subdivisions
        mt = 1 - t
        x = mt * mt * p0[0] + 2 * mt * t * ctrl[0] + t * t * p1[0]
        y = mt * mt * p0[1] + 2 * mt * t * ctrl[1] + t * t * p1[1]
        store.append((x, y))


# ---------------------------------------------------------------------------
# Geometry Utilities
# ---------------------------------------------------------------------------


def polygon_area(points: Sequence[Tuple[float, float]]) -> float:
    """Signed area via shoelace formula."""
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += x0 * y1 - x1 * y0
    return 0.5 * area


def compute_bbox(
    subpaths: List[List[Tuple[float, float]]],
) -> Tuple[float, float, float, float]:
    xs, ys = [], []
    for sp in subpaths:
        for x, y in sp:
            xs.append(x)
            ys.append(y)
    if not xs:
        return 0, 0, 0, 0
    return min(xs), min(ys), max(xs), max(ys)


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------


def rasterize_glyph_cairo(
    subpaths: List[List[Tuple[float, float]]],
    used_ascent: float,
    used_descent: float,
    cfg: FullConfig,
    advance_width: float | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Rasterize a glyph using Cairo for authoritative non-zero winding rule.

    Returns:
      bitmap: (H,W) uint8 0 or 255
      meta:   dict with fields required by Section 4.5
    """
    if not CAIRO_AVAILABLE:
        raise RuntimeError("Cairo is not available. Install with: pip install pycairo")

    if not subpaths:
        # Return blank canvas
        empty = np.zeros(
            (cfg.raster.canvas_size, cfg.raster.canvas_size), dtype=np.uint8
        )
        meta = {
            "is_diacritic": False,
            "original_bbox": [0, 0, 0, 0],
            "scale_factor": 0.0,
            "major_dim_ratio": 0.0,
            "target_dim": cfg.raster.main_target_dim,
            "engine": "cairo",
        }
        return empty, meta

    min_x, min_y, max_x, max_y = compute_bbox(subpaths)
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y
    major_dim = max(bbox_w, bbox_h)
    em_height = (
        (used_ascent - used_descent)
        if (used_ascent is not None and used_descent is not None)
        else bbox_h
    )
    em_height = em_height if em_height != 0 else 1.0
    ratio = major_dim / em_height

    # Hybrid diacritic rule: (advance_width < adv_threshold) OR (ratio < ratio_threshold)
    adv_threshold = getattr(cfg.raster, "diacritic_advance_threshold", 100)
    adv_ok = (advance_width is not None) and (advance_width < adv_threshold)
    is_diacritic = bool(adv_ok or (ratio < cfg.raster.diacritic_ratio_threshold))
    target_dim = (
        cfg.raster.diacritic_target_dim if is_diacritic else cfg.raster.main_target_dim
    )

    if major_dim == 0:
        scale = 1.0
    else:
        scale = target_dim / major_dim

    # Transform & scale points
    # Coordinate flip: font y upward -> image y downward
    transformed_subpaths: List[List[Tuple[float, float]]] = []
    for sp in subpaths:
        transformed = []
        for x, y in sp:
            sx = (x - min_x) * scale
            sy = (y - min_y) * scale
            transformed.append((sx, sy))
        transformed_subpaths.append(transformed)

    # After scaling, compute new bounds to flip y
    t_min_x, t_min_y, t_max_x, t_max_y = compute_bbox(transformed_subpaths)
    scaled_h = t_max_y - t_min_y if (t_max_y - t_min_y) != 0 else 1.0

    flipped_subpaths: List[List[Tuple[float, float]]] = []
    for sp in transformed_subpaths:
        flipped = []
        for x, y in sp:
            fy = scaled_h - (y - t_min_y)  # invert relative to min_y
            flipped.append((x, fy))
        flipped_subpaths.append(flipped)

    # Supersampled canvas
    canvas_size = cfg.raster.canvas_size
    ss = cfg.raster.supersample_factor
    render_size = canvas_size * ss

    # Create Cairo surface and context (FORMAT_A8 is sufficient for binary output)
    surface = cairo.ImageSurface(cairo.FORMAT_A8, render_size, render_size)
    ctx = cairo.Context(surface)

    # Set fill rule based on config
    fill_rule = getattr(cfg.raster, "fill_rule", "winding")
    if fill_rule == "even-odd":
        ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
    else:
        # Default to winding (non-zero) which is what we want
        ctx.set_fill_rule(cairo.FILL_RULE_WINDING)

    # Build path from subpaths
    for sp in flipped_subpaths:
        if len(sp) < 2:
            continue
        # Move to first point
        ctx.move_to(sp[0][0] * ss, sp[0][1] * ss)
        # Line to subsequent points
        for x, y in sp[1:]:
            ctx.line_to(x * ss, y * ss)
        # Close path
        ctx.close_path()

    # Fill the path with white
    ctx.set_source_rgb(1, 1, 1)  # White
    ctx.fill()

    # Get pixel data from Cairo surface
    buf = surface.get_data()
    img_arr = np.ndarray(shape=(render_size, render_size), dtype=np.uint8, buffer=buf)

    # Downsample to final size using high-quality LANCZOS resampling
    img = Image.fromarray(img_arr, mode="L")
    if ss > 1:
        img = img.resize((canvas_size, canvas_size), Image.LANCZOS)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    binary = (arr >= cfg.raster.binarize_threshold).astype(np.uint8) * 255

    meta = {
        "is_diacritic": bool(is_diacritic),
        "original_bbox": [float(min_x), float(min_y), float(max_x), float(max_y)],
        "scale_factor": float(scale),
        "major_dim_ratio": float(ratio),
        "target_dim": int(target_dim),
        "fill_rule_effective": fill_rule,
        "engine": "cairo",
    }
    return binary, meta


def rasterize_glyph(
    subpaths: List[List[Tuple[float, float]]],
    used_ascent: float,
    used_descent: float,
    cfg: FullConfig,
    advance_width: float | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Rasterize a glyph according to plan rules.

    Dispatches to Cairo or Python engine based on config.

    Returns:
      bitmap: (H,W) uint8 0 or 255
      meta:   dict with fields required by Section 4.5
    """
    engine = getattr(cfg.raster, "engine", "python")
    if engine == "cairo":
        return rasterize_glyph_cairo(
            subpaths, used_ascent, used_descent, cfg, advance_width=advance_width
        )
    else:
        return rasterize_glyph_python(
            subpaths, used_ascent, used_descent, cfg, advance_width=advance_width
        )


def rasterize_glyph_python(
    subpaths: List[List[Tuple[float, float]]],
    used_ascent: float,
    used_descent: float,
    cfg: FullConfig,
    advance_width: float | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Rasterize a glyph using Python/PIL implementation.

    Returns:
      bitmap: (H,W) uint8 0 or 255
      meta:   dict with fields required by Section 4.5
    """
    if not subpaths:
        # Return blank canvas
        empty = np.zeros(
            (cfg.raster.canvas_size, cfg.raster.canvas_size), dtype=np.uint8
        )
        meta = {
            "is_diacritic": False,
            "original_bbox": [0, 0, 0, 0],
            "scale_factor": 0.0,
            "major_dim_ratio": 0.0,
            "target_dim": cfg.raster.main_target_dim,
        }
        return empty, meta

    min_x, min_y, max_x, max_y = compute_bbox(subpaths)
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y
    major_dim = max(bbox_w, bbox_h)
    em_height = (
        (used_ascent - used_descent)
        if (used_ascent is not None and used_descent is not None)
        else bbox_h
    )
    em_height = em_height if em_height != 0 else 1.0
    ratio = major_dim / em_height

    # Hybrid diacritic rule: (advance_width < adv_threshold) OR (ratio < ratio_threshold)
    adv_threshold = getattr(cfg.raster, "diacritic_advance_threshold", 100)
    adv_ok = (advance_width is not None) and (advance_width < adv_threshold)
    is_diacritic = bool(adv_ok or (ratio < cfg.raster.diacritic_ratio_threshold))
    target_dim = (
        cfg.raster.diacritic_target_dim if is_diacritic else cfg.raster.main_target_dim
    )

    if major_dim == 0:
        scale = 1.0
    else:
        scale = target_dim / major_dim

    # Transform & scale points
    # Coordinate flip: font y upward -> image y downward. Use (y_max - y) after scaling for vertical flip.
    # We'll first translate so min_x, min_y -> (0,0), scale, then invert y by referencing scaled bbox height.
    transformed_subpaths: List[List[Tuple[float, float]]] = []
    for sp in subpaths:
        transformed = []
        for x, y in sp:
            sx = (x - min_x) * scale
            sy = (y - min_y) * scale
            transformed.append((sx, sy))
        transformed_subpaths.append(transformed)

    # After scaling, compute new bounds to flip y
    t_min_x, t_min_y, t_max_x, t_max_y = compute_bbox(transformed_subpaths)
    scaled_h = t_max_y - t_min_y if (t_max_y - t_min_y) != 0 else 1.0

    flipped_subpaths: List[List[Tuple[float, float]]] = []
    for sp in transformed_subpaths:
        flipped = []
        for x, y in sp:
            fy = scaled_h - (y - t_min_y)  # invert relative to min_y
            flipped.append((x, fy))
        flipped_subpaths.append(flipped)

    # Supersampled canvas
    canvas_size = cfg.raster.canvas_size
    ss = cfg.raster.supersample_factor
    render_size = canvas_size * ss

    # Create blank grayscale image
    img = Image.new("L", (render_size, render_size), color=0)
    draw = ImageDraw.Draw(img)

    # Fill rule handling (orientation / even-odd / winding) with optional debug outputs.
    #
    # Supported cfg.raster.fill_rule values:
    #   - "even-odd": parity fill (classic even–odd rule); robust for nested holes.
    #   - "orientation": dominant-orientation outer fill + opposite-sign hole carving (previous behavior).
    #   - "winding": non-zero winding accumulation (each positive area adds +1, negative area -1; fill where != 0).
    #
    # Debug (optional):
    #   If cfg.raster has attributes:
    #       debug_fill: bool
    #       debug_dir: str (existing directory)
    #   then we emit intermediate masks:
    #       parity_mask.png, winding_accum.npy / winding_mask.png, orientation_outer.png, orientation_holes.png
    #
    # NOTE: We do not mutate config; presence of attributes is probed dynamically to keep backward compatibility.
    areas: list[tuple[int, float]] = []
    for idx, sp in enumerate(flipped_subpaths):
        if len(sp) >= 3:
            areas.append((idx, polygon_area(sp)))

    # Filter non-zero-ish areas for sign detection
    nonzero_areas = [(i, a) for i, a in areas if abs(a) > 1e-6]
    if nonzero_areas:
        # Pick subpath with largest absolute area
        outer_index, outer_area = max(nonzero_areas, key=lambda t: abs(t[1]))
        outer_sign = 1 if outer_area >= 0 else -1
    else:
        # Degenerate: treat all as outer
        outer_sign = 1

    fill_rule = getattr(cfg.raster, "fill_rule", "orientation")
    debug_fill = bool(getattr(cfg.raster, "debug_fill", False))
    debug_dir = getattr(cfg.raster, "debug_dir", None)
    if debug_fill and debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    # Classify closed vs open (tolerant closure check)
    closed: list[tuple[list[tuple[float, float]], float]] = []
    open_segs: list[list[tuple[float, float]]] = []
    close_eps_sq = 1e-8
    for sp in flipped_subpaths:
        if len(sp) < 2:
            continue
        x0, y0 = sp[0]
        x1, y1 = sp[-1]
        if (x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1) <= close_eps_sq:
            if sp[-1] != sp[0]:
                sp = sp + [sp[0]]
            closed.append((sp, polygon_area(sp)))
        else:
            open_segs.append(sp)

    # Helper to rasterize a polygon into a mask (uint8 0/1)
    def _poly_mask(poly):
        tmp = Image.new("L", (render_size, render_size), color=0)
        ImageDraw.Draw(tmp).polygon([(x * ss, y * ss) for x, y in poly], fill=255)
        return (np.array(tmp, dtype=np.uint8) > 0).astype(np.uint8)

    if fill_rule == "even-odd":
        # Parity XOR logic using 1-bit intermediate masks
        from PIL import ImageChops

        parity_mask = Image.new("1", (render_size, render_size), color=0)
        for poly, _ in closed:
            tmp = Image.new("1", (render_size, render_size), color=0)
            ImageDraw.Draw(tmp).polygon([(x * ss, y * ss) for x, y in poly], fill=1)
            parity_mask = ImageChops.logical_xor(parity_mask, tmp)
        # Paste result
        img.paste(255, mask=parity_mask)
        if debug_fill and debug_dir:
            parity_mask.convert("L").save(os.path.join(debug_dir, "parity_mask.png"))

    elif fill_rule == "winding":
        # Non-zero winding accumulator
        accum = np.zeros((render_size, render_size), dtype=np.int16)
        for poly, area in closed:
            sign = 1 if area >= 0 else -1
            accum += sign * _poly_mask(poly)
        winding_mask_arr = (accum != 0).astype(np.uint8)
        img_arr = winding_mask_arr * 255
        effective_fill_rule = "winding"
        # Hybrid fallback for single self-intersecting contour producing unintended cancellations:
        # If only one closed contour AND winding removes a significant interior that parity would keep,
        # fall back to parity (even-odd) for that glyph.
        if len(closed) == 1:
            poly, _ = closed[0]
            parity_mask_arr = _poly_mask(poly)
            winding_filled = winding_mask_arr.sum()
            parity_filled = parity_mask_arr.sum()
            # If winding retains markedly less area (possible false exclusion) but parity keeps more, switch.
            if parity_filled > 0 and winding_filled < parity_filled * 0.75:
                img_arr = parity_mask_arr * 255
                effective_fill_rule = "winding->parity_fallback"
                if debug_fill and debug_dir:
                    Image.fromarray(parity_mask_arr * 255, mode="L").save(
                        os.path.join(debug_dir, "winding_parity_fallback.png")
                    )
        img = Image.fromarray(img_arr, mode="L")
        if debug_fill and debug_dir:
            np.save(os.path.join(debug_dir, "winding_accum.npy"), accum)
            Image.fromarray(winding_mask_arr * 255, mode="L").save(
                os.path.join(debug_dir, "winding_mask.png")
            )

    elif fill_rule == "orientation":
        # Determine dominant orientation by cumulative absolute area
        pos_sum = sum(abs(a) for (_, a) in closed if a > 0)
        neg_sum = sum(abs(a) for (_, a) in closed if a < 0)
        if closed:
            solid_orientation = 1 if pos_sum >= neg_sum else -1
        else:
            solid_orientation = 1
        # Fill solids
        for poly, area in closed:
            is_solid = (area >= 0 and solid_orientation == 1) or (
                area < 0 and solid_orientation == -1
            )
            if is_solid:
                ImageDraw.Draw(img).polygon(
                    [(x * ss, y * ss) for x, y in poly], fill=255
                )
        # Carve holes
        for poly, area in closed:
            is_hole = (area >= 0 and solid_orientation == -1) or (
                area < 0 and solid_orientation == 1
            )
            if is_hole:
                ImageDraw.Draw(img).polygon([(x * ss, y * ss) for x, y in poly], fill=0)
        if debug_fill and debug_dir:
            # Produce masks for solids vs holes
            solids_dbg = Image.new("L", (render_size, render_size), color=0)
            holes_dbg = Image.new("L", (render_size, render_size), color=0)
            sd = ImageDraw.Draw(solids_dbg)
            hd = ImageDraw.Draw(holes_dbg)
            for poly, area in closed:
                is_solid = (area >= 0 and solid_orientation == 1) or (
                    area < 0 and solid_orientation == -1
                )
                target = sd if is_solid else hd
                target.polygon([(x * ss, y * ss) for x, y in poly], fill=255)
            solids_dbg.save(os.path.join(debug_dir, "orientation_outer.png"))
            holes_dbg.save(os.path.join(debug_dir, "orientation_holes.png"))

    else:
        # Fallback: fill all closed outlines (no holes)
        for poly, _ in closed:
            ImageDraw.Draw(img).polygon([(x * ss, y * ss) for x, y in poly], fill=255)

    # Optional stroke rendering for open segments
    stroke_width = getattr(cfg.raster, "debug_stroke_width", 0)
    if stroke_width > 0 and open_segs:
        d2 = ImageDraw.Draw(img)
        for seg in open_segs:
            d2.line(
                [(x * ss, y * ss) for x, y in seg],
                fill=255,
                width=stroke_width,
                joint="curve",
            )

    # Downsample to final size
    if ss > 1:
        img = img.resize((canvas_size, canvas_size), Image.BICUBIC)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    binary = (arr >= cfg.raster.binarize_threshold).astype(np.uint8) * 255

    meta = {
        "is_diacritic": bool(is_diacritic),
        "original_bbox": [float(min_x), float(min_y), float(max_x), float(max_y)],
        "scale_factor": float(scale),
        "major_dim_ratio": float(ratio),
        "target_dim": int(target_dim),
        "fill_rule_effective": locals().get("effective_fill_rule", fill_rule),
        "engine": "python",
    }
    return binary, meta


# ---------------------------------------------------------------------------
# Grid Extraction (16x16 of 8x8 per plan)
# ---------------------------------------------------------------------------


def extract_cell_grid(bitmap: np.ndarray, cfg: FullConfig) -> np.ndarray:
    """
    Slice 128x128 binary bitmap into (rows, cols) cells each cell_px^2.
    Returns uint8 grid of occupancy (0/1).
    """
    rows, cols, cell_px = cfg.grid.rows, cfg.grid.cols, cfg.grid.cell_px
    expected_size = rows * cell_px
    if bitmap.shape[0] != expected_size or bitmap.shape[1] != expected_size:
        raise ValueError(
            f"Bitmap shape {bitmap.shape} does not match expected {expected_size}x{expected_size}"
        )

    grid = np.zeros((rows, cols), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            block = bitmap[
                r * cell_px : (r + 1) * cell_px, c * cell_px : (c + 1) * cell_px
            ]
            if np.any(block):
                grid[r, c] = 1
    return grid


# ---------------------------------------------------------------------------
# Primitive Vocabulary (K-Means) - Section 6.1
# ---------------------------------------------------------------------------


def sample_non_empty_cells(
    grids_dir: Path,
    rasters_dir: Path,
    cfg: FullConfig,
    max_cells: int,
    shuffle_seed: int,
) -> np.ndarray:
    """
    Iterate over existing occupancy grids (16x16), open corresponding raster PNG for cell content,
    and gather flattened 8x8 cell bitmaps for non-empty cells until max_cells.

    Returns: (N, 64) float32 array with values {0,1}
    """
    rng = random.Random(shuffle_seed)
    cell_vectors: List[np.ndarray] = []
    grid_paths = sorted(
        p for p in grids_dir.glob("*.npy") if not p.name.endswith("_ids.npy")
    )

    rng.shuffle(grid_paths)

    for gp in grid_paths:
        glyph_id = gp.stem
        try:
            grid = np.load(gp)
            if grid.shape != (cfg.grid.rows, cfg.grid.cols):
                continue
            raster_path = rasters_dir / f"{glyph_id}.png"
            if not raster_path.exists():
                continue
            img = Image.open(raster_path).convert("L")
            bitmap = (np.array(img) > 0).astype(np.uint8)  # assume binary PNG
        except Exception:
            continue

        # Extract cells
        for r in range(cfg.grid.rows):
            for c in range(cfg.grid.cols):
                if grid[r, c] == 0:
                    continue  # skip empty cells for clustering
                cell = bitmap[
                    r * cfg.grid.cell_px : (r + 1) * cfg.grid.cell_px,
                    c * cfg.grid.cell_px : (c + 1) * cfg.grid.cell_px,
                ]
                flat = cell.reshape(-1).astype(np.float32) / 255.0
                cell_vectors.append(flat)
                if len(cell_vectors) >= max_cells:
                    return np.vstack(cell_vectors)
    if not cell_vectors:
        return np.empty((0, 64), dtype=np.float32)
    return np.vstack(cell_vectors)


def run_kmeans(
    data: np.ndarray,
    k: int,
    seed: int,
    batch_size: int = 4096,
    max_iter: int = 100,
) -> np.ndarray:
    """
    Fit K-Means to data (N, 64).
    Returns centroids (k, 64).
    If scikit-learn is available, uses MiniBatchKMeans; else a naive implementation.
    """
    if data.shape[0] == 0:
        raise ValueError("No data provided for K-Means clustering.")

    if SKLEARN_AVAILABLE:
        mbk = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            batch_size=batch_size,
            max_iter=max_iter,
            n_init="auto",
            reassignment_ratio=0.01,
        )
        mbk.fit(data)
        return mbk.cluster_centers_.astype(np.float32)

    # Naive fallback (Lloyd's algorithm) - not optimized, only for small data; warns user.
    print(
        "[WARN] scikit-learn not available; using naive K-Means fallback (slow).",
        file=sys.stderr,
    )
    rng = np.random.default_rng(seed)
    indices = rng.choice(data.shape[0], size=k, replace=False)
    centroids = data[indices].copy()

    for iteration in range(max_iter):
        # Assign
        dists = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=2)  # (N,k)
        assign = np.argmin(dists, axis=1)
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(k, dtype=np.int64)
        for i, a in enumerate(assign):
            new_centroids[a] += data[i]
            counts[a] += 1
        for j in range(k):
            if counts[j] > 0:
                new_centroids[j] /= counts[j]
            else:
                # Reinitialize empty cluster
                new_centroids[j] = data[rng.integers(0, data.shape[0])]
        shift = np.linalg.norm(centroids - new_centroids)
        centroids = new_centroids
        if shift < 1e-4:
            break
    return centroids.astype(np.float32)


def assign_primitive_ids(
    centroids: np.ndarray,
    grids_dir: Path,
    rasters_dir: Path,
    cfg: FullConfig,
):
    """
    For each glyph's occupancy grid + raster (filenames keyed by DB primary key 'id'):
      - For each non-empty cell, find nearest centroid -> primitive ID = index + 1
      - Empty remains 0
    Persist as <id>_ids.npy (uint16, shape 16x16) where 'id' is the DB primary key (not font-local glyph_id).
    """
    centroid_norms = np.sum(centroids**2, axis=1, keepdims=True)  # (k,1)

    grid_paths = sorted(
        p for p in grids_dir.glob("*.npy") if not p.name.endswith("_ids.npy")
    )
    for gp in grid_paths:
        db_id = gp.stem  # DB primary key 'id'
        raster_path = rasters_dir / f"{db_id}.png"
        ids_path = grids_dir / f"{db_id}_ids.npy"
        if not raster_path.exists():
            continue
        try:
            grid = np.load(gp)
            img = Image.open(raster_path).convert("L")
            bitmap = (np.array(img) > 0).astype(np.uint8)
        except Exception:
            continue

        primitive_ids = np.zeros_like(grid, dtype=np.uint16)
        for r in range(cfg.grid.rows):
            for c in range(cfg.grid.cols):
                if grid[r, c] == 0:
                    continue
                cell = bitmap[
                    r * cfg.grid.cell_px : (r + 1) * cfg.grid.cell_px,
                    c * cfg.grid.cell_px : (c + 1) * cfg.grid.cell_px,
                ]
                vec = cell.reshape(1, -1).astype(np.float32) / 255.0  # (1,64)
                # Compute squared distances: ||v - c||^2 = ||c||^2 + ||v||^2 - 2 v c^T
                v_norm = np.sum(vec**2, axis=1, keepdims=True)  # (1,1)
                dists = centroid_norms.T + v_norm - 2.0 * (vec @ centroids.T)  # (1,k)
                idx = int(np.argmin(dists))
                primitive_ids[r, c] = idx + 1  # Reserve 0 for empty
        np.save(ids_path, primitive_ids, allow_pickle=False)


# ---------------------------------------------------------------------------
# Metadata Writer
# ---------------------------------------------------------------------------


def write_metadata_line(
    meta_path: Path, glyph_id: int, label: str, font_hash: str, extra: Dict[str, Any]
):
    record = {
        "glyph_id": glyph_id,
        "label": label,
        "font_hash": font_hash,
        **extra,
    }
    with open(meta_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Orchestration: Rasterization
# ---------------------------------------------------------------------------


def run_rasterization(cfg: FullConfig, limit: int | None = None):
    rng = random.Random(cfg.seed)
    np.random.seed(cfg.seed)

    cfg.paths.rasters_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.grids_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.metadata_path.parent.mkdir(parents=True, exist_ok=True)

    # Build whitelist
    rows_cache = list(iter_glyph_rows(cfg.paths.glyph_db))
    whitelist = build_label_whitelist(
        (r for r in rows_cache),
        cfg.label_filter.min_count,
        cfg.label_filter.drop_shaped_variants,
    )
    kept = [r for r in rows_cache if r["label"] in whitelist]

    print(f"[INFO] Total glyph rows: {len(rows_cache)} | After whitelist: {len(kept)}")

    start_time = time.time()
    processed = 0
    skipped_empty = 0  # Count glyphs with zero vector subpaths (blank / unusable)
    skipped_outlier = 0  # Count large diacritic outliers excluded from training

    for r in kept:
        # Use DB primary key 'id' for filenames; keep original glyph_id in metadata.
        orig_glyph_id = r["glyph_id"]
        db_id = r["id"]
        label = r["label"]
        contours_json = r.get("contours_json")
        used_ascent = r.get("used_ascent")
        used_descent = r.get("used_descent")
        font_hash = r.get("font_hash", "")
        char_class = r.get("char_class", "")
        advance_width = r.get("advance_width")

        subpaths = parse_contours(contours_json)
        if not subpaths:
            skipped_empty += 1
            continue

        # Hybrid diacritic detection and outlier filtering
        # First, compute ratio to determine if it's a diacritic
        bounds_str = r.get("bounds")
        if bounds_str:
            try:
                bounds = json.loads(bounds_str)
                if bounds and len(bounds) == 4:
                    min_x, min_y, max_x, max_y = bounds
                    bbox_w = max_x - min_x
                    bbox_h = max_y - min_y
                    major_dim = max(bbox_w, bbox_h)

                    em_height = (
                        (used_ascent - used_descent)
                        if (used_ascent and used_descent)
                        else bbox_h
                    )
                    em_height = em_height if em_height != 0 else 1.0
                    ratio = major_dim / em_height

                    # Hybrid rule: is_diacritic = (advance_width < 100) OR (ratio < 0.15)
                    adv = advance_width if advance_width is not None else 1000
                    is_diacritic_by_metrics = (adv < 100) or (ratio < 0.15)
                    is_diacritic_by_class = (
                        "diacritic" in char_class.lower() if char_class else False
                    )

                    # Exclude large diacritic outliers from training
                    # These are diacritics (by class) that are rendered unusually large
                    if is_diacritic_by_class and (adv >= 100 and ratio >= 0.15):
                        skipped_outlier += 1
                        continue
            except:
                pass  # If bounds parsing fails, proceed with rasterization

        bitmap, meta = rasterize_glyph(
            subpaths, used_ascent, used_descent, cfg, advance_width=advance_width
        )

        # Inject original glyph_id for traceability
        meta["orig_glyph_id"] = orig_glyph_id

        # Save raster named by DB primary key id
        import re  # local import for filename sanitization

        safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", label)[:64]
        raster_path = cfg.paths.rasters_dir / f"{safe_label}_{db_id}.png"
        Image.fromarray(bitmap, mode="L").save(
            raster_path,
            optimize=True,
            compress_level=cfg.raster.store_mode == "binary_uint8",
        )

        # Record raster filename for reverse lookup in metadata
        meta["raster_filename"] = raster_path.name

        # Extract & save occupancy grid (also keyed by DB id)
        grid = extract_cell_grid(bitmap, cfg)
        np.save(cfg.paths.grids_dir / f"{db_id}.npy", grid, allow_pickle=False)

        # Write metadata using db_id as glyph_id field, retaining orig_glyph_id inside meta
        write_metadata_line(cfg.paths.metadata_path, db_id, label, font_hash, meta)

        processed += 1
        if limit is not None and processed >= limit:
            print(f"[INFO] Reached glyph limit {limit}; stopping early.")
            break
        if processed % 1000 == 0:
            elapsed = time.time() - start_time
            print(
                f"[INFO] Processed {processed} glyphs (skipped empty: {skipped_empty}, outliers: {skipped_outlier}) in {elapsed:.1f}s"
            )

    print(
        f"[INFO] Completed rasterization of {processed} glyphs. Skipped empty: {skipped_empty}, outliers: {skipped_outlier}. Elapsed {time.time() - start_time:.1f}s"
    )


# ---------------------------------------------------------------------------
# Orchestration: Primitive Vocabulary
# ---------------------------------------------------------------------------


def run_primitive_kmeans(
    cfg: FullConfig, max_cells: int, output_centroids: Path | None
):
    if output_centroids is None:
        output_centroids = Path("assets/centroids/primitive_centroids.npy")
    output_centroids.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Sampling up to {max_cells} non-empty cells for K-Means...")
    cells = sample_non_empty_cells(
        grids_dir=cfg.paths.grids_dir,
        rasters_dir=cfg.paths.rasters_dir,
        cfg=cfg,
        max_cells=max_cells,
        shuffle_seed=cfg.seed,
    )
    print(f"[INFO] Sampled {cells.shape[0]} cells (each 64D).")

    if cells.shape[0] == 0:
        print("[ERROR] No non-empty cells available for clustering.")
        return

    k = 1023  # 1..1023; 0 is reserved for empty
    print(f"[INFO] Running K-Means with k={k} ...")
    centroids = run_kmeans(cells, k=k, seed=cfg.seed)
    np.save(output_centroids, centroids.astype(np.float32), allow_pickle=False)
    print(f"[INFO] Saved centroids to {output_centroids}")

    print("[INFO] Assigning primitive IDs per glyph grid...")
    assign_primitive_ids(centroids, cfg.paths.grids_dir, cfg.paths.rasters_dir, cfg)
    print("[INFO] Primitive ID assignment complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rasterization & Primitive Vocabulary Pipeline (Plan-Aligned)"
    )
    p.add_argument(
        "command", choices=["rasterize", "kmeans", "full"], help="Pipeline stage to run"
    )
    p.add_argument("--config", required=True, help="Path to rasterizer.yaml")
    p.add_argument(
        "--limit-glyphs",
        type=int,
        default=None,
        help="Process only the first N glyphs during rasterization (debug / quick run)",
    )
    p.add_argument(
        "--max-cells",
        type=int,
        default=1_000_000,
        help="Max non-empty cells to sample for K-Means",
    )
    p.add_argument(
        "--centroids-out", type=str, default=None, help="Override centroids output path"
    )
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Apply seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    if args.command in ("rasterize", "full"):
        run_rasterization(cfg, limit=args.limit_glyphs)

    if args.command in ("kmeans", "full"):
        centroids_path = Path(args.centroids_out) if args.centroids_out else None
        run_primitive_kmeans(
            cfg, max_cells=args.max_cells, output_centroids=centroids_path
        )


if __name__ == "__main__":
    main()
