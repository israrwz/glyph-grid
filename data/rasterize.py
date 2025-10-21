#!/usr/bin/env python3
"""
Rasterization with Stratified Splits & Multiprocessing

Generates 128x128 binary glyph rasters organized into train/val/test splits
with stratification by label class.

Features:
- Full CPU multiprocessing utilization
- Stratified splits: 86% train, 10% val, 4% test (within each label)
- Filename format: {label}_{row_id}.png (e.g., 65_latin_11356.png)
- Output structure:
    dataset/rasters/train/*.png
    dataset/rasters/val/*.png
    dataset/rasters/test/*.png

Dependencies: pillow, numpy, pyyaml

Usage:
    python data/rasterize.py --config configs/rasterizer.yaml

    # Override workers (default: all CPUs)
    RASTER_WORKERS=8 python data/rasterize.py --config configs/rasterizer.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

try:
    import yaml
except ImportError:
    raise SystemExit("Missing pyyaml: pip install pyyaml")

try:
    import cairo

    CAIRO_AVAILABLE = True
except ImportError:
    CAIRO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
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
    fill_rule: str
    store_mode: str


@dataclass
class PathsConfig:
    glyph_db: Path
    rasters_dir: Path
    metadata_path: Path


@dataclass
class SplitConfig:
    train_ratio: float
    val_ratio: float
    test_ratio: float


@dataclass
class FullConfig:
    seed: int
    raster: RasterConfig
    paths: PathsConfig
    split: SplitConfig


def load_config(path: Path) -> FullConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    sources = raw.get("sources", {})
    outputs = raw.get("outputs", {})
    raster = raw.get("raster", {})
    split = raw.get(
        "split", {"train_ratio": 0.86, "val_ratio": 0.10, "test_ratio": 0.04}
    )

    return FullConfig(
        seed=int(raw.get("seed", 42)),
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
            fill_rule=str(raster.get("fill_rule", "orientation")),
            store_mode=str(raster.get("store_mode", "binary_uint8")),
        ),
        paths=PathsConfig(
            glyph_db=Path(sources["glyph_db"]),
            rasters_dir=Path(outputs["rasters_dir"]),
            metadata_path=Path(
                outputs.get("metadata_path", "dataset/rasters/metadata.jsonl")
            ),
        ),
        split=SplitConfig(
            train_ratio=float(split["train_ratio"]),
            val_ratio=float(split["val_ratio"]),
            test_ratio=float(split["test_ratio"]),
        ),
    )


# ---------------------------------------------------------------------------
# Database Access
# ---------------------------------------------------------------------------


def load_glyphs(db_path: Path) -> List[Dict[str, Any]]:
    """Load all valid glyphs from database."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute("""
            SELECT
              g.id AS row_id,
              g.glyph_id,
              g.label,
              g.contours AS contours_json,
              g.char_class,
              g.advance_width,
              COALESCE(f.typo_ascent, f.ascent) AS used_ascent,
              COALESCE(f.typo_descent, f.descent) AS used_descent,
              f.file_hash AS font_hash,
              g.bounds,
              g.width,
              g.height
            FROM glyphs g
            JOIN fonts f ON f.file_hash = g.f_id
            WHERE g.contours IS NOT NULL
              AND g.has_contours != 0
              AND g.label NOT LIKE '%_shaped'
            ORDER BY g.id
        """)
        return [dict(row) for row in cur]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Contour Parsing
# ---------------------------------------------------------------------------


def parse_contours(contours_json: str) -> List[List[Tuple[float, float]]]:
    """Parse contours JSON into polyline subpaths."""
    if not contours_json or contours_json.strip() in ("", "[]"):
        return []

    try:
        data = json.loads(contours_json)
    except json.JSONDecodeError:
        return []

    def is_point(pt):
        return (
            isinstance(pt, (list, tuple))
            and len(pt) == 2
            and all(isinstance(v, (int, float)) for v in pt)
        )

    # Already flat polyline
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
        return [
            [(float(p[0]), float(p[1])) for p in poly]
            for poly in data
            if len(poly) >= 3
        ]

    # Check if this is a list of command-based contours (nested list structure)
    def is_command(cmd):
        return (
            isinstance(cmd, (list, tuple)) and len(cmd) == 2 and isinstance(cmd[0], str)
        )

    if (
        isinstance(data, list)
        and data
        and all(
            isinstance(contour, (list, tuple))
            and contour
            and all(is_command(cmd) for cmd in contour)
            for contour in data
        )
    ):
        # Process each contour separately
        all_subpaths = []
        for contour_cmds in data:
            subpaths = []
            current = []

            def finish_current():
                nonlocal current
                if current and len(current) >= 3:
                    subpaths.append(current)
                current = []

            for cmd in contour_cmds:
                op, payload = cmd

                if op == "moveTo":
                    finish_current()
                    if is_point(payload):
                        current.append((float(payload[0]), float(payload[1])))

                elif op == "lineTo":
                    if is_point(payload):
                        current.append((float(payload[0]), float(payload[1])))

                elif op == "qCurveTo":
                    if isinstance(payload, list) and current and payload:
                        pts = [tuple(map(float, p)) for p in payload if is_point(p)]
                        if not pts:
                            continue
                        p0 = current[-1]
                        if len(pts) == 1:
                            end = pts[0]
                            ctrl = ((p0[0] + end[0]) * 0.5, (p0[1] + end[1]) * 0.5)
                            for i in range(1, 9):
                                t = i / 8
                                mt = 1 - t
                                x = (
                                    mt * mt * p0[0]
                                    + 2 * mt * t * ctrl[0]
                                    + t * t * end[0]
                                )
                                y = (
                                    mt * mt * p0[1]
                                    + 2 * mt * t * ctrl[1]
                                    + t * t * end[1]
                                )
                                current.append((x, y))
                        else:
                            controls = pts[:-1]
                            final_on = pts[-1]
                            prev = p0
                            for i in range(len(controls) - 1):
                                c1 = controls[i]
                                c2 = controls[i + 1]
                                on_implied = (
                                    (c1[0] + c2[0]) * 0.5,
                                    (c1[1] + c2[1]) * 0.5,
                                )
                                for j in range(1, 9):
                                    t = j / 8
                                    mt = 1 - t
                                    x = (
                                        mt * mt * prev[0]
                                        + 2 * mt * t * c1[0]
                                        + t * t * on_implied[0]
                                    )
                                    y = (
                                        mt * mt * prev[1]
                                        + 2 * mt * t * c1[1]
                                        + t * t * on_implied[1]
                                    )
                                    current.append((x, y))
                                prev = on_implied
                            for j in range(1, 9):
                                t = j / 8
                                mt = 1 - t
                                x = (
                                    mt * mt * prev[0]
                                    + 2 * mt * t * controls[-1][0]
                                    + t * t * final_on[0]
                                )
                                y = (
                                    mt * mt * prev[1]
                                    + 2 * mt * t * controls[-1][1]
                                    + t * t * final_on[1]
                                )
                                current.append((x, y))

                elif op in ("cubicTo", "curveTo"):
                    if isinstance(payload, list) and len(payload) == 3 and current:
                        try:
                            c1 = tuple(map(float, payload[0]))
                            c2 = tuple(map(float, payload[1]))
                            p1 = tuple(map(float, payload[2]))
                            p0 = current[-1]
                            for i in range(1, 9):
                                t = i / 8
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
                        except Exception:
                            pass

                elif op == "closePath":
                    if current and current[0] != current[-1]:
                        current.append(current[0])
                    finish_current()

            finish_current()
            all_subpaths.extend(subpaths)

        return [sp for sp in all_subpaths if len(sp) >= 3]

    # Command-based paths (flat list of commands)
    subpaths = []
    current = []

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
            if isinstance(payload, list) and current and payload:
                pts = [tuple(map(float, p)) for p in payload if is_point(p)]
                if not pts:
                    continue
                p0 = current[-1]
                if len(pts) == 1:
                    end = pts[0]
                    ctrl = ((p0[0] + end[0]) * 0.5, (p0[1] + end[1]) * 0.5)
                    for i in range(1, 9):
                        t = i / 8
                        mt = 1 - t
                        x = mt * mt * p0[0] + 2 * mt * t * ctrl[0] + t * t * end[0]
                        y = mt * mt * p0[1] + 2 * mt * t * ctrl[1] + t * t * end[1]
                        current.append((x, y))
                else:
                    controls = pts[:-1]
                    final_on = pts[-1]
                    prev = p0
                    for i in range(len(controls) - 1):
                        c1 = controls[i]
                        c2 = controls[i + 1]
                        on_implied = ((c1[0] + c2[0]) * 0.5, (c1[1] + c2[1]) * 0.5)
                        for j in range(1, 9):
                            t = j / 8
                            mt = 1 - t
                            x = (
                                mt * mt * prev[0]
                                + 2 * mt * t * c1[0]
                                + t * t * on_implied[0]
                            )
                            y = (
                                mt * mt * prev[1]
                                + 2 * mt * t * c1[1]
                                + t * t * on_implied[1]
                            )
                            current.append((x, y))
                        prev = on_implied
                    for j in range(1, 9):
                        t = j / 8
                        mt = 1 - t
                        x = (
                            mt * mt * prev[0]
                            + 2 * mt * t * controls[-1][0]
                            + t * t * final_on[0]
                        )
                        y = (
                            mt * mt * prev[1]
                            + 2 * mt * t * controls[-1][1]
                            + t * t * final_on[1]
                        )
                        current.append((x, y))

        elif op in ("cubicTo", "curveTo"):
            if isinstance(payload, list) and len(payload) == 3 and current:
                try:
                    c1 = tuple(map(float, payload[0]))
                    c2 = tuple(map(float, payload[1]))
                    p1 = tuple(map(float, payload[2]))
                    p0 = current[-1]
                    for i in range(1, 9):
                        t = i / 8
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
                except Exception:
                    pass

        elif op == "closePath":
            if current and current[0] != current[-1]:
                current.append(current[0])
            finish_current()

    finish_current()
    return [sp for sp in subpaths if len(sp) >= 3]


def polygon_area(poly: List[Tuple[float, float]]) -> float:
    """Compute signed area (positive = CCW)."""
    area = 0.0
    for i in range(len(poly)):
        j = (i + 1) % len(poly)
        area += poly[i][0] * poly[j][1]
        area -= poly[j][0] * poly[i][1]
    return area * 0.5


def compute_bbox(
    subpaths: List[List[Tuple[float, float]]],
) -> Tuple[float, float, float, float]:
    """Compute bounding box of all subpaths."""
    all_pts = [pt for poly in subpaths for pt in poly]
    if not all_pts:
        return (0, 0, 0, 0)
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    return (min(xs), min(ys), max(xs), max(ys))


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------


def rasterize_glyph_cairo(
    subpaths: List[List[Tuple[float, float]]],
    used_ascent: float,
    used_descent: float,
    cfg: RasterConfig,
    advance_width: float = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Rasterize glyph using Cairo for authoritative non-zero winding rule."""
    if not CAIRO_AVAILABLE:
        raise RuntimeError("Cairo is not available. Install with: pip install pycairo")

    if not subpaths:
        return np.zeros((cfg.canvas_size, cfg.canvas_size), dtype=np.uint8), {}

    min_x, min_y, max_x, max_y = compute_bbox(subpaths)
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y

    if bbox_w <= 0 or bbox_h <= 0:
        return np.zeros((cfg.canvas_size, cfg.canvas_size), dtype=np.uint8), {}

    major_dim = max(bbox_w, bbox_h)
    em_height = (
        (used_ascent - used_descent) if (used_ascent and used_descent) else bbox_h
    )
    em_height = em_height if em_height != 0 else 1.0
    ratio = major_dim / em_height

    # Two-stage diacritic detection
    # Stage 1: Primary heuristic (strict thresholds)
    adv = advance_width if advance_width is not None else 1000
    is_diacritic = (adv < cfg.diacritic_advance_threshold) or (
        ratio < cfg.diacritic_ratio_threshold
    )

    # Stage 2: Secondary heuristic for edge cases (relaxed ratio + vertical position)
    # If not detected by stage 1, check if it's positioned outside normal text range
    if not is_diacritic and ratio < (cfg.diacritic_ratio_threshold + 0.1):
        # Check if positioned significantly above or below normal text range
        if used_ascent and used_descent:
            positioned_above = min_y > used_ascent * 0.5
            positioned_below = max_y < used_descent * 0.5
            is_diacritic = positioned_above or positioned_below

    target_dim = cfg.diacritic_target_dim if is_diacritic else cfg.main_target_dim

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
    canvas_size = cfg.canvas_size
    ss = cfg.supersample_factor
    render_size = canvas_size * ss

    # Create Cairo surface and context
    surface = cairo.ImageSurface(cairo.FORMAT_A8, render_size, render_size)
    ctx = cairo.Context(surface)

    # Set fill rule to non-zero winding (default, but explicit)
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

    # Fill the path
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
    binary = (arr >= cfg.binarize_threshold).astype(np.uint8) * 255

    meta = {
        "bbox": [float(min_x), float(min_y), float(max_x), float(max_y)],
        "scale": float(scale),
        "is_diacritic": is_diacritic,
        "target_dim": target_dim,
    }

    return binary, meta


def rasterize_glyph_python(
    subpaths: List[List[Tuple[float, float]]],
    used_ascent: float,
    used_descent: float,
    cfg: RasterConfig,
    advance_width: float = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Rasterize glyph using Python/PIL with area-based hole detection."""
    if not subpaths:
        return np.zeros((cfg.canvas_size, cfg.canvas_size), dtype=np.uint8), {}

    min_x, min_y, max_x, max_y = compute_bbox(subpaths)
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y

    if bbox_w <= 0 or bbox_h <= 0:
        return np.zeros((cfg.canvas_size, cfg.canvas_size), dtype=np.uint8), {}

    major_dim = max(bbox_w, bbox_h)
    em_height = (
        (used_ascent - used_descent) if (used_ascent and used_descent) else bbox_h
    )
    em_height = em_height if em_height != 0 else 1.0
    ratio = major_dim / em_height

    # Two-stage diacritic detection
    # Stage 1: Primary heuristic (strict thresholds)
    adv = advance_width if advance_width is not None else 1000
    is_diacritic = (adv < cfg.diacritic_advance_threshold) or (
        ratio < cfg.diacritic_ratio_threshold
    )

    # Stage 2: Secondary heuristic for edge cases (relaxed ratio + vertical position)
    # If not detected by stage 1, check if it's positioned outside normal text range
    if not is_diacritic and ratio < (cfg.diacritic_ratio_threshold + 0.1):
        # Check if positioned significantly above or below normal text range
        if used_ascent and used_descent:
            positioned_above = min_y > used_ascent * 0.5
            positioned_below = max_y < used_descent * 0.5
            is_diacritic = positioned_above or positioned_below

    target_dim = cfg.diacritic_target_dim if is_diacritic else cfg.main_target_dim

    if major_dim == 0:
        scale = 1.0
    else:
        scale = target_dim / major_dim

    # Transform & scale points
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

    # Supersample canvas
    ss = cfg.supersample_factor
    canvas_ss = cfg.canvas_size * ss

    # Render with PIL
    img = Image.new("L", (canvas_ss, canvas_ss), 0)
    draw = ImageDraw.Draw(img)

    # Sort by area for proper fill (outer positive, holes negative)
    sorted_paths = sorted(
        flipped_subpaths, key=lambda p: abs(polygon_area(p)), reverse=True
    )

    for poly in sorted_paths:
        area = polygon_area(poly)
        scaled_poly = [(x * ss, y * ss) for x, y in poly]
        if area > 0:
            draw.polygon(scaled_poly, fill=255, outline=255)
        else:
            # Negative area = hole, carve out
            draw.polygon(scaled_poly, fill=0, outline=0)

    # Downsample
    img = img.resize((cfg.canvas_size, cfg.canvas_size), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0

    # Binarize
    binary = (arr >= cfg.binarize_threshold).astype(np.uint8) * 255

    meta = {
        "bbox": [float(min_x), float(min_y), float(max_x), float(max_y)],
        "scale": float(scale),
        "is_diacritic": is_diacritic,
        "target_dim": target_dim,
    }

    return binary, meta


def rasterize_glyph(
    subpaths: List[List[Tuple[float, float]]],
    used_ascent: float,
    used_descent: float,
    cfg: RasterConfig,
    advance_width: float = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Rasterize glyph to 128x128 binary bitmap.

    Dispatches to Cairo (if available and configured) or Python/PIL fallback.
    """
    engine = getattr(cfg, "engine", "cairo")

    if engine == "cairo" and CAIRO_AVAILABLE:
        return rasterize_glyph_cairo(
            subpaths, used_ascent, used_descent, cfg, advance_width
        )
    else:
        if engine == "cairo" and not CAIRO_AVAILABLE:
            print(
                "[WARNING] Cairo requested but not available, falling back to Python/PIL"
            )
        return rasterize_glyph_python(
            subpaths, used_ascent, used_descent, cfg, advance_width
        )


# ---------------------------------------------------------------------------
# Stratified Splitting
# ---------------------------------------------------------------------------


def create_stratified_splits(
    rows: List[Dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split data by label with stratification."""
    random.seed(seed)

    # Group by label
    by_label = defaultdict(list)
    for row in rows:
        label = row["label"]
        by_label[label].append(row)

    train, val, test = [], [], []

    for label, items in by_label.items():
        n = len(items)
        if n == 0:
            continue

        # Shuffle within label
        shuffled = items.copy()
        random.shuffle(shuffled)

        # Calculate splits
        # Train gets its share first (guaranteed at least 1)
        n_train = max(1, int(n * train_ratio))
        remainder = n - n_train

        # Split remainder between val and test proportionally
        if remainder > 0:
            # Normalize val and test ratios for the remainder
            val_test_total = val_ratio + test_ratio
            if val_test_total > 0:
                val_proportion = val_ratio / val_test_total
                n_val = int(remainder * val_proportion)
                n_test = remainder - n_val
            else:
                n_val = remainder
                n_test = 0
        else:
            n_val = 0
            n_test = 0

        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train : n_train + n_val])
        test.extend(shuffled[n_train + n_val :])

    return train, val, test


# ---------------------------------------------------------------------------
# Worker Function (Picklable)
# ---------------------------------------------------------------------------


def raster_worker(
    args: Tuple[Dict[str, Any], RasterConfig, Path, str],
) -> Tuple[int, int]:
    """Multiprocessing worker to rasterize a single glyph."""
    row, raster_cfg, base_dir, split = args

    try:
        row_id = row["row_id"]
        label = row["label"]
        contours_json = row.get("contours_json")
        used_ascent = row.get("used_ascent")
        used_descent = row.get("used_descent")
        advance_width = row.get("advance_width")

        # Parse contours
        subpaths = parse_contours(contours_json)
        if not subpaths:
            return (0, 1)

        # Rasterize
        bitmap, meta = rasterize_glyph(
            subpaths, used_ascent, used_descent, raster_cfg, advance_width
        )

        # Filename: {label}_{row_id}.png
        safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", label)[:64]
        filename = f"{safe_label}_{row_id}.png"

        # Save to appropriate split directory
        split_dir = base_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        out_path = split_dir / filename

        Image.fromarray(bitmap, mode="L").save(out_path, optimize=True)

        return (1, 0)

    except Exception as e:
        return (0, 0)


# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------


def run_rasterization(cfg: FullConfig, limit: int = None):
    """Run rasterization with multiprocessing."""
    start_time = time.time()

    # Load data
    print("[INFO] Loading glyphs from database...")
    rows = load_glyphs(cfg.paths.glyph_db)

    if limit:
        rows = rows[:limit]

    print(f"[INFO] Loaded {len(rows)} glyphs")

    # Create stratified splits
    print("[INFO] Creating stratified splits...")
    train, val, test = create_stratified_splits(
        rows,
        cfg.split.train_ratio,
        cfg.split.val_ratio,
        cfg.split.test_ratio,
        cfg.seed,
    )

    print(f"[INFO] Splits: train={len(train)}, val={len(val)}, test={len(test)}")

    # Create output directories
    base_dir = cfg.paths.rasters_dir
    for split in ["train", "val", "test"]:
        (base_dir / split).mkdir(parents=True, exist_ok=True)

    # Prepare work items
    work_items = []
    for row in train:
        work_items.append((row, cfg.raster, base_dir, "train"))
    for row in val:
        work_items.append((row, cfg.raster, base_dir, "val"))
    for row in test:
        work_items.append((row, cfg.raster, base_dir, "test"))

    total = len(work_items)

    # Determine worker count
    workers_env = os.environ.get("RASTER_WORKERS")
    if workers_env:
        workers = int(workers_env)
    else:
        workers = cpu_count()

    print(f"[INFO] Using {workers} workers (set RASTER_WORKERS to override)")

    # Process with multiprocessing
    processed = 0
    skipped = 0

    with Pool(processes=workers) as pool:
        for i, (p, s) in enumerate(
            pool.imap_unordered(raster_worker, work_items, chunksize=32), 1
        ):
            processed += p
            skipped += s

            if i % 2000 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                print(
                    f"[INFO] Progress: {i}/{total} ({processed} OK, {skipped} skipped) - {rate:.1f} glyphs/s"
                )

    elapsed = time.time() - start_time
    rate = processed / elapsed if elapsed > 0 else 0

    print(f"\n[INFO] Completed rasterization:")
    print(f"  Total processed: {processed}/{total}")
    print(f"  Skipped (empty): {skipped}")
    print(f"  Time: {elapsed:.1f}s ({rate:.2f} glyphs/s)")
    print(f"  Output: {cfg.paths.rasters_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Rasterize glyphs with stratified splits"
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to rasterizer.yaml"
    )
    parser.add_argument(
        "--limit", type=int, help="Limit number of glyphs (for testing)"
    )

    args = parser.parse_args()

    cfg = load_config(args.config)
    run_rasterization(cfg, limit=args.limit)


if __name__ == "__main__":
    main()
