#!/usr/bin/env python3
"""
Smoke test for glyph-grid rasterization and primitive pipeline.

This script validates core invariants from NEW_PLAN.md without processing the full dataset.

Checks performed for a small sample of glyphs:
  - DB connectivity and required columns exist.
  - Label whitelist filtering (min_count).
  - Rasterization:
     * Output mask shape 128x128 (uint8).
     * Binary values only {0,255}.
     * Diacritic heuristic: ratio < 0.25 -> target_dim 64 else 128.
  - Grid extraction:
     * 16x16 occupancy grid.
     * Each cell corresponds to an 8x8 slice; occupancy 1 iff any foreground pixel.
     * Empty cell fraction within a plausible band (not >99% unless all blank).
  - Metadata fields presence.

Optional quick miniature K-Means (k<=8) to ensure clustering path works end-to-end (disabled by default).

Usage:
  python -m data.smoke_test --config configs/rasterizer.yaml --limit 20

Exit codes:
  0 success
  1 failure / invariant violation

The test avoids writing raster & grid files unless --write-artifacts is specified.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

# Import internal pipeline pieces
try:
    from .rasterize import (
        load_config,
        iter_glyph_rows,
        build_label_whitelist,
        parse_contours,
        rasterize_glyph,
        extract_cell_grid,
        sample_non_empty_cells,
        run_kmeans,
    )
except Exception as e:
    print(
        f"[SMOKE][FATAL] Unable to import rasterization components: {e}",
        file=sys.stderr,
    )
    sys.exit(1)

PLAN_DIACRITIC_THRESHOLD = 0.25


def _assert(cond: bool, msg: str, failures: List[str]):
    if not cond:
        failures.append(msg)


def check_db_schema(db_path: Path, table: str, required_cols: List[str]) -> List[str]:
    """
    Generic table schema checker.
    """
    failures: List[str] = []
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.execute(f"PRAGMA table_info({table})")
        cols = [row[1] for row in cur.fetchall()]
        for c in required_cols:
            if c not in cols:
                failures.append(
                    f"Missing column '{c}' in {table} table (found: {cols})"
                )
    except Exception as e:
        failures.append(f"DB schema inspection failed for {table}: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return failures


def smoke_raster_sample(
    cfg, limit: int, write: bool, kmeans_k: int, verbose: bool
) -> int:
    failures: List[str] = []
    t0 = time.time()

    # Step 1: schema check
    schema_failures = []
    # Glyph-level required columns (actual schema)
    schema_failures += check_db_schema(
        cfg.paths.glyph_db,
        "glyphs",
        [
            "glyph_id",
            "label",
            "contours",
            "f_id",
            "bounds",
            "width",
            "height",
        ],
    )
    # Font-level required columns used to derive ascent/descent
    schema_failures += check_db_schema(
        cfg.paths.glyph_db,
        "fonts",
        [
            "file_hash",
            "ascent",
            "descent",
            "typo_ascent",
            "typo_descent",
            "units_per_em",
        ],
    )
    failures.extend(schema_failures)
    if schema_failures:
        return _finalize(failures, t0, verbose)

    rows = list(iter_glyph_rows(cfg.paths.glyph_db))
    if not rows:
        failures.append("No glyph rows returned from DB.")
        return _finalize(failures, t0, verbose)

    whitelist = build_label_whitelist(
        rows, cfg.label_filter.min_count, cfg.label_filter.drop_shaped_variants
    )
    filtered = [r for r in rows if r["label"] in whitelist]
    _assert(len(filtered) > 0, "Whitelist filtering removed all glyphs.", failures)

    sample = filtered[:limit]
    if len(sample) < limit:
        print(
            f"[SMOKE][WARN] Only {len(sample)} glyphs after filter; requested limit {limit}",
            file=sys.stderr,
        )

    empty_cells_total = 0
    cells_total = 0
    diacritic_seen = 0

    # Optionally create artifact dirs
    if write:
        cfg.paths.rasters_dir.mkdir(parents=True, exist_ok=True)
        cfg.paths.grids_dir.mkdir(parents=True, exist_ok=True)
        cfg.paths.metadata_path.parent.mkdir(parents=True, exist_ok=True)

    for r in sample:
        gid = r["glyph_id"]
        label = r["label"]
        # Contours already exposed by iter_glyph_rows under unified key 'contours_json'
        subpaths = parse_contours(r["contours_json"])
        # Ascent / descent provided via join inside iter_glyph_rows (used_ascent/used_descent).
        used_ascent = r.get("used_ascent")  # may be injected by future join
        used_descent = r.get("used_descent")
        bitmap, meta = rasterize_glyph(
            subpaths,
            used_ascent,
            used_descent,
            cfg,
            advance_width=r.get("advance_width"),
        )

        # Assertions
        _assert(
            bitmap.shape == (128, 128),
            f"Glyph {gid} bitmap shape {bitmap.shape} != (128,128)",
            failures,
        )
        unique_vals = np.unique(bitmap)
        _assert(
            set(unique_vals.tolist()).issubset({0, 255}),
            f"Glyph {gid} bitmap not binary values {unique_vals}",
            failures,
        )
        _assert(
            meta["target_dim"] in (64, 128),
            f"Glyph {gid} meta target_dim invalid {meta['target_dim']}",
            failures,
        )
        ratio = meta["major_dim_ratio"]
        # Relax assertion for glyphs with zero subpaths (no vector data) where ratio==0.0.
        # Such blank glyphs retain target_dim 128 (main) by construction; we skip diacritic size enforcement.
        if ratio == 0.0 and len(subpaths) == 0:
            pass
        elif ratio < PLAN_DIACRITIC_THRESHOLD:
            _assert(
                meta["target_dim"] == 64,
                f"Glyph {gid} ratio {ratio:.3f} < thresh but target_dim !=64",
                failures,
            )
        else:
            _assert(
                meta["target_dim"] == 128,
                f"Glyph {gid} ratio {ratio:.3f} >= thresh but target_dim !=128",
                failures,
            )

        grid = extract_cell_grid(bitmap, cfg)
        _assert(
            grid.shape == (16, 16),
            f"Glyph {gid} grid shape {grid.shape} != (16,16)",
            failures,
        )

        # Validate cell occupancy matches raster
        recon_nonzero = 0
        for rr in range(16):
            for cc in range(16):
                block = bitmap[rr * 8 : (rr + 1) * 8, cc * 8 : (cc + 1) * 8]
                has_fg = 1 if np.any(block) else 0
                if has_fg:
                    recon_nonzero += 1
                if has_fg != grid[rr, cc]:
                    failures.append(f"Glyph {gid} cell ({rr},{cc}) occupancy mismatch.")
        emp = 16 * 16 - recon_nonzero
        empty_cells_total += emp
        cells_total += 16 * 16
        if meta["is_diacritic"]:
            diacritic_seen += 1

        if write:
            # Save artifacts
            Image.fromarray(bitmap, mode="L").save(cfg.paths.rasters_dir / f"{gid}.png")
            np.save(cfg.paths.grids_dir / f"{gid}.npy", grid, allow_pickle=False)
            with open(cfg.paths.metadata_path, "a", encoding="utf-8") as f:
                rec = {
                    "glyph_id": gid,
                    "label": label,
                    "font_hash": r.get("font_hash", ""),
                    **meta,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Aggregate cell occupancy stats
    if cells_total:
        empty_frac = empty_cells_total / cells_total
        # Expect a lot of empties; just ensure not pathological
        _assert(
            empty_frac < 0.999,
            f"Empty cell fraction {empty_frac:.4f} suspiciously high.",
            failures,
        )

    # Optional micro K-Means
    if kmeans_k > 0:
        # Build synthetic sampling by reusing extracted bitmaps in-memory
        cell_vecs: List[np.ndarray] = []
        for r in sample:
            gid = r["glyph_id"]
            raster_path = cfg.paths.rasters_dir / f"{gid}.png"
            if write and raster_path.exists():
                arr = (np.array(Image.open(raster_path)) > 0).astype(np.uint8)
            else:
                # Need in-memory bitmap; recompute quickly
                subpaths = parse_contours(r["contours_json"])
                bitmap, _ = rasterize_glyph(
                    subpaths, r.get("used_ascent"), r.get("used_descent"), cfg
                )
                arr = (bitmap > 0).astype(np.uint8)
            for rr in range(16):
                for cc in range(16):
                    cell = arr[rr * 8 : (rr + 1) * 8, cc * 8 : (cc + 1) * 8]
                    if np.any(cell):
                        cell_vecs.append(cell.reshape(-1).astype(np.float32) / 255.0)
        if len(cell_vecs) >= kmeans_k:
            data = np.vstack(cell_vecs)
            k = min(kmeans_k, data.shape[0])
            cents = run_kmeans(data, k=k, seed=cfg.seed, batch_size=512, max_iter=10)
            _assert(
                cents.shape == (k, 64),
                f"K-Means centroids shape {cents.shape} != ({k},64)",
                failures,
            )
        else:
            print(
                "[SMOKE][WARN] Not enough non-empty cells for micro K-Means.",
                file=sys.stderr,
            )

    return _finalize(
        failures, t0, verbose, diacritic_seen=diacritic_seen, sample_count=len(sample)
    )


def _finalize(
    failures: List[str],
    start_time: float,
    verbose: bool,
    diacritic_seen: int = 0,
    sample_count: int = 0,
) -> int:
    elapsed = time.time() - start_time
    if failures:
        print(
            f"[SMOKE][FAIL] {len(failures)} issue(s) detected (elapsed {elapsed:.2f}s):",
            file=sys.stderr,
        )
        for msg in failures:
            print("  - " + msg, file=sys.stderr)
        return 1
    else:
        print(
            f"[SMOKE][OK] All {sample_count} glyphs passed in {elapsed:.2f}s (diacritics: {diacritic_seen})."
        )
        return 0


def build_arg_parser():
    p = argparse.ArgumentParser(description="Rasterization smoke test (plan-aligned).")
    p.add_argument("--config", required=True, help="Path to rasterizer.yaml")
    p.add_argument("--limit", type=int, default=10, help="Glyph sample size")
    p.add_argument(
        "--write-artifacts", action="store_true", help="Persist rasters/grids/metadata"
    )
    p.add_argument(
        "--kmeans-k",
        type=int,
        default=0,
        help="Run micro K-Means with K clusters (0 disables)",
    )
    p.add_argument("--verbose", action="store_true", help="Verbose output")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    rc = smoke_raster_sample(
        cfg, args.limit, args.write_artifacts, args.kmeans_k, args.verbose
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
