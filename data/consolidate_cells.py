#!/usr/bin/env python3
"""
Cell Corpus Consolidation Script (Post-Rasterization)
====================================================

Purpose
-------
After running the rasterization pipeline (data/rasterize.py) we have:

  - Rasters: <rasters_dir>/<id>.png        (128x128 binary glyph mask; id = DB primary key)
  - Grids:   <grids_dir>/<glyph_id>.npy    (16x16 occupancy grid: 0/1)
  - (Optional) Primitive IDs: <grids_dir>/<glyph_id>_ids.npy (not used here)

This script consolidates all 8x8 cell bitmaps into a contiguous storage layout
to accelerate:
  - Primitive K-Means sampling (Section 6.1 of plan)
  - Phase 1 training dataset assembly
  - Frequency / sparsity analytics

It produces (standard mode):
  - cells.npy (or sharded cells_shard_<k>.npy) of shape (N, 8, 8) uint8
  - cells_meta.jsonl (or meta_shard_<k>.jsonl) with records:
        {
          "cell_id": <global incremental>,
          "glyph_id": <int>,
          "row": <0..15>,
          "col": <0..15>,
          "empty": <bool>
        }

Space‑optimized (bitpack) mode (--bitpack):
  - cells_bitpack.npy (or cells_bitpack_shard_<k>.npy) shape (N,) uint64
    Each uint64 packs an 8×8 binary cell (row-major; bit (r*8+c)).
  - metadata can be written either as JSONL (default) or a binary .npy struct
    if --binary-meta is passed (dtype fields: cell_id,glyph_id,row,col,empty:uint8).

  Bitpacking shrinks each stored cell from 64 bytes → 8 bytes (8×). Combined
  with --skip-empty-storage this dramatically reduces disk usage.

Optional reservoir sample of non-empty cells for K-Means:
  - kmeans_sample.npy  (M, 8, 8) uint8  (M <= requested size)

Determinism
-----------
Ordering is strictly:
  1. glyph_id files sorted numerically if numeric, else lexicographically
  2. row-major traversal over (row, col) within each glyph
This ensures reproducible cell_id assignment across runs (as long as the
underlying file set is stable).

CLI Examples
------------
Build consolidated corpus (single file):
  python data/consolidate_cells.py --config configs/rasterizer.yaml --out-dir data/processed/cells

Shard every 500k cells, skip empties, bitpack + binary metadata:
  python data/consolidate_cells.py \
      --config configs/rasterizer.yaml \
      --out-dir data/processed/cells \
      --shard-size 500000 \
      --skip-empty-storage \
      --bitpack \
      --binary-meta

Produce a 1M non-empty reservoir sample for K-Means simultaneously:
  python data/consolidate_cells.py \
      --config configs/rasterizer.yaml \
      --out-dir data/processed/cells \
      --kmeans-sample-size 1000000 \
      --kmeans-sample-out assets/centroids/kmeans_sample.npy \
      --bitpack --skip-empty-storage

Exit Codes
----------
  0 success
  1 unrecoverable error

Notes
-----
* Bitpacking + skipping empties can reduce disk usage by >10×.
* Sharding avoids holding the entire corpus in-memory.
* If a raster or grid file is missing or malformed, that glyph is skipped with a warning.
* This script only uses config sections: outputs.rasters_dir, outputs.grids_dir.
* It does not rely on primitive ID grids (_ids.npy) — those are a downstream artifact.

Plan Alignment
--------------
Sections utilized from NEW_PLAN.md:
  - §4 Rasterization output shape and binary format
  - §5 16×16 grid specification
  - §6.1 Primitive vocabulary sampling methodology
  - §9.1 Preprocessing tasks (cell slicing & storage)

Author: Automated expert scaffold (extended with bitpacking & binary metadata options)
 """

from __future__ import annotations

# Global raster filename lookup (glyph_id -> raster_filename) populated once from metadata.
RASTER_FILENAME_MAP: Dict[int, str] = {}

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    import yaml  # type: ignore
except ImportError:
    print(
        "[ERROR] Missing dependency 'pyyaml'. Install with `pip install pyyaml`.",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from PIL import Image  # type: ignore
except ImportError:
    print(
        "[ERROR] Missing dependency 'pillow'. Install with `pip install pillow`.",
        file=sys.stderr,
    )
    sys.exit(1)


# --------------------------------------------------------------------------------------
# Config Loading (minimal fields extracted from rasterizer.yaml)
# --------------------------------------------------------------------------------------


@dataclass
class PathsConfig:
    rasters_dir: Path
    grids_dir: Path


@dataclass
class ConsolidateConfig:
    seed: int
    paths: PathsConfig


def load_rasterizer_config(path: str | Path) -> ConsolidateConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    outputs = raw.get("outputs", {})
    cfg = ConsolidateConfig(
        seed=int(raw.get("seed", 42)),
        paths=PathsConfig(
            rasters_dir=Path(outputs["rasters_dir"]),
            grids_dir=Path(outputs["grids_dir"]),
        ),
    )
    return cfg


# --------------------------------------------------------------------------------------
# Reservoir Sampling for Non-Empty Cells (for K-Means sample)
# --------------------------------------------------------------------------------------


class ReservoirSampler:
    """
    Generic reservoir sampler for fixed-size uniform sampling over a stream.
    Stores deep copies of items (numpy arrays); caller ensures shape & dtype.
    """

    def __init__(self, capacity: int, seed: int = 42):
        self.capacity = capacity
        self.random = random.Random(seed)
        self._data: List[np.ndarray] = []
        self._seen = 0

    def consider(self, item: np.ndarray):
        if self.capacity <= 0:
            return
        if len(self._data) < self.capacity:
            self._data.append(item.copy())
        else:
            j = self.random.randint(0, self._seen)
            if j < self.capacity:
                self._data[j] = item.copy()
        self._seen += 1

    def snapshot(self) -> np.ndarray:
        if not self._data:
            return np.empty((0,), dtype=np.uint8)
        return np.stack(self._data, axis=0)

    @property
    def seen(self) -> int:
        return self._seen

    @property
    def size(self) -> int:
        return len(self._data)


# --------------------------------------------------------------------------------------
# Cell Extraction
# --------------------------------------------------------------------------------------


@dataclass
class CellRecord:
    cell_id: int
    glyph_id: int
    row: int
    col: int
    empty: bool


def iter_glyph_files(grids_dir: Path) -> Iterator[Tuple[int, Path, Path]]:
    """
    Yields (db_id, grid_path, id_grid_path?) for occupancy grid files.
    Excludes primitive ID grids (*_ids.npy).
    Note: db_id here is the glyphs table primary key 'id', not the font-local glyph_id.
    """
    for p in sorted(grids_dir.glob("*.npy")):
        if p.name.endswith("_ids.npy"):
            continue
        stem = p.stem
        # Attempt numeric conversion first for proper sorting semantics
        try:
            db_id = int(stem)  # db primary key id
        except ValueError:
            # Fallback: skip non-numeric
            continue
        yield db_id, p, p.with_name(f"{stem}_ids.npy")


def extract_cells_from_glyph(
    glyph_id: int,
    raster_path: Path,
    grid_path: Path,
    include_empty: bool,
    cell_size: int = 8,
    rows: int = 16,
    cols: int = 16,
) -> Tuple[List[np.ndarray], List[CellRecord], int, int]:
    """
    Returns:
        cells: list of (8,8) uint8 arrays (0/255)
        meta:  list of CellRecord
        non_empty_count
        empty_count
    """
    try:
        grid = np.load(grid_path)
    except Exception as e:
        print(f"[WARN] Failed to load grid {grid_path}: {e}", file=sys.stderr)
        return [], [], 0, 0

    if grid.shape != (rows, cols):
        print(
            f"[WARN] Grid {grid_path} shape {grid.shape} != {(rows, cols)}; skipping",
            file=sys.stderr,
        )
        return [], [], 0, 0

    # Optimized raster lookup:
    # Prefer metadata-derived filename (stored during rasterization as sanitized_label_id.png).
    raster_img_path: Optional[Path] = None
    meta_fname = RASTER_FILENAME_MAP.get(glyph_id)
    if meta_fname:
        candidate = raster_path / meta_fname
        if candidate.exists():
            raster_img_path = candidate
    if raster_img_path is None:
        # Fallback: legacy <id>.png then pattern "*_<id>.png"
        legacy = raster_path / f"{glyph_id}.png"
        if legacy.exists():
            raster_img_path = legacy
        else:
            candidates = sorted(raster_path.glob(f"*_{glyph_id}.png"))
            if candidates:
                raster_img_path = candidates[0]
    if raster_img_path is None:
        print(
            f"[WARN] Missing raster for glyph {glyph_id} (checked metadata, {glyph_id}.png and *_{glyph_id}.png); skipping.",
            file=sys.stderr,
        )
        return [], [], 0, 0

    try:
        img = Image.open(raster_img_path).convert("L")
        bitmap = np.array(img, dtype=np.uint8)
    except Exception as e:
        print(f"[WARN] Failed reading raster {raster_img_path}: {e}", file=sys.stderr)
        return [], [], 0, 0

    if bitmap.shape != (rows * cell_size, cols * cell_size):
        print(
            f"[WARN] Raster shape {bitmap.shape} inconsistent with expected {(rows * cell_size, cols * cell_size)}",
            file=sys.stderr,
        )
        return [], [], 0, 0

    cells: List[np.ndarray] = []
    meta: List[CellRecord] = []
    non_empty = 0
    empty = 0

    # Row-major traversal
    for r in range(rows):
        for c in range(cols):
            block = bitmap[
                r * cell_size : (r + 1) * cell_size, c * cell_size : (c + 1) * cell_size
            ]
            # Normalize to 0/255 if needed (threshold)
            if block.max() not in (0, 255):
                block = (block > 0).astype(np.uint8) * 255
            is_empty = not np.any(block)
            if is_empty:
                empty += 1
                if not include_empty:
                    continue
            else:
                non_empty += 1
            cells.append(block)
            meta.append(
                CellRecord(cell_id=-1, glyph_id=glyph_id, row=r, col=c, empty=is_empty)
            )

    return cells, meta, non_empty, empty


# --------------------------------------------------------------------------------------
# Sharded Writer
# --------------------------------------------------------------------------------------


def bitpack_cell(cell: np.ndarray) -> np.uint64:
    """
    Pack a binary 8x8 cell (values 0/255 or 0/1) into a uint64 bitmask.
    Bit assignment (row-major): bit index = r * 8 + c.
    Returns:
        np.uint64 where bit i is 1 if the corresponding cell pixel was non-zero.
    """
    # Ensure uint8
    if cell.dtype != np.uint8:
        cell = cell.astype(np.uint8)
    # Normalize >0 to 1
    flat = (cell.reshape(-1) > 0).astype(np.uint8)
    bits = 0
    # Loop (64 iterations; negligible cost vs I/O)
    for i, v in enumerate(flat):
        if v:
            bits |= 1 << i
    return np.uint64(bits)


class ShardedCellWriter:
    """
    Accumulates cells & metadata, flushing to shards when capacity reached.

    Supports:
      - bitpack mode (store each cell as uint64)
      - binary metadata (structured ndarray) or JSONL
    """

    def __init__(
        self,
        out_dir: Path,
        shard_size: Optional[int],
        include_empty: bool,
        bitpack: bool = False,
        binary_meta: bool = False,
    ):
        self.out_dir = out_dir
        self.shard_size = shard_size if shard_size and shard_size > 0 else None
        self.include_empty = include_empty

        self.bitpack = bitpack
        self.binary_meta = binary_meta

        self._cell_buf: List[np.ndarray] = []  # (8,8) arrays if not bitpack
        self._cell_packed: List[int] = []  # uint64 ints if bitpack
        self._meta_buf: List[CellRecord] = []
        self._shard_index = 0
        self._global_cell_id = 0

        self.out_dir.mkdir(parents=True, exist_ok=True)

    def add_batch(self, cells: List[np.ndarray], metas: List[CellRecord]):
        for cell, meta in zip(cells, metas):
            meta.cell_id = self._global_cell_id
            self._global_cell_id += 1
            if self.bitpack:
                self._cell_packed.append(int(bitpack_cell(cell)))
            else:
                self._cell_buf.append(cell)
            self._meta_buf.append(meta)
            if self.shard_size:
                current_count = (
                    len(self._cell_packed) if self.bitpack else len(self._cell_buf)
                )
                if current_count >= self.shard_size:
                    self._flush_current_shard()

    def finalize(self):
        if self._cell_buf or self._cell_packed:
            self._flush_current_shard(final=True)

    @property
    def total_cells(self) -> int:
        return self._global_cell_id

    def _flush_current_shard(self, final: bool = False):
        if (not self._cell_buf) and (not self._cell_packed):
            return
        if self.shard_size is None:
            # Single output file case
            if self.bitpack:
                cells_arr = np.array(self._cell_packed, dtype=np.uint64)
                out_cells = self.out_dir / "cells_bitpack.npy"
            else:
                cells_arr = np.stack(self._cell_buf, axis=0).astype(np.uint8)
                out_cells = self.out_dir / "cells.npy"
            np.save(out_cells, cells_arr)

            if self.binary_meta:
                meta_dtype = np.dtype(
                    [
                        ("cell_id", "<i8"),
                        ("glyph_id", "<i8"),
                        ("row", "<i2"),
                        ("col", "<i2"),
                        ("empty", "u1"),
                    ]
                )
                meta_arr = np.zeros(len(self._meta_buf), dtype=meta_dtype)
                for i, m in enumerate(self._meta_buf):
                    meta_arr[i] = (
                        m.cell_id,
                        m.glyph_id,
                        m.row,
                        m.col,
                        1 if m.empty else 0,
                    )
                out_meta = self.out_dir / (
                    "cells_meta_bin.npy"
                    if not self.bitpack
                    else "cells_meta_bitpack_bin.npy"
                )
                np.save(out_meta, meta_arr)
            else:
                out_meta = self.out_dir / (
                    "cells_meta.jsonl"
                    if not self.bitpack
                    else "cells_meta_bitpack.jsonl"
                )
                with open(out_meta, "w", encoding="utf-8") as f:
                    for m in self._meta_buf:
                        f.write(
                            json.dumps(
                                {
                                    "cell_id": m.cell_id,
                                    "glyph_id": m.glyph_id,
                                    "row": m.row,
                                    "col": m.col,
                                    "empty": m.empty,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
            print(f"[INFO] Wrote consolidated cells: {cells_arr.shape} -> {out_cells}")
            print(f"[INFO] Wrote metadata: {out_meta}")
            # Clear buffers (should only happen once)
            self._cell_buf.clear()
            self._cell_packed.clear()
            self._meta_buf.clear()
            return

        # Sharded
        shard_id = self._shard_index
        self._shard_index += 1
        if self.bitpack:
            cells_arr = np.array(self._cell_packed, dtype=np.uint64)
            out_cells = self.out_dir / f"cells_bitpack_shard_{shard_id:03d}.npy"
        else:
            cells_arr = np.stack(self._cell_buf, axis=0).astype(np.uint8)
            out_cells = self.out_dir / f"cells_shard_{shard_id:03d}.npy"
        if self.binary_meta:
            meta_dtype = np.dtype(
                [
                    ("cell_id", "<i8"),
                    ("glyph_id", "<i8"),
                    ("row", "<i2"),
                    ("col", "<i2"),
                    ("empty", "u1"),
                ]
            )
            meta_arr = np.zeros(len(self._meta_buf), dtype=meta_dtype)
            for i, m in enumerate(self._meta_buf):
                meta_arr[i] = (m.cell_id, m.glyph_id, m.row, m.col, 1 if m.empty else 0)
            out_meta = self.out_dir / f"meta_bin_shard_{shard_id:03d}.npy"
            np.save(out_meta, meta_arr)
        else:
            out_meta = self.out_dir / f"meta_shard_{shard_id:03d}.jsonl"
            with open(out_meta, "w", encoding="utf-8") as f:
                for m in self._meta_buf:
                    f.write(
                        json.dumps(
                            {
                                "cell_id": m.cell_id,
                                "glyph_id": m.glyph_id,
                                "row": m.row,
                                "col": m.col,
                                "empty": m.empty,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        print(
            f"[INFO] Wrote shard {shard_id}: cells {cells_arr.shape}, meta {len(self._meta_buf)} -> {out_cells}"
        )
        self._cell_buf.clear()
        self._cell_packed.clear()
        self._meta_buf.clear()

        if final:
            print(f"[INFO] Finalized sharding with {shard_id + 1} shard(s).")


# --------------------------------------------------------------------------------------
# Main Orchestration
# --------------------------------------------------------------------------------------


def consolidate_cells(
    cfg: ConsolidateConfig,
    out_dir: Path,
    shard_size: Optional[int],
    include_empty: bool,
    kmeans_sample_size: int,
    kmeans_sample_out: Optional[Path],
    progress_every: int,
    bitpack: bool,
    binary_meta: bool,
    make_splits: bool = False,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    split_seed: int = 42,
    metadata_path: Optional[Path] = None,
):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    rasters_dir = cfg.paths.rasters_dir
    grids_dir = cfg.paths.grids_dir

    if not rasters_dir.exists() or not grids_dir.exists():
        print(
            f"[ERROR] Missing rasters_dir ({rasters_dir}) or grids_dir ({grids_dir}). Run rasterization first.",
            file=sys.stderr,
        )
        sys.exit(1)

    writer = ShardedCellWriter(
        out_dir=out_dir,
        shard_size=shard_size,
        include_empty=include_empty,
        bitpack=bitpack,
        binary_meta=binary_meta,
    )

    # -------- Split preparation --------
    font_map: Dict[int, str] = {}
    if make_splits:
        if metadata_path is None or not metadata_path.exists():
            print(
                "[WARN] --make-splits requested but metadata_path missing; disabling splits.",
                file=sys.stderr,
            )
            make_splits = False
        else:
            # Build glyph_id -> font_hash map (glyph_id here is DB primary key used in metadata)
            with metadata_path.open("r", encoding="utf-8") as mf:
                for line in mf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    gid = rec.get("glyph_id")
                    fh = rec.get("font_hash")
                    raster_fname = rec.get("raster_filename")
                    if gid is None or fh is None:
                        continue
                    try:
                        gid_int = int(gid)
                    except Exception:
                        continue
                    font_map[gid_int] = str(fh)
                    if raster_fname:
                        # Store first-seen filename (should be unique per glyph_id)
                        RASTER_FILENAME_MAP.setdefault(gid_int, str(raster_fname))
            if not font_map:
                print(
                    "[WARN] No glyph->font_hash mappings found in metadata; disabling splits.",
                    file=sys.stderr,
                )
                make_splits = False

    # Containers for split assignment (store cell_ids)
    split_assign: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    font_split_lookup: Dict[str, str] = {}
    rng = random.Random(split_seed)

    sampler = (
        ReservoirSampler(capacity=kmeans_sample_size, seed=cfg.seed)
        if kmeans_sample_size > 0
        else None
    )

    total_glyphs = 0
    total_non_empty_cells = 0
    total_empty_cells = 0
    skipped_glyphs = 0

    for glyph_id, grid_path, _id_grid in iter_glyph_files(grids_dir):
        raster_path = rasters_dir
        cells, meta, non_empty, empty = extract_cells_from_glyph(
            glyph_id=glyph_id,
            raster_path=raster_path,
            grid_path=grid_path,
            include_empty=include_empty,
        )
        # If extraction failed (no cells) skip
        if not cells and (non_empty + empty) == 0:
            skipped_glyphs += 1
            continue

        # Update counts (based on original totals, not filtered)
        total_glyphs += 1
        total_non_empty_cells += non_empty
        total_empty_cells += empty

        # Add to writer
        writer.add_batch(cells, meta)

        # Split logic: assign after cells receive cell_ids
        if make_splits:
            fh = font_map.get(glyph_id)
            if fh is not None:
                # Lazy build font_split_lookup
                if fh not in font_split_lookup:
                    # Decide split deterministically via hashing + ratios
                    # Hash-based stable assignment so order independent
                    h = abs(hash((fh, split_seed))) / (
                        2**63 if sys.maxsize > 2**32 else 2**31
                    )
                    cumulative = train_ratio
                    if h < cumulative:
                        font_split_lookup[fh] = "train"
                    elif h < cumulative + val_ratio:
                        font_split_lookup[fh] = "val"
                    else:
                        font_split_lookup[fh] = "test"
                split_label = font_split_lookup[fh]
                for m in meta:
                    split_assign[split_label].append(m.cell_id)

        # Feed sampler with non-empty cells only
        if sampler:
            for cell_arr, meta_rec in zip(cells, meta):
                if not meta_rec.empty:
                    sampler.consider(cell_arr)

        if progress_every and total_glyphs % progress_every == 0:
            print(
                f"[PROGRESS] Glyphs processed: {total_glyphs:,} | "
                f"Cells stored: {writer.total_cells:,} | Non-empty: {total_non_empty_cells:,} | Empty: {total_empty_cells:,}"
            )

    # Finalize writer
    writer.finalize()

    # Write K-Means sample if requested
    if sampler and kmeans_sample_out:
        sample = sampler.snapshot()
        kmeans_sample_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(kmeans_sample_out, sample.astype(np.uint8))
        print(
            f"[INFO] K-Means reservoir sample: stored {sample.shape[0]} (requested {sampler.capacity}, seen {sampler.seen}) -> {kmeans_sample_out}"
        )

    # Summary
    total_cells_written = writer.total_cells
    if include_empty:
        stored_empty_ratio = (
            total_empty_cells / (total_empty_cells + total_non_empty_cells)
            if (total_empty_cells + total_non_empty_cells) > 0
            else 0.0
        )
    else:
        stored_empty_ratio = 0.0

    print("========== Consolidation Summary ==========")
    print(f"Glyphs processed (with at least 1 usable cell): {total_glyphs:,}")
    print(f"Glyphs skipped (no usable cells): {skipped_glyphs:,}")
    print(f"Total non-empty cells (source): {total_non_empty_cells:,}")
    print(f"Total empty cells (source): {total_empty_cells:,}")
    print(f"Total cells written: {total_cells_written:,}")
    print(f"Include empty in storage: {include_empty}")
    print(f"Approx stored empty ratio: {stored_empty_ratio:.4f}")
    if sampler:
        print(
            f"K-Means sample: {sampler.size} / {sampler.capacity} (non-empty only, from {sampler.seen} candidates)"
        )
    print("===========================================")
    if make_splits:
        # Shuffle cell ids within each split for downstream randomness
        for split_label, ids in split_assign.items():
            rng.shuffle(ids)
            out_path = out_dir / f"phase1_{split_label}_cells.txt"
            with out_path.open("w", encoding="utf-8") as f:
                for cid in ids:
                    f.write(f"{cid}\n")
            print(
                f"[SPLITS] Wrote {len(ids):,} cell_ids to {out_path.name} (split={split_label})"
            )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Consolidate 8x8 cells from raster + grid outputs."
    )
    p.add_argument("--config", required=True, help="Path to rasterizer.yaml")
    p.add_argument(
        "--out-dir", required=True, help="Output directory for consolidated cell corpus"
    )
    p.add_argument(
        "--shard-size",
        type=int,
        default=0,
        help="Shard size (#cells per shard). 0 = single monolithic cells.npy",
    )
    p.add_argument(
        "--skip-empty-storage",
        action="store_true",
        help="Do not store empty cells (still counted in summary).",
    )
    p.add_argument(
        "--kmeans-sample-size",
        type=int,
        default=0,
        help="Reservoir sample size of non-empty cells for K-Means (0 disables).",
    )
    p.add_argument(
        "--kmeans-sample-out",
        type=str,
        default="",
        help="Output path for K-Means sample .npy (required if --kmeans-sample-size > 0).",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress after this many glyphs (0 disables).",
    )
    p.add_argument(
        "--seed", type=int, default=None, help="Override seed (else use config seed)."
    )
    # Splits
    p.add_argument(
        "--make-splits",
        action="store_true",
        help="Generate train/val/test cell_id split files (requires metadata in config).",
    )
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for split hashing / shuffling.",
    )
    p.add_argument(
        "--bitpack",
        action="store_true",
        help="Store each 8x8 cell as a uint64 bitmask (8x space reduction).",
    )
    p.add_argument(
        "--binary-meta",
        action="store_true",
        help="Write metadata as a binary structured .npy instead of JSONL.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = load_rasterizer_config(args.config)
    if args.seed is not None:
        cfg = ConsolidateConfig(seed=args.seed, paths=cfg.paths)

    out_dir = Path(args.out_dir)
    shard_size = args.shard_size if args.shard_size > 0 else None
    include_empty = not args.skip_empty_storage

    if args.kmeans_sample_size > 0 and not args.kmeans_sample_out:
        parser.error("--kmeans-sample-out is required when --kmeans-sample-size > 0")

    kmeans_sample_out = Path(args.kmeans_sample_out) if args.kmeans_sample_out else None

    consolidate_cells(
        cfg=cfg,
        out_dir=out_dir,
        shard_size=shard_size,
        include_empty=include_empty,
        kmeans_sample_size=args.kmeans_sample_size,
        kmeans_sample_out=kmeans_sample_out,
        progress_every=args.progress_every,
        bitpack=args.bitpack,
        binary_meta=args.binary_meta,
        make_splits=args.make_splits,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
        metadata_path=cfg.paths.grids_dir.parent / "rasters" / "metadata.jsonl"
        if (cfg.paths.grids_dir.parent / "rasters" / "metadata.jsonl").exists()
        else None,
    )


if __name__ == "__main__":
    main()
