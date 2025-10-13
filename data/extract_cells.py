#!/usr/bin/env python3
"""
extract_cells.py

Phase 1 Cell Extraction Utility
--------------------------------

Purpose:
  Convert per-glyph 128x128 binary raster PNGs into a corpus of 8x8 cell
  patches suitable for primitive (Phase 1) clustering and classification.

Pipeline:
  1. Read metadata JSONL (produced by rasterization) to iterate glyphs.
  2. For each raster image (binary 0/255; shape 128x128):
       - Slice into 16x16 grid of 8x8 cells.
       - Append each cell (uint8 0/255) to an in-memory shard buffer.
       - Emit manifest lines with per-cell metadata.
  3. Flush shard buffers to:
       out_dir/
         shard_00000.npy
         shard_00001.npy
         ...
         manifest.jsonl
  4. Optionally (if --single-file) concatenate all cells into cells.npy
     instead of sharded output (only recommended for modest corpus sizes).
  5. Optionally generate deterministic train/val/test splits at the CELL LEVEL
     by first partitioning fonts (font_hash) → splits, then assigning glyphs
     and their cells to those splits.

Outputs:
  - out_dir/shard_XXXXX.npy  (or cells.npy if --single-file)
  - out_dir/manifest.jsonl   (one JSON per cell)
  - out_dir/phase1_train_cells.txt
  - out_dir/phase1_val_cells.txt
  - out_dir/phase1_test_cells.txt

Manifest Line Schema:
  {
    "cell_id": int,
    "glyph_id": int,
    "font_hash": str,
    "row": int,          # 0..15
    "col": int,          # 0..15
    "is_empty": bool
  }

Why we need this:
  The project’s Phase 1 primitive learning operates on full 8x8 pixel
  patterns, not a single occupancy bit. This script materializes those
  patterns for clustering (K-Means) and later CNN training.

CLI Examples:
  Basic (sharded):
    python -m data.extract_cells \
      --rasters-dir data/rasters \
      --metadata data/rasters/metadata.jsonl \
      --out-dir data/processed/cells

  Single-file (all cells in cells.npy):
    python -m data.extract_cells \
      --rasters-dir data/rasters \
      --metadata data/rasters/metadata.jsonl \
      --out-dir data/processed/cells \
      --single-file

  With splits (80/10/10) and shard size 250k:
    python -m data.extract_cells \
      --rasters-dir data/rasters \
      --metadata data/rasters/metadata.jsonl \
      --out-dir data/processed/cells \
      --make-splits \
      --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \
      --shard-size 250000

  Debug small sample:
    python -m data.extract_cells \
      --rasters-dir data/rasters \
      --metadata data/rasters/metadata.jsonl \
      --out-dir data/processed/cells_debug \
      --max-glyphs 500 \
      --single-file

Notes:
  - Assumes rasters are binary (0 or 255). If slight grayscale noise present,
    a binarization threshold is applied (default 128).
  - Memory: sharding keeps RAM bounded; each shard flushes at shard_size cells.
  - Splits: Deterministic per font_hash (so all glyphs of a font reside in the same split).

Limitations / Future:
  - Could add LMDB backend for very large corpora.
  - Could store per-cell fill ratio to assist sampling / weighting.
  - Could optionally compress shards (np.savez_compressed) if disk is a concern.

License:
  Follows the root project license.

"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Sequence, Tuple

try:
    from PIL import Image
except ImportError as e:  # pragma: no cover
    raise RuntimeError("Pillow is required (pip install Pillow)") from e

import numpy as np


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class MetaRecord:
    glyph_id: int
    raster_filename: str
    font_hash: str


# ---------------------------------------------------------------------------
# Metadata Loading
# ---------------------------------------------------------------------------


def iter_metadata(
    path: Path,
    limit: Optional[int] = None,
    require_fields: Sequence[str] = ("glyph_id", "raster_filename", "font_hash"),
) -> Generator[MetaRecord, None, None]:
    """
    Stream metadata.jsonl lines, yield minimal fields required for cell extraction.
    """
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if any(k not in rec for k in require_fields):
                continue
            yield MetaRecord(
                glyph_id=int(rec["glyph_id"]),
                raster_filename=str(rec["raster_filename"]),
                font_hash=str(rec["font_hash"]),
            )
            count += 1
            if limit is not None and count >= limit:
                break


# ---------------------------------------------------------------------------
# Cell Extraction
# ---------------------------------------------------------------------------


def slice_raster_to_cells(
    raster_path: Path,
    binarize: bool = True,
    threshold: int = 128,
) -> np.ndarray:
    """
    Load a 128x128 raster (binary or near-binary) and return an array shape (16,16,8,8) uint8 (0 or 255).
    """
    img = Image.open(raster_path).convert("L")
    arr = np.array(img)
    if arr.shape != (128, 128):
        raise ValueError(f"Unexpected raster shape {arr.shape} for {raster_path.name}")
    if binarize:
        arr = (arr >= threshold).astype(np.uint8) * 255

    cells = np.zeros((16, 16, 8, 8), dtype=np.uint8)
    for r in range(16):
        for c in range(16):
            block = arr[r * 8 : (r + 1) * 8, c * 8 : (c + 1) * 8]
            cells[r, c] = block
    return cells


# ---------------------------------------------------------------------------
# Sharded Writer
# ---------------------------------------------------------------------------


class CellShardWriter:
    """
    Accumulates 8x8 cells until shard_size reached; flushes to disk as shard_XXXXX.npy
    under out_dir. If single_file=True, defers flush until finalize() and writes
    'cells.npy'.
    """

    def __init__(self, out_dir: Path, shard_size: int, single_file: bool):
        self.out_dir = out_dir
        self.shard_size = shard_size
        self.single_file = single_file
        self.buffer: List[np.ndarray] = []
        self.shard_index = 0
        self.total_written = 0
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def add(self, cell: np.ndarray):
        self.buffer.append(cell)
        if not self.single_file and len(self.buffer) >= self.shard_size:
            self._flush_shard()

    def _flush_shard(self):
        if not self.buffer:
            return
        arr = np.stack(self.buffer, axis=0)  # (N,8,8)
        fname = (
            "cells.npy"
            if self.single_file and self.shard_index == 0
            else f"shard_{self.shard_index:05d}.npy"
        )
        np.save(self.out_dir / fname, arr)
        self.total_written += len(arr)
        self.shard_index += 1
        self.buffer.clear()
        print(
            f"[write] Flushed {arr.shape[0]:,} cells to {fname} (total written {self.total_written:,})",
            file=sys.stderr,
        )

    def finalize(self):
        if self.buffer:
            self._flush_shard()
        # If single_file and multiple shards flushed (shouldn't happen), consolidate.
        if self.single_file and self.shard_index > 1:
            # Merge shards into single file (rare path).
            all_arrays = []
            for i in range(self.shard_index):
                shard_path = self.out_dir / f"shard_{i:05d}.npy"
                if shard_path.exists():
                    all_arrays.append(np.load(shard_path, mmap_mode="r"))
            merged = np.concatenate(all_arrays, axis=0)
            np.save(self.out_dir / "cells.npy", merged)
            # Optionally remove shards
            for i in range(self.shard_index):
                p = self.out_dir / f"shard_{i:05d}.npy"
                try:
                    p.unlink()
                except OSError:
                    pass
            print(
                f"[finalize] Consolidated {self.shard_index} shards → cells.npy (shape={merged.shape})",
                file=sys.stderr,
            )


# ---------------------------------------------------------------------------
# Font-Based Splitting
# ---------------------------------------------------------------------------


def split_fonts(
    font_hashes: List[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, str]:
    """
    Deterministically partition font_hash list into train/val/test splits.
    Returns: { font_hash: split_label }
    """
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-6):
        raise ValueError("Train/val/test ratios must sum to 1.0")

    rng = random.Random(seed)
    fonts = list(sorted(set(font_hashes)))
    rng.shuffle(fonts)

    n = len(fonts)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    # Ensure total coverage
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    n_test = n - n_train - n_val

    train_set = set(fonts[:n_train])
    val_set = set(fonts[n_train : n_train + n_val])
    test_set = set(fonts[n_train + n_val :])

    mapping = {}
    for fh in fonts:
        if fh in train_set:
            mapping[fh] = "train"
        elif fh in val_set:
            mapping[fh] = "val"
        else:
            mapping[fh] = "test"
    return mapping


def write_cell_splits(
    manifest_path: Path,
    font_split_map: Dict[str, str],
    out_dir: Path,
    seed: int,
    shuffle_each: bool = True,
):
    """
    Read manifest.jsonl, aggregate cell_ids per split via font_hash mapping,
    and write text files listing cell_ids (one per line).
    """
    splits: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            fh = rec["font_hash"]
            cell_id = rec["cell_id"]
            split = font_split_map.get(fh)
            if split in splits:
                splits[split].append(cell_id)

    rng = random.Random(seed)
    for k, ids in splits.items():
        if shuffle_each:
            rng.shuffle(ids)
        out_path = out_dir / f"phase1_{k}_cells.txt"
        with out_path.open("w", encoding="utf-8") as f:
            for cid in ids:
                f.write(f"{cid}\n")
        print(
            f"[splits] Wrote {len(ids):,} cell_ids to {out_path.name} (split={k})",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Main Extraction Logic
# ---------------------------------------------------------------------------


def extract_cells(
    rasters_dir: Path,
    metadata_path: Path,
    out_dir: Path,
    shard_size: int,
    single_file: bool,
    max_glyphs: Optional[int],
    make_splits: bool,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    split_seed: int,
    binarize: bool,
    threshold: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    writer = CellShardWriter(
        out_dir=out_dir, shard_size=shard_size, single_file=single_file
    )

    cell_id = 0
    glyph_count = 0
    font_hashes_seen: List[str] = []

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for meta in iter_metadata(metadata_path, limit=max_glyphs):
            raster_path = rasters_dir / meta.raster_filename
            if not raster_path.exists():
                print(
                    f"[warn] Missing raster file: {raster_path.name}", file=sys.stderr
                )
                continue

            try:
                cells = slice_raster_to_cells(
                    raster_path, binarize=binarize, threshold=threshold
                )  # (16,16,8,8)
            except Exception as e:
                print(
                    f"[error] Failed slicing {raster_path.name}: {e}", file=sys.stderr
                )
                continue

            for r in range(16):
                for c in range(16):
                    patch = cells[r, c]  # (8,8)
                    is_empty = bool(np.count_nonzero(patch) == 0)
                    writer.add(patch)
                    record = {
                        "cell_id": cell_id,
                        "glyph_id": meta.glyph_id,
                        "font_hash": meta.font_hash,
                        "row": r,
                        "col": c,
                        "is_empty": is_empty,
                    }
                    manifest_file.write(json.dumps(record) + "\n")
                    cell_id += 1

            font_hashes_seen.append(meta.font_hash)
            glyph_count += 1
            if glyph_count % 500 == 0:
                print(
                    f"[progress] Processed {glyph_count:,} glyphs → {cell_id:,} cells",
                    file=sys.stderr,
                )

    writer.finalize()
    print(
        f"[done] Cells extraction complete: glyphs={glyph_count:,}, cells={cell_id:,}",
        file=sys.stderr,
    )

    # Optional splits
    if make_splits:
        print("[splits] Generating font-based splits...", file=sys.stderr)
        font_split_map = split_fonts(
            font_hashes_seen,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=split_seed,
        )
        write_cell_splits(
            manifest_path=manifest_path,
            font_split_map=font_split_map,
            out_dir=out_dir,
            seed=split_seed,
        )
        print("[splits] Split generation complete.", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract 8x8 cell patches from 128x128 glyph rasters."
    )
    p.add_argument(
        "--rasters-dir",
        type=Path,
        required=True,
        help="Directory containing raster PNGs (label_id.png style).",
    )
    p.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Path to metadata.jsonl produced during rasterization.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for cells (shards or single file) and manifest.",
    )
    p.add_argument(
        "--shard-size",
        type=int,
        default=200_000,
        help="Number of cells per shard file (ignored if --single-file).",
    )
    p.add_argument(
        "--single-file",
        action="store_true",
        help="If set, writes all cells to a single cells.npy (may be large).",
    )
    p.add_argument(
        "--max-glyphs",
        type=int,
        default=None,
        help="Optional limit for debugging (process only first N glyphs).",
    )
    # Splits
    p.add_argument(
        "--make-splits",
        action="store_true",
        help="Generate train/val/test cell split files grouped by font.",
    )
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for font split shuffling.",
    )
    # Binarization
    p.add_argument(
        "--no-binarize",
        action="store_true",
        help="Disable binarization thresholding (assume rasters already pure 0/255).",
    )
    p.add_argument(
        "--threshold",
        type=int,
        default=128,
        help="Binarization threshold (ignored if --no-binarize).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global seed (used only for internal deterministic behavior).",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if not args.rasters_dir.exists():
        print(f"[error] rasters dir not found: {args.rasters_dir}", file=sys.stderr)
        return 2
    if not args.metadata.exists():
        print(f"[error] metadata file not found: {args.metadata}", file=sys.stderr)
        return 2

    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        extract_cells(
            rasters_dir=args.rasters_dir,
            metadata_path=args.metadata,
            out_dir=args.out_dir,
            shard_size=args.shard_size,
            single_file=args.single_file,
            max_glyphs=args.max_glyphs,
            make_splits=args.make_splits,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            split_seed=args.split_seed,
            binarize=not args.no_binarize,
            threshold=args.threshold,
        )
    except Exception as e:
        print(f"[fatal] {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
