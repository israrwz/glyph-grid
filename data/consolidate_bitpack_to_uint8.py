#!/usr/bin/env python3
"""
Consolidate Bitpacked Cell Shards -> Single uint8 Memmap Array

Purpose
-------
Decodes all bitpacked cell shard files (cells_bitpack_shard_*.npy) in a directory
into a single memory-mapped uint8 tensor file:
    cells_uint8.npy   (shape: (TOTAL_CELLS, 8, 8), dtype=uint8, values {0,255})

Optionally also writes:
    cells_bool.npy    (shape: (TOTAL_CELLS, 64), dtype=bool) if --flat-bool requested
    labels_uint16.npy (copied / sliced from a provided dense assignments .npy) if --assign-dense given
    metadata.json     (JSON with summary statistics)

Motivation
----------
Training currently decodes bitpack masks on-the-fly (CPU-heavy and memory-fragmenting).
Consolidation trades one-time decode + ~8x on-disk size increase (still modest) for
dramatically lower CPU overhead and stable batch assembly throughput.

Workflow
--------
1. Provide the original bitpack directory (containing cells_bitpack_shard_*.npy).
2. (Optional) Provide the dense assignments file (primitive_assignments_dense_uint16.npy)
   if you want an aligned labels file for zero-copy dataset loading later.
3. Specify an output directory (will be created if missing).
4. Run this script. A progress bar (or textual shard progress) prints to stderr.

After that, you can implement / switch a dataset class to mmap load the new
cells_uint8.npy + labels_uint16.npy for near-zero per-item overhead.

Bitpacking Assumptions
----------------------
- Each uint64 packs an 8x8 binary cell with bit index (r*8 + c) (consistent with existing code).
- We reuse the same decoding strategy (`view(np.uint8) -> unpackbits -> reshape`) as
  the training script to preserve orientation.

CLI
---
python -m data.consolidate_bitpack_to_uint8 \
    --cells-dir data/processed/cells_bitpack \
    --out data/processed/cells_bitpack_consolidated \
    --assign-dense data/processed/cells_bitpack_subset80k_nomani/primitive_assignments_dense_uint16.npy \
    --flat-bool \
    --overwrite

Arguments
---------
--cells-dir       Directory containing bitpack shards (cells_bitpack_shard_*.npy)
--shard-pattern   Glob pattern for shard discovery (default: cells_bitpack_shard_*.npy)
--out             Output directory for consolidated files
--assign-dense    (Optional) Path to dense assignment array (uint16) to write labels_uint16.npy
--flat-bool       Additionally write a boolean flat (N,64) array (cells_bool.npy)
--chunk           Decode shards in sub-chunks of this many cells (to limit peak RAM) [default: 0 => whole shard]
--overwrite       Overwrite existing output files
--no-metadata     Skip writing metadata.json
--verify          Perform random verification samples after writing

Outputs
-------
Required:
  cells_uint8.npy
Optional:
  cells_bool.npy
  labels_uint16.npy
  metadata.json

Performance Notes
-----------------
- Memory footprint during a shard decode: ~ (shard_cells * (8 bytes + 64 bytes)) transient.
  You can reduce this with --chunk.
- If dataset is extremely large and you need to stay within RAM limits, use a non-zero --chunk
  (e.g. 1_000_000) to decode and write in slices.

Future Extensions
-----------------
- Support writing float16 directly to skip later float conversion in training.
- Parallel shard decode (current implementation is sequential for deterministic order
  and lower peak RAM).

Author: Auto-generated utility script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np


def log(msg: str) -> None:
    print(f"[consolidate] {msg}", file=sys.stderr, flush=True)


def discover_shards(cells_dir: Path, pattern: str) -> List[Path]:
    shards = sorted(cells_dir.glob(pattern))
    if not shards:
        raise FileNotFoundError(
            f"No shard files matched pattern '{pattern}' under {cells_dir}"
        )
    return shards


def count_total_cells(shards: List[Path]) -> int:
    total = 0
    lengths = []
    for sp in shards:
        arr = np.load(sp, mmap_mode="r")
        n = len(arr)
        total += n
        lengths.append(n)
    return total


def decode_bitpack_chunk(raw: np.ndarray) -> np.ndarray:
    """
    raw: uint64 vector length N
    Returns uint8 array (N,8,8) with values {0,255}
    Orientation: little-endian bit significance per byte (bit 0 -> column 0),
    matching original iterative unpack logic.
    """
    # View underlying bytes (little-endian); shape (N, 8 bytes)
    bytes_view = raw.view(np.uint8).reshape(-1, 8)
    # Unpack bits MSB->LSB per byte, then reverse within each byte to get little-endian order
    bits_msb = np.unpackbits(
        bytes_view, axis=1
    )  # (N,64) groups of 8 bits per original byte
    bits_le = bits_msb.reshape(-1, 8, 8)[:, :, ::-1].reshape(-1, 64)
    decoded = (bits_le.reshape(-1, 8, 8) * 255).astype(np.uint8)
    return decoded


def consolidate(
    cells_dir: Path,
    shard_pattern: str,
    out_dir: Path,
    assign_dense: Optional[Path],
    write_flat_bool: bool,
    chunk_size: int,
    overwrite: bool,
    write_metadata: bool,
    verify: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = discover_shards(cells_dir, shard_pattern)
    log(f"Found {len(shards)} shard(s).")
    total_cells = count_total_cells(shards)
    log(f"Total bitpacked cells: {total_cells:,}")

    cells_out = out_dir / "cells_uint8.npy"
    flat_out = out_dir / "cells_bool.npy"
    labels_out = out_dir / "labels_uint16.npy"
    meta_out = out_dir / "metadata.json"

    if not overwrite and cells_out.exists():
        raise FileExistsError(
            f"{cells_out} exists. Use --overwrite to force regeneration."
        )
    if assign_dense and not assign_dense.exists():
        raise FileNotFoundError(f"Dense assignments file missing: {assign_dense}")

    # Prepare memmaps
    log("Allocating memmap for decoded cells (uint8)...")
    decoded_mm = np.memmap(
        cells_out, dtype=np.uint8, mode="w+", shape=(total_cells, 8, 8)
    )

    if write_flat_bool:
        flat_mm = np.memmap(
            flat_out, dtype=np.bool_, mode="w+", shape=(total_cells, 64)
        )
    else:
        flat_mm = None

    if assign_dense:
        log("Loading dense assignments (mmap read-only)...")
        labels_src = np.load(assign_dense, mmap_mode="r")
        if labels_src.dtype != np.uint16:
            log(
                f"[warn] Dense assignments dtype {labels_src.dtype} != uint16; casting on write."
            )
        labels_mm = np.memmap(
            labels_out, dtype=np.uint16, mode="w+", shape=(total_cells,)
        )
    else:
        labels_src = None
        labels_mm = None

    # Decode each shard
    cursor = 0
    for shard_idx, sp in enumerate(shards, start=1):
        raw = np.load(sp, mmap_mode="r")  # uint64 vector
        n = len(raw)
        log(f"Shard {shard_idx}/{len(shards)} ({sp.name}) cells={n:,}")

        if chunk_size and chunk_size > 0 and n > chunk_size:
            # Chunked decode
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                chunk_raw = raw[start:end]
                decoded = decode_bitpack_chunk(chunk_raw)
                decoded_mm[cursor + start : cursor + end] = decoded
                if flat_mm is not None:
                    flat_mm[cursor + start : cursor + end] = decoded.reshape(-1, 64) > 0
        else:
            decoded = decode_bitpack_chunk(raw)
            decoded_mm[cursor : cursor + n] = decoded
            if flat_mm is not None:
                flat_mm[cursor : cursor + n] = decoded.reshape(-1, 64) > 0

        # Labels slice copy (only if assignments provided)
        if labels_mm is not None and labels_src is not None:
            hi = min(cursor + n, labels_src.shape[0])
            labels_mm[cursor:hi] = labels_src[cursor:hi].astype(np.uint16)
            if hi < cursor + n:
                # Fill out-of-range with sentinel (65535)
                labels_mm[hi : cursor + n] = 65535

        cursor += n
        log(f"Progress: {cursor:,}/{total_cells:,} ({cursor / total_cells:.2%})")

    # Flush
    decoded_mm.flush()
    if flat_mm is not None:
        flat_mm.flush()
    if labels_mm is not None:
        labels_mm.flush()
    log("Decoding complete. Files written:")

    log(f" - {cells_out}  (uint8, shape=(N,8,8))")
    if write_flat_bool:
        log(f" - {flat_out}  (bool, shape=(N,64))")
    if labels_src is not None:
        log(f" - {labels_out} (uint16 labels)")

    # Metadata
    if write_metadata:
        meta = {
            "total_cells": total_cells,
            "cells_uint8": cells_out.name,
            "has_flat_bool": bool(write_flat_bool),
            "flat_bool_file": flat_out.name if write_flat_bool else None,
            "has_labels": labels_src is not None,
            "labels_file": labels_out.name if labels_src is not None else None,
            "bitpack_shards": [p.name for p in shards],
            "chunk_size": chunk_size,
            "version": 1,
        }
        with meta_out.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        log(f" - {meta_out} (metadata)")

    # Verification
    if verify:
        log("Running verification sample...")
        import random

        sample_indices = random.sample(range(total_cells), k=min(5, total_cells))
        fail = False
        for idx in sample_indices:
            cell = decoded_mm[idx]
            if cell.shape != (8, 8):
                log(f"[verify][fail] Shape mismatch at {idx}: {cell.shape}")
                fail = True
            if not (
                np.array_equal(cell, 0 * cell) or np.all((cell == 0) | (cell == 255))
            ):
                log(f"[verify][warn] Non-binary values detected at {idx}")
        if not fail:
            log("[verify] Sample checks passed.")

    log("DONE.")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Consolidate bitpacked cell shards into a single uint8 memmap."
    )
    p.add_argument(
        "--cells-dir",
        type=Path,
        required=True,
        help="Directory containing cells_bitpack_shard_*.npy files.",
    )
    p.add_argument(
        "--shard-pattern",
        type=str,
        default="cells_bitpack_shard_*.npy",
        help="Glob pattern to match shard files.",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for consolidated artifacts.",
    )
    p.add_argument(
        "--assign-dense",
        type=Path,
        default=None,
        help="Optional dense assignment array (.npy) to write labels_uint16.npy aligned.",
    )
    p.add_argument(
        "--flat-bool",
        action="store_true",
        help="Also write a (N,64) boolean flat array (cells_bool.npy).",
    )
    p.add_argument(
        "--chunk",
        type=int,
        default=0,
        help="Chunk size for intra-shard decode (0 = decode whole shard at once).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files if present.",
    )
    p.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip writing metadata.json.",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="Perform small random verification samples after writing.",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        consolidate(
            cells_dir=args.cells_dir,
            shard_pattern=args.shard_pattern,
            out_dir=args.out,
            assign_dense=args.assign_dense,
            write_flat_bool=args.flat_bool,
            chunk_size=int(args.chunk),
            overwrite=bool(args.overwrite),
            write_metadata=not args.no_metadata,
            verify=bool(args.verify),
        )
    except Exception as e:
        log(f"FATAL: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
