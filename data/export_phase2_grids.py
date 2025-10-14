#!/usr/bin/env python3
"""
Phase 2 Grid Export Script
==========================

Purpose
-------
Generate per‑glyph primitive ID grids (16x16, dtype=uint16) required for Phase 2
glyph classification. The grids are derived from Phase 1 primitive assignments
(cell_id -> primitive_id) together with the cell manifest (cell_id -> glyph_id,row,col).

Outputs (default layout):
  <out_dir>/grids/<glyph_id>.u16          # Raw binary uint16 array (256 entries) row-major
  <out_dir>/grids/<glyph_id>.npy          # (optional) NumPy .npy version (if --npy also set)
  <out_dir>/label_map.json                # { "label_string": class_index, ... } (sorted)
  <out_dir>/splits/phase2_train_ids.txt   # glyph_id per line (if split inference enabled)
  <out_dir>/splits/phase2_val_ids.txt
  <out_dir>/splits/phase2_test_ids.txt
  <out_dir>/stats/phase2_grid_stats.json  # summary statistics (primitive freq, empties, etc.)

Two Modes of Grid Construction
------------------------------
1. From Assignments + Manifest (default, fast):
   - Requires:
       --cells-dir            (contains shard_*.npy + manifest.jsonl + phase1 split *.txt)
       --assignments          (Parquet or JSONL: cell_id, primitive_id)
   - Simply scatters primitive_id values into a 16x16 array for each glyph.

2. From Model Inference (future extension; NOT implemented here):
   - Would load Phase 1 checkpoint and rerun CNN; placeholder left for extension.

Label Mapping
-------------
If you provide --chars-csv (default: dataset/chars.csv) with a 'label' column,
a label_map.json is produced, assigning contiguous indices based on sorted label strings.
If omitted, a trivial identity mapping over encountered glyph_ids is emitted (not ideal
for training; prefer providing chars.csv).

Split Inference
---------------
If Phase 1 produced cell split files (phase1_train_cells.txt, etc.) we can infer
glyph-level splits by collecting the set of glyph_ids whose cells appear in each
split. When a glyph appears in multiple splits (edge case), precedence is:
  train > val > test  (first file processed wins; configurable later).
Alternatively, you can provide explicit glyph split files via:
  --train-glyph-ids, --val-glyph-ids, --test-glyph-ids

Primitive Frequency & Grid Stats
--------------------------------
A stats JSON is written summarizing:
  - glyph_count
  - primitive_vocab_size (max primitive id + 1)
  - mean / std / median non-empty cells per glyph
  - global primitive histogram
  - top_k primitives (by frequency)
  - per-glyph empty fraction summary (min/mean/max)

Performance Considerations
--------------------------
The script streams manifest + assignments without loading all cell bitmaps.
Assignments are loaded fully into memory (one int per cell) for fast scatter;
for very large corpora (> tens of millions cells) a chunked streaming variant
may be added later.

File Formats
------------
Raw .u16 format:
  - Exactly 256 * 2 bytes = 512 bytes per glyph.
  - Row-major order (r=0..15, c=0..15).
  - Easier and faster to memory-map than .npy in bulk consumption code.
  - Use numpy.fromfile(path, dtype=np.uint16).reshape(16,16).

CLI Examples
------------
Export grids + label map (assignments already computed):

  python -m data.export_phase2_grids \
      --cells-dir data/processed/cells \
      --assignments data/processed/primitive_assignments.parquet \
      --out-dir data/processed \
      --chars-csv dataset/chars.csv

Include NumPy .npy alongside .u16:

  python -m data.export_phase2_grids \
      --cells-dir data/processed/cells \
      --assignments data/processed/primitive_assignments.parquet \
      --out-dir data/processed \
      --write-npy

Dependencies
------------
  - Python 3.9+
  - numpy
  - (optional) pandas (for Parquet; falls back to JSONL)
  - (optional) tqdm for progress

License: Project root license.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd  # Optional; fall back gracefully if missing
except ImportError:
    pd = None  # type: ignore

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(x, **kwargs):
        return x


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class Args:
    cells_dir: Path
    assignments: Path
    out_dir: Path
    chars_csv: Optional[Path]
    train_glyph_ids: Optional[Path]
    val_glyph_ids: Optional[Path]
    test_glyph_ids: Optional[Path]
    infer_splits: bool
    write_npy: bool
    fail_on_missing: bool
    verbose: bool
    top_k_primitive_stats: int


# ---------------------------------------------------------------------------
# Loading Helpers
# ---------------------------------------------------------------------------


def load_assignments(path: Path) -> np.ndarray:
    """
    Returns an array A where A[cell_id] = primitive_id (int32).
    Supports Parquet (preferred) or JSONL.
    """
    if path.suffix.lower() in (".parquet", ".pq"):
        if pd is None:
            raise RuntimeError("pandas required to read parquet assignments.")
        df = pd.read_parquet(path, columns=["cell_id", "primitive_id"])
        max_id = int(df["cell_id"].max())
        arr = np.full(max_id + 1, -1, dtype=np.int32)
        arr[df["cell_id"].to_numpy()] = df["primitive_id"].to_numpy()
        return arr
    # JSONL fallback
    jsonl_path = path if path.suffix.lower() == ".jsonl" else path.with_suffix(".jsonl")
    max_cell = -1
    tmp: List[Tuple[int, int]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = int(rec["cell_id"])
            pid = int(rec["primitive_id"])
            tmp.append((cid, pid))
            if cid > max_cell:
                max_cell = cid
    arr = np.full(max_cell + 1, -1, dtype=np.int32)
    for cid, pid in tmp:
        arr[cid] = pid
    return arr


def iter_manifest(manifest_path: Path) -> Iterator[Dict]:
    """
    Yields dicts with keys: cell_id, glyph_id, row, col, (font_hash, is_empty) if present.
    Manifest is JSONL: one record per cell.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                yield rec
            except Exception:
                continue


def build_glyph_grids(
    manifest_path: Path,
    assignments: np.ndarray,
    *,
    verbose: bool = False,
) -> Dict[int, np.ndarray]:
    """
    Construct 16x16 primitive ID grids for each glyph.

    Returns:
        glyph_id -> np.ndarray shape (16,16) dtype=uint16
    """
    glyph_grids: Dict[int, np.ndarray] = {}
    filled_cells: Dict[int, int] = defaultdict(int)

    for rec in tqdm(iter_manifest(manifest_path), desc="scatter", disable=not verbose):
        cid = rec["cell_id"]
        gid = rec["glyph_id"]
        row = rec["row"]
        col = rec["col"]
        if cid >= len(assignments):
            continue
        pid = int(assignments[cid])
        if gid not in glyph_grids:
            glyph_grids[gid] = np.zeros((16, 16), dtype=np.uint16)
        # Defensive clamp
        if 0 <= row < 16 and 0 <= col < 16:
            glyph_grids[gid][row, col] = pid
            filled_cells[gid] += 1
    if verbose:
        print(f"[info] Built {len(glyph_grids)} glyph grids.", file=sys.stderr)
    return glyph_grids


def load_label_rows(chars_csv: Path) -> List[Dict]:
    """
    Reads chars.csv expecting a 'label' column.
    Returns list of row dicts.
    """
    rows: List[Dict] = []
    with open(chars_csv, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        if "label" not in header:
            raise ValueError("chars.csv missing 'label' column.")
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split(",")
            if len(cols) != len(header):
                continue
            rec = dict(zip(header, cols))
            rows.append(rec)
    return rows


def build_label_map(rows: List[Dict]) -> Dict[str, int]:
    labels = sorted({r["label"] for r in rows})
    return {lbl: i for i, lbl in enumerate(labels)}


# ---------------------------------------------------------------------------
# Split Inference
# ---------------------------------------------------------------------------


def read_id_list(path: Path) -> List[int]:
    out: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(int(line))
            except ValueError:
                continue
    return out


def infer_glyph_splits_from_cell_splits(
    cells_dir: Path,
    glyph_grids: Dict[int, np.ndarray],
    *,
    precedence: Tuple[str, str, str] = ("train", "val", "test"),
    verbose: bool = False,
) -> Dict[str, List[int]]:
    """
    Builds glyph-level splits by reading existing Phase 1 cell split files.
    Precedence: first split listing a glyph claims it.
    """
    split_files = {
        "train": cells_dir / "phase1_train_cells.txt",
        "val": cells_dir / "phase1_val_cells.txt",
        "test": cells_dir / "phase1_test_cells.txt",
    }
    glyph_split: Dict[int, str] = {}
    for split_name in precedence:
        path = split_files[split_name]
        if not path.exists():
            if verbose:
                print(f"[warn] Missing split file {path}, skipping.", file=sys.stderr)
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    cell_id = int(line)
                except ValueError:
                    continue
                # We need a reverse map (cell_id -> glyph_id) but we only have manifest streaming.
                # For efficiency, we reconstruct when manifest is large: here, we fallback to O(N) scan
                # once (cache), but as we already built all glyph grids we lost per-cell mapping.
                # Simpler approach: create a cached cell_id->glyph_id mapping the first time we need splits.
                # We'll build it now.
                pass  # We'll replace logic below after building mapping externally.
    # Revised approach: build mapping once from manifest:
    manifest_path = cells_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found for split inference: {manifest_path}"
        )
    cell_to_glyph: Dict[int, int] = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                cell_to_glyph[int(rec["cell_id"])] = int(rec["glyph_id"])
            except Exception:
                continue
    for split_name in precedence:
        path = split_files[split_name]
        if not path.exists():
            continue
        ids = read_id_list(path)
        for cid in ids:
            gid = cell_to_glyph.get(cid)
            if gid is None or gid not in glyph_grids:
                continue
            if gid not in glyph_split:
                glyph_split[gid] = split_name
    # Aggregate
    splits_out = {"train": [], "val": [], "test": []}
    for gid, split_name in glyph_split.items():
        splits_out[split_name].append(gid)
    if verbose:
        print(
            f"[info] Inferred glyph splits: "
            f"train={len(splits_out['train'])} val={len(splits_out['val'])} test={len(splits_out['test'])}",
            file=sys.stderr,
        )
    return splits_out


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def compute_stats(
    glyph_grids: Dict[int, np.ndarray],
    primitive_vocab_size_hint: Optional[int] = None,
    top_k: int = 20,
) -> Dict:
    primitive_counter = Counter()
    non_empty_counts: List[int] = []
    empty_fractions: List[float] = []
    for gid, grid in glyph_grids.items():
        vals = grid.ravel().tolist()
        primitive_counter.update(vals)
        non_empty = int(np.count_nonzero(grid))
        non_empty_counts.append(non_empty)
        empty_fractions.append(1.0 - non_empty / 256.0)
    total_glyphs = len(glyph_grids)
    if total_glyphs == 0:
        return {}
    primitive_vocab_size = (
        primitive_vocab_size_hint
        if primitive_vocab_size_hint is not None
        else max(primitive_counter.keys()) + 1
    )
    most_common = primitive_counter.most_common(top_k)
    stats = {
        "glyph_count": total_glyphs,
        "primitive_vocab_size": primitive_vocab_size,
        "total_primitive_assignments": int(sum(primitive_counter.values())),
        "non_empty_cells_mean": float(np.mean(non_empty_counts)),
        "non_empty_cells_std": float(np.std(non_empty_counts)),
        "non_empty_cells_median": float(np.median(non_empty_counts)),
        "empty_fraction_mean": float(np.mean(empty_fractions)),
        "empty_fraction_min": float(np.min(empty_fractions)),
        "empty_fraction_max": float(np.max(empty_fractions)),
        "top_primitives": [
            {"primitive_id": int(pid), "count": int(cnt)} for pid, cnt in most_common
        ],
    }
    return stats


# ---------------------------------------------------------------------------
# I/O Helpers
# ---------------------------------------------------------------------------


def write_grid_u16(path: Path, grid: np.ndarray):
    """
    Writes a 16x16 uint16 grid as raw .u16 (512 bytes).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if grid.dtype != np.uint16:
        grid = grid.astype(np.uint16)
    # Row-major contiguous
    grid.tofile(path)


def write_grid_npy(path: Path, grid: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, grid.astype(np.uint16))


def write_id_list(path: Path, ids: List[int]):
    path.parent.mkdir(parents=True, exist_ok=True)
    ids_sorted = sorted(ids)
    with open(path, "w", encoding="utf-8") as f:
        for gid in ids_sorted:
            f.write(f"{gid}\n")


# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------


def export_phase2_grids(args: Args) -> None:
    manifest_path = args.cells_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if not args.assignments.exists():
        raise FileNotFoundError(f"Assignments file missing: {args.assignments}")

    if args.verbose:
        print("[info] Loading assignments...", file=sys.stderr)
    assignments = load_assignments(args.assignments)
    if args.verbose:
        print(f"[info] Assignments loaded: length={len(assignments)}", file=sys.stderr)

    glyph_grids = build_glyph_grids(
        manifest_path,
        assignments,
        verbose=args.verbose,
    )

    # Write grids
    grids_dir = args.out_dir / "grids"
    written = 0
    for gid, grid in tqdm(
        glyph_grids.items(), desc="write_grids", disable=not args.verbose
    ):
        out_u16 = grids_dir / f"{gid}.u16"
        write_grid_u16(out_u16, grid)
        if args.write_npy:
            write_grid_npy(grids_dir / f"{gid}.npy", grid)
        written += 1
    if args.verbose:
        print(f"[info] Wrote {written} grids to {grids_dir}", file=sys.stderr)

    # Label map
    label_map_path = args.out_dir / "label_map.json"
    if args.chars_csv and args.chars_csv.exists():
        rows = load_label_rows(args.chars_csv)
        label_map = build_label_map(rows)
    else:
        # Fallback: map synthetic labels "gid_<id>" to indices (not ideal).
        label_map = {
            f"gid_{gid}": i for i, gid in enumerate(sorted(glyph_grids.keys()))
        }
        if args.verbose:
            print(
                "[warn] chars.csv not provided or missing; using synthetic gid_* labels.",
                file=sys.stderr,
            )
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    if args.verbose:
        print(f"[info] Wrote label map → {label_map_path}", file=sys.stderr)

    # Splits
    splits_dir = args.out_dir / "splits"
    if args.train_glyph_ids and args.train_glyph_ids.exists():
        train_ids = read_id_list(args.train_glyph_ids)
        val_ids = read_id_list(args.val_glyph_ids) if args.val_glyph_ids else []
        test_ids = read_id_list(args.test_glyph_ids) if args.test_glyph_ids else []
    elif args.infer_splits:
        splits = infer_glyph_splits_from_cell_splits(
            args.cells_dir, glyph_grids, verbose=args.verbose
        )
        train_ids = splits["train"]
        val_ids = splits["val"]
        test_ids = splits["test"]
    else:
        # All -> train
        train_ids = list(glyph_grids.keys())
        val_ids = []
        test_ids = []
        if args.verbose:
            print(
                "[warn] No splits provided; assigning all glyphs to train.",
                file=sys.stderr,
            )

    if train_ids:
        write_id_list(splits_dir / "phase2_train_ids.txt", train_ids)
    if val_ids:
        write_id_list(splits_dir / "phase2_val_ids.txt", val_ids)
    if test_ids:
        write_id_list(splits_dir / "phase2_test_ids.txt", test_ids)
    if args.verbose:
        print(
            f"[info] Split sizes glyph-level: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}",
            file=sys.stderr,
        )

    # Stats
    stats_dir = args.out_dir / "stats"
    stats = compute_stats(
        glyph_grids,
        primitive_vocab_size_hint=int(assignments.max()) + 1,
        top_k=args.top_k_primitive_stats,
    )
    stats_path = stats_dir / "phase2_grid_stats.json"
    stats_dir.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    if args.verbose:
        print(f"[info] Wrote stats → {stats_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> Args:
    p = argparse.ArgumentParser(
        description="Export Phase 2 primitive ID grids (16x16) from Phase 1 assignments."
    )
    p.add_argument(
        "--cells-dir",
        type=Path,
        required=True,
        help="Directory with cell shards + manifest.jsonl (from extract_cells).",
    )
    p.add_argument(
        "--assignments",
        type=Path,
        required=True,
        help="Primitive assignments file (Parquet or JSONL).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Root output directory (grids/, splits/, label_map.json, stats/).",
    )
    p.add_argument(
        "--chars-csv",
        type=Path,
        default=Path("dataset/chars.csv"),
        help="Optional chars.csv to derive label_map.json (must contain 'label').",
    )
    p.add_argument(
        "--train-glyph-ids",
        type=Path,
        default=None,
        help="Optional explicit glyph id list for train split (overrides inference).",
    )
    p.add_argument(
        "--val-glyph-ids",
        type=Path,
        default=None,
        help="Optional explicit glyph id list for val split.",
    )
    p.add_argument(
        "--test-glyph-ids",
        type=Path,
        default=None,
        help="Optional explicit glyph id list for test split.",
    )
    p.add_argument(
        "--infer-splits",
        action="store_true",
        help="Infer glyph splits from Phase 1 cell split files in cells_dir.",
    )
    p.add_argument(
        "--write-npy",
        action="store_true",
        help="Also write .npy versions of each grid (alongside .u16 raw).",
    )
    p.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Exit with error if any expected resource is missing instead of warning.",
    )
    p.add_argument(
        "--top-k-primitive-stats",
        type=int,
        default=20,
        help="Number of top primitives to record in stats JSON.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    args = p.parse_args(argv)

    return Args(
        cells_dir=args.cells_dir,
        assignments=args.assignments,
        out_dir=args.out_dir,
        chars_csv=args.chars_csv if args.chars_csv else None,
        train_glyph_ids=args.train_glyph_ids,
        val_glyph_ids=args.val_glyph_ids,
        test_glyph_ids=args.test_glyph_ids,
        infer_splits=args.infer_splits,
        write_npy=args.write_npy,
        fail_on_missing=args.fail_on_missing,
        verbose=args.verbose,
        top_k_primitive_stats=args.top_k_primitive_stats,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        export_phase2_grids(args)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
