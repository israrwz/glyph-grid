#!/usr/bin/env python3
"""
Primitive Vocabulary (Phase 1) Pipeline

Implements the workflow described in NEW_PLAN.md Sections 5 and 6.1:

Plan Recap (Section 6.1):
  - Sample ~1M non-empty cells (random across fonts).
  - Flatten each 8×8 mask → 64D vector.
  - K-Means (k = 1023) ignoring empty; assign cluster IDs 1..1023.
  - Class 0 reserved for empty cell.
  - Save prototype centroids for visualization.

This module provides a scaffold for:
  1. Streaming iteration over previously sliced 8x8 cell bitmaps.
  2. Random sampling / subsampling of non-empty cells for clustering.
  3. Running MiniBatch K-Means (default) or optional Faiss-based k-means if available.
  4. Persisting centroids (including an inserted zero-vector for EMPTY -> index 0).
  5. Assigning every non-empty cell to nearest centroid (primitive ID) and writing assignments.
  6. Reporting cluster population statistics / frequency buckets.

Assumptions / Expected Inputs (from earlier preprocessing steps):
  - Cell extraction already completed from 128x128 rasters:
      data/processed/cells/
         ├── cells.npy (optional consolidated array) OR
         ├── shard_*.npy (multiple shards) OR
         ├── cells.lmdb (future option)
         └── manifest.json (metadata about cell ids, ordering, glyph ids)
  - Each cell is stored as uint8 (0 or 255) or bool shape (8,8).
  - A companion mapping (e.g., cells_meta.parquet / JSONL) may store:
        cell_id, glyph_id, row, col, is_empty (or determinable on the fly).
  - Empty cell definition: all zeros (after binarization) → primitive ID 0.

Outputs:
  - assets/centroids/primitive_centroids.npy   (shape: (1024, 64); row 0 = zeros)
  - data/processed/primitive_assignments.parquet (cell_id -> primitive_id)  OR JSONL fallback
  - Optional: data/processed/primitive_stats.json

The code below is intentionally modular & documented for incremental filling.
Sections marked with TODO need actual data integration depending on chosen storage
format (LMDB, NumPy shards, Parquet, etc.).

Usage Examples (CLI):
  # Sample and run k-means (writes centroids)
  python -m data.primitives sample-and-cluster \\
      --cells-dir data/processed/cells \\
      --output-centroids assets/centroids/primitive_centroids.npy \\
      --sample-size 1000000 --k 1023

  # Assign full corpus to centroids
  python -m data.primitives assign \\
      --cells-dir data/processed/cells \\
      --centroids assets/centroids/primitive_centroids.npy \\
      --out-assignments data/processed/primitive_assignments.parquet

  # Cluster stats
  python -m data.primitives stats \\
      --assignments data/processed/primitive_assignments.parquet \\
      --k 1024

Dependencies:
  - numpy
  - scikit-learn (for MiniBatchKMeans) OR optional faiss (if installed)
  - pandas (optional; enables parquet output; falls back to JSONL if absent)

Reproducibility:
  - Fixed random seed applied to sampling & MiniBatchKMeans initialization.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.cluster import MiniBatchKMeans
except ImportError:  # pragma: no cover
    MiniBatchKMeans = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None  # type: ignore


# --------------------------------------------------------------------------------------
# Configuration Dataclass
# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class KMeansConfig:
    k: int = (
        1023  # Number of non-empty clusters (total primitives = k + 1 including EMPTY)
    )
    sample_size: int = 1_000_000
    seed: int = 42
    batch_size: int = 10_000  # MiniBatchKMeans batch size
    max_iter: int = 200
    reassignment_ratio: float = 0.01
    # When True, attempt to use Faiss if available (future extension).
    prefer_faiss: bool = False

    def total_vocab(self) -> int:
        return self.k + 1  # +1 for EMPTY cluster 0


# --------------------------------------------------------------------------------------
# Cell Access Layer (Scaffold)
# --------------------------------------------------------------------------------------


class CellSource:
    """
    Abstracts access to 8x8 cell bitmaps.

    Supports multiple backends (to be implemented as needed). Current scaffold
    provides in-memory / NumPy examples.

    A cell bitmap MUST be returned as a numpy array of shape (8, 8) with dtype uint8 or bool
    where foreground is >0 (binary form expected).
    """

    def __init__(self, root: Path):
        self.root = root
        if not root.exists():
            raise FileNotFoundError(f"Cells directory not found: {root}")

        # Discover available format(s)
        self._format = self._detect_format()

    def _detect_format(self) -> str:
        """
        Basic heuristics; extend as necessary.
        """
        if (self.root / "cells.npy").exists():
            return "consolidated_npy"
        shard_list = sorted(self.root.glob("shard_*.npy"))
        if shard_list:
            return "sharded_npy"
        bitpack_list = sorted(self.root.glob("cells_bitpack_shard_*.npy"))
        if bitpack_list:
            return "bitpack_sharded"
        # Placeholder for LMDB or other formats
        if (self.root / "cells.lmdb").exists():
            return "lmdb"  # TODO: implement LMDB reading
        raise RuntimeError(
            "Unable to detect cell storage format (expected cells.npy, shard_*.npy, cells_bitpack_shard_*.npy, or cells.lmdb)"
        )

    def iter_cells(self) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Yields (cell_id, bitmap). Order should be deterministic across runs for reproducibility
        provided the underlying storage is stable.

        NOTE: For very large datasets, prefer streaming. Here we load arrays shard-wise.
        """
        if self._format == "consolidated_npy":
            arr = np.load(self.root / "cells.npy", mmap_mode="r")
            # Expected shape: (N, 8, 8) or (N, 1, 8, 8)
            for cid in range(len(arr)):
                cell = arr[cid]
                if cell.ndim == 3 and cell.shape[0] == 1:
                    cell = cell[0]
                yield cid, cell
        elif self._format == "sharded_npy":
            shards = sorted(self.root.glob("shard_*.npy"))
            offset = 0
            for shard_path in shards:
                shard = np.load(shard_path, mmap_mode="r")
                for i in range(len(shard)):
                    cell = shard[i]
                    if cell.ndim == 3 and cell.shape[0] == 1:
                        cell = cell[0]
                    yield offset + i, cell
                offset += len(shard)
        elif self._format == "bitpack_sharded":
            # Bitpacked shards: uint64 array, one 64-bit mask per 8x8 cell.
            shards = sorted(self.root.glob("cells_bitpack_shard_*.npy"))
            offset = 0
            for shard_path in shards:
                masks = np.load(shard_path, mmap_mode="r")  # shape (N,)
                for i in range(len(masks)):
                    yield offset + i, unpack_bitcell(masks[i])
                offset += len(masks)
        elif self._format == "lmdb":
            # TODO: Implement LMDB reading logic
            raise NotImplementedError("LMDB backend not yet implemented.")
        else:  # pragma: no cover
            raise RuntimeError(f"Unsupported cell format: {self._format}")

    @staticmethod
    def is_empty(cell: np.ndarray) -> bool:
        # A cell is empty if all values are zero.
        return np.count_nonzero(cell) == 0


# --------------------------------------------------------------------------------------
# Sampling & Vectorization
# --------------------------------------------------------------------------------------


def unpack_bitcell(val: np.uint64) -> np.ndarray:
    """
    Unpack a uint64 bitmask into an (8,8) uint8 binary cell (0/255).
    Bit (r*8 + c) corresponds to pixel (r,c).
    """
    bits = int(val)
    out = np.zeros((8, 8), dtype=np.uint8)
    # Iterate over set bits only for efficiency
    while bits:
        lsb = bits & -bits
        idx = lsb.bit_length() - 1
        r, c = divmod(idx, 8)
        out[r, c] = 255
        bits ^= lsb
    return out


def flatten_cell(cell: np.ndarray) -> np.ndarray:
    """
    Convert an 8x8 binary/uint8 cell to a float32 vector of length 64.
    Values are normalized to {0.0, 1.0}.
    """
    if cell.shape != (8, 8):
        raise ValueError(f"Expected cell shape (8,8); got {cell.shape}")
    # Convert to float32 0/1
    if cell.dtype != np.uint8 and cell.dtype != np.bool_:
        cell = cell.astype(np.uint8)
    vec = (cell > 0).astype(np.float32).reshape(-1)  # shape (64,)
    return vec


def sample_non_empty_cells(
    source: CellSource,
    sample_size: int,
    seed: int,
    progress_interval: int = 250_000,
) -> np.ndarray:
    """
    Reservoir-sample up to sample_size non-empty cells uniformly at random across the corpus.
    Returns an array of shape (N, 64) where N <= sample_size.
    """
    random.seed(seed)
    np.random.seed(seed)

    reservoir: List[np.ndarray] = []
    total_non_empty = 0

    for cell_id, cell in source.iter_cells():
        if CellSource.is_empty(cell):
            continue

        vec = flatten_cell(cell)
        if len(reservoir) < sample_size:
            reservoir.append(vec)
        else:
            # Reservoir sampling replacement
            j = random.randint(0, total_non_empty)
            if j < sample_size:
                reservoir[j] = vec

        total_non_empty += 1
        if progress_interval and total_non_empty % progress_interval == 0:
            print(
                f"[sample] Seen non-empty: {total_non_empty:,} "
                f"- Current reservoir filled: {len(reservoir):,}",
                file=sys.stderr,
            )

    final = np.stack(reservoir, axis=0)
    print(
        f"[sample] Total non-empty seen: {total_non_empty:,}; Sampled: {final.shape[0]:,}",
        file=sys.stderr,
    )
    return final


# --------------------------------------------------------------------------------------
# K-Means (MiniBatch) Execution
# --------------------------------------------------------------------------------------


def run_kmeans(vectors: np.ndarray, cfg: KMeansConfig) -> np.ndarray:
    """
    Overcluster + refine strategy (default):
      1. Overcluster with MiniBatchKMeans (k_init = ceil(k_target * 1.15))
      2. Remove low-population and near-duplicate centroids
      3. If too many survivors -> prune by population
      4. If too few -> add back best remaining or (last resort) noisy copies
      5. Return exactly cfg.k centroids (float32, shape (k,64))

    Rationale: ensures final 1023 non-empty primitive classes (plus empty externally)
    are fully utilized and reduces wasted slots due to zero-pop or duplicate clusters.
    """
    if MiniBatchKMeans is None:
        raise ImportError(
            "scikit-learn is required for MiniBatchKMeans (pip install scikit-learn)."
        )

    import math

    k_target = cfg.k
    overcluster_margin = 0.15  # 15% headroom
    k_init = int(math.ceil(k_target * (1.0 + overcluster_margin)))
    dup_thresh = 0.22  # squared L2 duplicate threshold (tunable)
    # Minimum population threshold relative to expected cluster size
    expected = max(1, len(vectors) / max(1, k_init))
    min_pop_sample = max(10, int(0.2 * expected))  # at least 20% of expected size

    print(
        f"[kmeans] Overcluster pass: target={k_target} k_init={k_init} samples={len(vectors):,} "
        f"min_pop_sample={min_pop_sample}",
        file=sys.stderr,
    )

    mbk = MiniBatchKMeans(
        n_clusters=k_init,
        init="k-means++",
        batch_size=cfg.batch_size,
        max_iter=cfg.max_iter,
        reassignment_ratio=cfg.reassignment_ratio,
        random_state=cfg.seed,
        verbose=0,
    )
    start = time.time()
    labels = mbk.fit_predict(vectors)
    dt = time.time() - start
    centroids_full = mbk.cluster_centers_.astype(np.float32)
    counts = np.bincount(labels, minlength=k_init)

    print(
        f"[kmeans] Initial overcluster done in {dt:.2f}s | inertia={mbk.inertia_:.4f} "
        f"| zero-pop={int((counts == 0).sum())}",
        file=sys.stderr,
    )

    # Compute nearest-neighbor squared distances between centroids
    norms = (centroids_full**2).sum(axis=1, keepdims=True)
    d2 = norms + norms.T - 2 * (centroids_full @ centroids_full.T)
    np.fill_diagonal(d2, np.inf)
    nn_d2 = d2.min(axis=1)

    # Initial survivor mask: keep those with adequate population AND not near-duplicate
    survivor_mask = (counts >= min_pop_sample) & (nn_d2 > dup_thresh)

    # Guarantee at least some survivors even if thresholds too strict
    if survivor_mask.sum() == 0:
        print(
            "[kmeans][warn] Thresholds eliminated all clusters; relaxing criteria.",
            file=sys.stderr,
        )
        # Keep top k_target by population
        top_idx = np.argsort(counts)[-k_target:]
        survivor_mask[top_idx] = True

    survivors = np.where(survivor_mask)[0].tolist()

    # If still too many survivors, prune smallest populations until k_target
    if len(survivors) > k_target:
        survivors.sort(key=lambda i: counts[i])  # ascending pop
        survivors = survivors[-k_target:]

    # If too few, add back best remaining (by count) ignoring dup filter first
    if len(survivors) < k_target:
        deficit = k_target - len(survivors)
        remaining = [i for i in range(k_init) if i not in survivors and counts[i] > 0]
        remaining.sort(key=lambda i: (counts[i], -nn_d2[i]))  # prefer higher count
        add_list = remaining[-deficit:]
        survivors.extend(add_list)

    # Last resort: synthesize noisy copies if still short (should be rare)
    rng = np.random.default_rng(cfg.seed)
    while len(survivors) < k_target:
        base = rng.choice(survivors)
        noisy = centroids_full[base] + rng.normal(0, 0.02, size=centroids_full.shape[1])
        noisy = np.clip(noisy, 0.0, 1.0)
        centroids_full = np.vstack([centroids_full, noisy.astype(np.float32)])
        survivors.append(len(centroids_full) - 1)

    # Assemble final centroid set
    final_centroids = centroids_full[survivors][:k_target]

    # Sanity checks
    if final_centroids.shape[0] != k_target:
        print(
            f"[kmeans][warn] Final centroid count mismatch: {final_centroids.shape[0]} != {k_target}",
            file=sys.stderr,
        )

    if final_centroids.min() < -1e-3 or final_centroids.max() > 1.0 + 1e-3:
        print(
            "[kmeans][warn] Centroid range outside expected bounds "
            f"({final_centroids.min():.4f}, {final_centroids.max():.4f})",
            file=sys.stderr,
        )

    print(
        f"[kmeans] Refined centroids: kept={len(survivors)} "
        f"| duplicates_removed={(k_init - len(survivors)) - int((counts == 0).sum())} "
        f"| zero_pop={(counts == 0).sum()} "
        f"| final_k={final_centroids.shape[0]}",
        file=sys.stderr,
    )

    return final_centroids.astype(np.float32)


def save_centroids_with_empty(centroids: np.ndarray, out_path: Path):
    """
    Insert a zero-vector at index 0 for the EMPTY primitive and save (k+1,64) array.
    """
    if centroids.ndim != 2:
        raise ValueError("Centroids must be 2D.")
    if centroids.shape[1] != 64:
        raise ValueError("Expected centroid dimensionality 64.")
    zero_vec = np.zeros((1, centroids.shape[1]), dtype=np.float32)
    full = np.vstack([zero_vec, centroids])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, full)
    print(
        f"[centroids] Saved centroids including EMPTY to {out_path} (shape={full.shape})",
        file=sys.stderr,
    )


# --------------------------------------------------------------------------------------
# Assignment
# --------------------------------------------------------------------------------------


def assign_cells_to_centroids(
    source: CellSource,
    centroids_with_empty: np.ndarray,
    output_path: Path,
    batch_report: int = 500_000,
):
    """
    For every cell:
      - If empty -> primitive_id = 0
      - Else find nearest centroid among centroids_with_empty[1:]
    Writes a Parquet file (if pandas available) else JSONL with columns:
      cell_id, primitive_id
    """
    if centroids_with_empty.shape[0] < 2:
        raise ValueError(
            "Centroids array must include EMPTY + at least one non-empty centroid."
        )
    if centroids_with_empty.shape[1] != 64:
        raise ValueError("Centroid dimensionality mismatch (expected 64).")

    non_empty_centroids = centroids_with_empty[1:]  # shape (k,64)

    # Precompute squared norms for faster distance (optional optimization)
    centroid_norms = (non_empty_centroids**2).sum(axis=1)

    assignments: List[Tuple[int, int]] = []
    processed = 0
    non_empty_assigned = 0

    for cell_id, cell in source.iter_cells():
        if CellSource.is_empty(cell):
            assignments.append((cell_id, 0))
        else:
            vec = flatten_cell(cell)
            # Compute squared L2 distances: ||x - c||^2 = ||x||^2 - 2x·c + ||c||^2
            x_norm = (vec * vec).sum()
            dots = non_empty_centroids @ vec
            dists = x_norm - 2 * dots + centroid_norms
            nearest_index = int(np.argmin(dists)) + 1  # +1 to account for EMPTY at 0
            assignments.append((cell_id, nearest_index))
            non_empty_assigned += 1

        processed += 1
        if batch_report and processed % batch_report == 0:
            print(
                f"[assign] Processed {processed:,} cells "
                f"(non-empty assigned: {non_empty_assigned:,})",
                file=sys.stderr,
            )

    print(
        f"[assign] Finished. Total cells: {processed:,}, Non-empty: {non_empty_assigned:,}",
        file=sys.stderr,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if pd is not None and output_path.suffix in (".parquet", ".pq"):
        df = pd.DataFrame(assignments, columns=["cell_id", "primitive_id"])
        df.to_parquet(output_path, index=False)
        print(f"[assign] Wrote Parquet assignments to {output_path}", file=sys.stderr)
    else:
        # JSONL fallback
        jsonl_path = (
            output_path
            if output_path.suffix == ".jsonl"
            else output_path.with_suffix(".jsonl")
        )
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for cid, pid in assignments:
                f.write(json.dumps({"cell_id": cid, "primitive_id": pid}) + "\n")
        print(f"[assign] Wrote JSONL assignments to {jsonl_path}", file=sys.stderr)


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------


def load_assignments(path: Path) -> np.ndarray:
    """
    Load assignments into an array of primitive IDs aligned by cell_id index.
    """
    if pd is not None and path.suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
        # Expect columns: cell_id, primitive_id
        max_id = df["cell_id"].max()
        arr = np.full(max_id + 1, -1, dtype=np.int32)
        arr[df["cell_id"].to_numpy()] = df["primitive_id"].to_numpy()
        return arr
    # JSONL fallback
    jsonl_path = path if path.suffix == ".jsonl" else path.with_suffix(".jsonl")
    max_cell = -1
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cid = rec["cell_id"]
            records.append(rec)
            if cid > max_cell:
                max_cell = cid
    arr = np.full(max_cell + 1, -1, dtype=np.int32)
    for rec in records:
        arr[rec["cell_id"]] = rec["primitive_id"]
    return arr


def primitive_frequency_stats(assignments: np.ndarray, k_total: int) -> dict:
    """
    Compute frequency distribution and simple bucketed statistics.
    """
    counts = np.bincount(assignments[assignments >= 0], minlength=k_total)
    total = counts.sum()
    pct = counts / total
    # Frequency buckets (log-scale-ish): define boundaries relative to total
    bucket_edges = [0, 10, 100, 1_000, 10_000, 100_000, int(1e9)]
    buckets = []
    for lo, hi in zip(bucket_edges[:-1], bucket_edges[1:]):
        mask = (counts >= lo) & (counts < hi)
        bucket = {
            "range": [lo, hi],
            "cluster_count": int(mask.sum()),
            "population": int(counts[mask].sum()),
            "population_pct": float(counts[mask].sum() / total if total else 0),
        }
        buckets.append(bucket)

    return {
        "total_cells": int(total),
        "primitive_count": int(k_total),
        "counts": counts.tolist(),
        "pct": pct.tolist(),
        "buckets": buckets,
    }


def write_stats(stats: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[stats] Wrote primitive stats to {out_path}", file=sys.stderr)


# --------------------------------------------------------------------------------------
# CLI Interface
# --------------------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Primitive K-Means Pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    # sample-and-cluster
    sc = sub.add_parser(
        "sample-and-cluster", help="Sample non-empty cells and run k-means."
    )
    sc.add_argument("--cells-dir", required=True)
    sc.add_argument("--k", type=int, default=1023)
    sc.add_argument("--sample-size", type=int, default=1_000_000)
    sc.add_argument(
        "--sample-path",
        type=str,
        default="",
        help="Optional precomputed sample .npy (overrides on-the-fly sampling & --sample-size).",
    )
    sc.add_argument("--seed", type=int, default=42)
    sc.add_argument("--batch-size", type=int, default=10_000)
    sc.add_argument("--max-iter", type=int, default=200)
    sc.add_argument("--reassignment-ratio", type=float, default=0.01)
    sc.add_argument(
        "--output-centroids",
        required=True,
        help="Path for centroids .npy (includes EMPTY at index 0).",
    )

    # assign
    ap = sub.add_parser(
        "assign", help="Assign entire cell corpus to existing centroids."
    )
    ap.add_argument("--cells-dir", required=True)
    ap.add_argument(
        "--centroids", required=True, help="Centroids .npy including EMPTY row 0."
    )
    ap.add_argument(
        "--out-assignments",
        required=True,
        help="Output .parquet or .jsonl with (cell_id, primitive_id).",
    )

    # stats
    st = sub.add_parser(
        "stats", help="Compute frequency distribution over assignments."
    )
    st.add_argument("--assignments", required=True)
    st.add_argument("--k-total", type=int, default=1024)
    st.add_argument("--out", required=False, help="Optional JSON stats output path.")

    return p


def cmd_sample_and_cluster(args: argparse.Namespace):
    cfg = KMeansConfig(
        k=args.k,
        sample_size=args.sample_size,
        seed=args.seed,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        reassignment_ratio=args.reassignment_ratio,
    )

    source = CellSource(Path(args.cells_dir))
    if args.sample_path:
        sample_path = Path(args.sample_path)
        if not sample_path.exists():
            raise FileNotFoundError(f"--sample-path file not found: {sample_path}")
        vectors_raw = np.load(sample_path)
        # Accept (N,8,8) or (N,64)
        if vectors_raw.ndim == 3 and vectors_raw.shape[1:] == (8, 8):
            vectors = (vectors_raw.reshape(vectors_raw.shape[0], -1) > 0).astype(
                np.float32
            )
        elif vectors_raw.ndim == 2 and vectors_raw.shape[1] == 64:
            vectors = vectors_raw.astype(np.float32)
        else:
            raise ValueError(
                f"Unsupported sample array shape {vectors_raw.shape}; expected (N,8,8) or (N,64)."
            )
        print(
            f"[sample] Loaded precomputed sample from {sample_path} shape={vectors.shape}",
            file=sys.stderr,
        )
    else:
        if cfg.sample_size <= 0:
            # Sample size 0 => load ALL non-empty cells (streaming)
            collected: List[np.ndarray] = []
            for _cid, cell in source.iter_cells():
                if CellSource.is_empty(cell):
                    continue
                collected.append(flatten_cell(cell))
            vectors = np.stack(collected, axis=0)
            print(
                f"[sample] Loaded all non-empty cells: {vectors.shape[0]:,}",
                file=sys.stderr,
            )
        else:
            vectors = sample_non_empty_cells(source, cfg.sample_size, cfg.seed)
    centroids = run_kmeans(vectors, cfg)
    save_centroids_with_empty(centroids, Path(args.output_centroids))


def cmd_assign(args: argparse.Namespace):
    source = CellSource(Path(args.cells_dir))
    centroids = np.load(args.centroids)
    assign_cells_to_centroids(source, centroids, Path(args.out_assignments))


def cmd_stats(args: argparse.Namespace):
    assignments = load_assignments(Path(args.assignments))
    stats = primitive_frequency_stats(assignments, args.k_total)
    if args.out:
        write_stats(stats, Path(args.out))
    else:
        # Print summarized stats to stdout
        print(json.dumps(stats, indent=2))


def main(argv: Optional[Sequence[str]] = None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cmd = args.command
    if cmd == "sample-and-cluster":
        cmd_sample_and_cluster(args)
    elif cmd == "assign":
        cmd_assign(args)
    elif cmd == "stats":
        cmd_stats(args)
    else:  # pragma: no cover
        parser.error(f"Unknown command {cmd}")


if __name__ == "__main__":
    main()
