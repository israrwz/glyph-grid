#!/usr/bin/env python3
"""
Recompute Primitive Centroids & Assignments (Corrected Decode Orientation)

Purpose
-------
After fixing the bit-unpack orientation (little‑endian per byte) for 8x8 cell bitmasks,
previously generated primitive centroids and assignments may be in a "flipped" frame.
This script rebuilds (1) the primitive vocabulary (centroids) and (2) per‑cell primitive
assignments from the original *bitpacked* cell shards using the corrected decode logic.

Key Outputs
-----------
1. Centroids (with EMPTY row optionally prepended):
     assets/centroids/primitive_centroids_rebuilt.npy
2. Assignments parquet:
     data/processed/primitive_assignments_rebuilt.parquet
3. Dense assignments array (uint16):
     data/processed/primitive_assignments_rebuilt_dense_uint16.npy
4. (Optional) A verification JSON summary of centroid orientation sanity.

What It Does
------------
1. Loads all bitpacked shards (cells_bitpack_shard_*.npy).
2. Decodes them with corrected little-endian bit order.
3. Samples non-empty cells (reservoir) for K-Means (K = PRIMITIVE_COUNT - 1).
4. Runs MiniBatchKMeans (if scikit-learn available) or full KMeans fallback.
5. Saves centroids with EMPTY (all zeros) inserted at index 0.
6. Assigns each cell (including empties) to nearest centroid (EMPTY id=0 for all-zero cells).
7. Writes parquet + dense mapping for downstream fast loading.
8. Optional: builds quick orientation diagnostics (counts of asymmetric vs symmetric cells)
   to detect lingering flip issues.

Assumptions
-----------
- Bitpacked cell shards path: data/processed/cells_bitpack
- Each cell is represented by one uint64; bit index (r*8 + c) with bit 0 = leftmost pixel (after fix).
- K (primitive_count) default 1024 (0=EMPTY).
- Empties: all-zero 8x8 decoded cells.

Usage
-----
  python -m data.recompute_primitives \
      --cells-dir data/processed/cells_bitpack \
      --out-centroids assets/centroids/primitive_centroids_rebuilt.npy \
      --assign-parquet data/processed/primitive_assignments_rebuilt.parquet \
      --assign-dense data/processed/primitive_assignments_rebuilt_dense_uint16.npy \
      --primitive-count 1024 \
      --sample-size 500000 \
      --seed 42 \
      --verify

If you are satisfied with the result you can replace the "official" centroid file:
  cp assets/centroids/primitive_centroids_rebuilt.npy assets/centroids/primitive_centroids.npy
and similarly swap the assignments in configs.

Performance Notes
-----------------
- Decoding all shards vectorized; memory footprint roughly (#cells * 64 bytes) transient
  if you decode whole shards at once. If memory constrained, use --chunk-decode.
- For large corpora consider smaller --sample-size first, inspect results, then upscale.

Safety
------
Script does NOT overwrite existing outputs unless --overwrite is passed.

"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Optional dependencies
try:
    from sklearn.cluster import MiniBatchKMeans, KMeans
except Exception:  # pragma: no cover
    MiniBatchKMeans = None  # type: ignore
    KMeans = None  # type: ignore

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover
    pa = None  # type: ignore
    pq = None  # type: ignore


def log(msg: str) -> None:
    print(f"[recompute] {msg}", file=sys.stderr, flush=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recompute primitive centroids & assignments with corrected decode."
    )
    p.add_argument(
        "--cells-dir",
        type=Path,
        default=Path("data/processed/cells_bitpack"),
        help="Directory containing cells_bitpack_shard_*.npy",
    )
    p.add_argument(
        "--out-centroids",
        type=Path,
        default=Path("assets/centroids/primitive_centroids_rebuilt.npy"),
        help="Output centroid .npy path (with EMPTY row 0).",
    )
    p.add_argument(
        "--assign-parquet",
        type=Path,
        default=Path("data/processed/primitive_assignments_rebuilt.parquet"),
        help="Output assignments parquet path (cell_id, primitive_id).",
    )
    p.add_argument(
        "--assign-dense",
        type=Path,
        default=Path("data/processed/primitive_assignments_rebuilt_dense_uint16.npy"),
        help="Dense uint16 assignment array path (index=cell_id).",
    )
    p.add_argument(
        "--primitive-count",
        type=int,
        default=1024,
        help="Total primitive vocabulary size including EMPTY (default 1024).",
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=500_000,
        help="Non-empty cell samples for K-Means (pre-EMPTY).",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling and K-Means."
    )
    p.add_argument(
        "--chunk-decode",
        type=int,
        default=0,
        help="If >0, decode shards in chunks of this many cells to reduce peak RAM.",
    )
    p.add_argument(
        "--max-shards",
        type=int,
        default=0,
        help="Optional limit on number of shards processed (debug). 0 = all.",
    )
    p.add_argument(
        "--force-full-kmeans",
        action="store_true",
        help="Force full KMeans even if MiniBatchKMeans is available.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing outputs.",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="Write verification JSON about orientation & centroid stats.",
    )
    return p.parse_args(argv)


# --------------------------------------------------------------------------------------
# Bitpack Decode (Corrected)
# --------------------------------------------------------------------------------------


def decode_bitpack_uint64_vector(raw: np.ndarray) -> np.ndarray:
    """
    raw: (N,) uint64
    Returns (N,8,8) uint8 with values {0,255} using little-endian bit significance.
    """
    bytes_view = raw.view(np.uint8).reshape(-1, 8)  # (N,8 bytes)
    # unpackbits returns MSB->LSB within each byte ⇒ reverse per-byte
    bits_msb = np.unpackbits(bytes_view, axis=1)  # (N,64)
    bits_le = bits_msb.reshape(-1, 8, 8)[:, :, ::-1].reshape(-1, 64)
    return (bits_le.reshape(-1, 8, 8) * 255).astype(np.uint8)


# --------------------------------------------------------------------------------------
# Reservoir Sampling for K-Means
# --------------------------------------------------------------------------------------


def reservoir_sample_nonempty(
    shard_iter,
    sample_size: int,
    seed: int,
    chunk_decode: int = 0,
) -> np.ndarray:
    """
    Iterate decoded shard batches, performing reservoir sampling on flattened non-empty cells.
    Returns (M,64) float32 normalized {0,1}.
    """
    rng = np.random.default_rng(seed)
    reservoir: List[np.ndarray] = []
    seen = 0

    def consider(batch_cells: np.ndarray):
        nonlocal reservoir, seen
        # batch_cells: (B,8,8)
        flat = batch_cells.reshape(batch_cells.shape[0], 64)
        # mask non-empty
        mask = flat.any(axis=1)
        flat_nonempty = flat[mask]
        for row in flat_nonempty:
            if len(reservoir) < sample_size:
                reservoir.append(row.copy())
            else:
                j = rng.integers(0, seen + 1)
                if j < sample_size:
                    reservoir[j] = row.copy()
            seen += 1

    for raw in shard_iter:
        if chunk_decode and chunk_decode > 0 and len(raw) > chunk_decode:
            for start in range(0, len(raw), chunk_decode):
                end = min(start + chunk_decode, len(raw))
                decoded = decode_bitpack_uint64_vector(raw[start:end])
                consider(decoded)
        else:
            decoded = decode_bitpack_uint64_vector(raw)
            consider(decoded)

    if not reservoir:
        raise RuntimeError("No non-empty cells sampled.")
    arr = np.stack(reservoir, axis=0).astype(np.float32)
    if arr.max() > 1.0:
        arr /= 255.0
    return arr


# --------------------------------------------------------------------------------------
# K-Means (MiniBatch or full)
# --------------------------------------------------------------------------------------


def run_kmeans(
    samples: np.ndarray, k: int, seed: int, use_minibatch: bool
) -> np.ndarray:
    if samples.ndim != 2 or samples.shape[1] != 64:
        raise ValueError("samples must be (N,64)")
    if KMeans is None:
        raise RuntimeError("scikit-learn not installed; cannot run K-Means.")
    if use_minibatch and MiniBatchKMeans is not None:
        log(f"MiniBatchKMeans: k={k}, samples={len(samples):,}")
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            batch_size=4096,
            max_iter=100,
            reassignment_ratio=0.01,
            n_init="auto",
        )
    else:
        log(f"Full KMeans: k={k}, samples={len(samples):,}")
        model = KMeans(n_clusters=k, random_state=seed, max_iter=300, n_init="auto")
    model.fit(samples)
    return model.cluster_centers_.astype(np.float32)


# --------------------------------------------------------------------------------------
# Assign Primitives
# --------------------------------------------------------------------------------------


def assign_cells_all(
    cells_uint64_vector: np.ndarray,
    centroids_no_empty: np.ndarray,
    empty_id: int,
    chunk: int = 0,
) -> np.ndarray:
    """
    Assign every cell id:
      - EMPTY if fully zero
      - else nearest centroid (L2)
    Returns (N,) int32 primitive ids.
    """
    N = len(cells_uint64_vector)
    K = centroids_no_empty.shape[0]
    assignments = np.zeros(N, dtype=np.int32)

    # Precompute centroid norms for distance speed
    # centroids_no_empty in [0,1]
    centroids_flat = centroids_no_empty  # (K,64)
    c2 = np.einsum("ij,ij->i", centroids_flat, centroids_flat)

    def process(decoded_batch: np.ndarray, offset: int):
        flat = (
            decoded_batch.reshape(decoded_batch.shape[0], 64).astype(np.float32)
        ) / 255.0
        # Mask empties
        mask_nonempty = flat.any(axis=1)
        # Distances only for non-empty subset
        idxs = np.where(mask_nonempty)[0]
        if idxs.size == 0:
            return
        sub = flat[idxs]
        # Compute distances: ||x - c||^2 = x^2 -2x·c + c^2
        x2 = np.einsum("ij,ij->i", sub, sub)
        dots = sub @ centroids_flat.T  # (B,K)
        dists = x2[:, None] - 2 * dots + c2[None, :]
        nearest = np.argmin(dists, axis=1)
        assignments[offset + idxs] = nearest + 1  # shift by +1 to account for EMPTY=0

    if chunk and chunk > 0:
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            decoded = decode_bitpack_uint64_vector(cells_uint64_vector[start:end])
            process(decoded, start)
    else:
        decoded = decode_bitpack_uint64_vector(cells_uint64_vector)
        process(decoded, 0)
    return assignments


# --------------------------------------------------------------------------------------
# Parquet & Dense Writers
# --------------------------------------------------------------------------------------


def write_parquet(assignments: np.ndarray, out_path: Path, overwrite: bool):
    if pq is None or pa is None:
        raise RuntimeError("pyarrow is required to write parquet.")
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists (use --overwrite).")
    cell_ids = np.arange(len(assignments), dtype=np.int64)
    tbl = pa.Table.from_arrays(
        [pa.array(cell_ids), pa.array(assignments.astype(np.int32))],
        names=["cell_id", "primitive_id"],
    )
    pq.write_table(tbl, out_path)
    log(f"Wrote assignments parquet: {out_path} rows={len(assignments):,}")


def write_dense(assignments: np.ndarray, out_path: Path, overwrite: bool):
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists (use --overwrite).")
    np.save(out_path, assignments.astype(np.uint16))
    log(f"Wrote dense assignments: {out_path} shape={assignments.shape}")


# --------------------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------------------


def verify_orientation(centroids_with_empty: np.ndarray, out_json: Path):
    """
    Simple heuristic: count how many centroids are asymmetric under horizontal flip.
    If orientation got accidentally flipped mid-process, large symmetric ratio might appear.
    """
    cents = centroids_with_empty[1:]  # exclude EMPTY
    resh = cents.reshape(-1, 8, 8)
    flipped = resh[:, :, ::-1]
    eq = np.all(resh == flipped, axis=(1, 2))
    symmetric = int(eq.sum())
    total = len(resh)
    data = {
        "total_centroids": total,
        "symmetric_count": symmetric,
        "symmetric_ratio": symmetric / max(1, total),
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    log(f"Verification summary: {data}")


# --------------------------------------------------------------------------------------
# Main Orchestration
# --------------------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # Safety checks
    if not args.cells_dir.exists():
        log(f"Cells dir not found: {args.cells_dir}")
        return 2

    if args.out_centroids.exists() and not args.overwrite:
        log(f"Centroids file exists: {args.out_centroids} (use --overwrite).")
        return 3
    if args.assign_parquet.exists() and not args.overwrite:
        log(f"Assignments parquet exists: {args.assign_parquet} (use --overwrite).")
        return 4
    if args.assign_dense.exists() and not args.overwrite:
        log(f"Dense assignment file exists: {args.assign_dense} (use --overwrite).")
        return 5

    # Load shard uint64 arrays
    shard_paths = sorted(args.cells_dir.glob("cells_bitpack_shard_*.npy"))
    if not shard_paths:
        log("No bitpack shards found.")
        return 6
    if args.max_shards > 0:
        shard_paths = shard_paths[: args.max_shards]
        log(f"[debug] Limiting to first {len(shard_paths)} shard(s)")

    log(f"Discovered {len(shard_paths)} shard(s). Sampling for K-Means...")

    def shard_uint64_iter():
        for sp in shard_paths:
            arr = np.load(sp, mmap_mode="r")
            yield arr

    # Reservoir sample for K-Means
    samples = reservoir_sample_nonempty(
        (arr for arr in shard_uint64_iter()),
        sample_size=args.sample_size,
        seed=args.seed,
        chunk_decode=args.chunk_decode,
    )
    log(f"Sampled non-empty cells: {len(samples):,}")

    k_no_empty = args.primitive_count - 1
    use_minibatch = not args.force_full_kmeans
    centroids_no_empty = run_kmeans(samples, k_no_empty, args.seed, use_minibatch)

    # Prepend EMPTY centroid (all zeros)
    empty_row = np.zeros((1, 64), dtype=np.float32)
    centroids_with_empty = np.vstack([empty_row, centroids_no_empty])
    args.out_centroids.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_centroids, centroids_with_empty)
    log(f"Wrote centroids: {args.out_centroids} shape={centroids_with_empty.shape}")

    # Full assignments
    log("Assigning all cells...")
    # Concatenate all shard uint64 vectors into one big array (careful with RAM).
    all_masks = np.concatenate([np.load(sp, mmap_mode="r") for sp in shard_paths])
    assignments = assign_cells_all(
        all_masks,
        centroids_no_empty,
        empty_id=0,
        chunk=args.chunk_decode,
    )
    # Write parquet + dense
    args.assign_parquet.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(assignments, args.assign_parquet, overwrite=args.overwrite)
    write_dense(assignments, args.assign_dense, overwrite=args.overwrite)

    # Optional verification
    if args.verify:
        verify_orientation(
            centroids_with_empty,
            args.out_centroids.with_suffix(".verify.json"),
        )

    log("DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
