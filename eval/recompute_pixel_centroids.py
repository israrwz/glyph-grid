#!/usr/bin/env python3
"""
Recompute Pixel-Space Primitive Centroids (Path A)

Purpose:
  Given:
    - A directory of 8x8 cell bitmaps (consolidated `cells.npy` or sharded `shard_*.npy`)
    - An assignments file mapping cell_id -> primitive_id
  This script recomputes the empirical pixel-space mean (prototype) for each primitive
  class after (or before) training, producing an updated centroid matrix you can use
  for:
    * Visualization
    * Duplicate detection
    * Drift analysis versus the original K-Means centroids
    * Upstream refinement / audit (without yet changing labels)

Output:
  - A `.npy` file of shape (K, 64) float32 where each row is the mean (values in [0,1])
    of the flattened 8x8 binary cell bitmaps belonging to that primitive id.
  - A JSON stats file with counts and summary metrics.
  - (Optional) A visualization PNG (grid of thresholded centroid thumbnails).
  - (Optional) Duplicate / near-duplicate distance report (basic nearest neighbor stats).

Assumptions:
  - Primitive id 0 is reserved for the EMPTY cell (all zeros) unless overridden.
  - Cell bitmaps are binary (0 or 255) or 0/1; anything >0 is treated as foreground.
  - Assignments file may be Parquet (.parquet / .pq), JSONL (.jsonl), or CSV (.csv),
    matching the conventions in data/primitives.py.

Typical Usage:
  python -m eval.recompute_pixel_centroids \
      --cells-dir data/processed/cells \
      --assignments-file data/processed/primitive_assignments.parquet \
      --output-centroids assets/centroids/primitive_centroids_recomputed.npy \
      --output-stats assets/centroids/primitive_centroids_recomputed_stats.json \
      --viz-grid assets/centroids/primitive_centroids_recomputed_grid.png \
      --orig-centroids assets/centroids/primitive_centroids.npy

Optional Enhancements (future):
  - Merge candidate detection & mapping export.
  - Feature-space (embedding) prototype computation.
  - Incremental / streaming partial export in very large corpora.

Exit Codes:
  0 success
  1 argument / file errors
  2 runtime failure

Author: Project Path A refinement utility.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# Pandas optional (Parquet convenience)
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore

# Local import (CellSource) for efficient iteration
try:
    from data.primitives import CellSource  # type: ignore
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "Failed to import CellSource from data.primitives. Adjust PYTHONPATH or run from project root."
    ) from e


# ---------------------------------------------------------------------------
# Assignment Loading
# ---------------------------------------------------------------------------


def load_assignments(path: Path) -> Dict[int, int]:
    if not path.exists():
        raise FileNotFoundError(f"Assignments file not found: {path}")
    ext = path.suffix.lower()
    mapping: Dict[int, int] = {}
    if ext in (".parquet", ".pq"):
        if pd is None:
            raise RuntimeError("pandas is required for parquet assignments.")
        df = pd.read_parquet(path, columns=["cell_id", "primitive_id"])
        for row in df.itertuples(index=False):
            mapping[int(row.cell_id)] = int(row.primitive_id)
    elif ext in (".jsonl", ".json"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    mapping[int(rec["cell_id"])] = int(rec["primitive_id"])
                except Exception:
                    continue
    elif ext == ".csv":
        import csv

        with open(path, "r", newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for rec in rdr:
                try:
                    mapping[int(rec["cell_id"])] = int(rec["primitive_id"])
                except Exception:
                    continue
    else:
        raise ValueError(
            f"Unsupported assignments extension '{ext}' (expected parquet/jsonl/csv)."
        )
    if not mapping:
        raise RuntimeError("No assignments loaded (empty mapping).")
    return mapping


# ---------------------------------------------------------------------------
# Centroid Computation
# ---------------------------------------------------------------------------


def compute_centroids(
    cells_dir: Path,
    assignments_file: Path,
    k_total: Optional[int] = None,
    empty_id: int = 0,
    limit_cells: Optional[int] = None,
    progress_every: int = 500_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      centroids: (K,64) float32 means (0..1)
      counts:    (K,) int64 counts
    """
    mapping = load_assignments(assignments_file)
    max_pid = max(mapping.values())
    if k_total is None:
        k_total = max_pid + 1
    if max_pid >= k_total:
        raise ValueError(
            f"k_total={k_total} insufficient for max primitive id {max_pid}; "
            f"please provide --k-total > {max_pid}."
        )

    # Accumulators
    sums = np.zeros((k_total, 64), dtype=np.float64)
    counts = np.zeros(k_total, dtype=np.int64)

    source = CellSource(cells_dir)
    processed = 0
    for cell_id, cell in source.iter_cells():
        if limit_cells is not None and processed >= limit_cells:
            break
        pid = mapping.get(cell_id, None)
        if pid is None:
            processed += 1
            continue
        if pid < 0 or pid >= k_total:
            processed += 1
            continue
        # Normalize cell to binary 0/1
        if cell.ndim == 3 and cell.shape[0] == 1:
            cell = cell[0]
        vec = (cell > 0).astype(np.float32).reshape(64)
        sums[pid] += vec
        counts[pid] += 1
        processed += 1
        if progress_every and processed % progress_every == 0:
            print(
                f"[progress] processed={processed:,} "
                f"(covered cells={counts.sum():,}, unique classes seen={(counts > 0).sum():,})",
                file=sys.stderr,
            )

    # Final centroids
    centroids = np.zeros_like(sums, dtype=np.float32)
    nonzero = counts > 0
    centroids[nonzero] = (sums[nonzero].T / counts[nonzero]).T
    # Ensure empty id (if defined) stays all zeros (OPTIONAL)
    if 0 <= empty_id < k_total and counts[empty_id] > 0:
        # Keep the empirical empty row; if you want to force zeros uncomment:
        # centroids[empty_id].fill(0)
        pass

    return centroids, counts


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def nearest_neighbor_stats(centroids: np.ndarray, counts: np.ndarray, skip_id: int = 0):
    """
    Compute per-centroid nearest neighbor squared L2 distance (excluding self)
    and highlight possible duplicates (those below a heuristic threshold).
    """
    K = centroids.shape[0]
    if K <= 2:
        return {}
    C = centroids.copy()
    # Optionally skip empty row in duplicate logic by setting it to large sentinel
    mask = np.arange(K) != skip_id
    C_work = C[mask]
    if C_work.size == 0:
        return {}

    # Precompute norms
    norms = (C_work**2).sum(axis=1, keepdims=True)
    # Squared distance matrix
    D = norms + norms.T - 2 * (C_work @ C_work.T)
    np.fill_diagonal(D, np.inf)

    nn_dist = D.min(axis=1)
    # Heuristic threshold: distances close to 0 => near duplicates
    # Max possible squared distance between two binary 64D vectors is 64 (if disjoint)
    # We'll mark those below 0.5 as suspicious (tunable).
    duplicate_candidates = int((nn_dist < 0.5).sum())

    # Map back to original centroid indices
    original_indices = np.arange(K)[mask]
    stats = []
    for i, dist in enumerate(nn_dist):
        stats.append(
            {
                "primitive_id": int(original_indices[i]),
                "nn_squared_l2": float(dist),
                "count": int(counts[original_indices[i]]),
            }
        )

    return {
        "duplicate_candidate_count": duplicate_candidates,
        "nn_stats": stats,
        "threshold_note": "primitive ids with nn_squared_l2 < 0.5 are potential duplicates (tune threshold after inspecting histogram)",
    }


def compare_with_original(
    new_centroids: np.ndarray, orig_centroids_path: Optional[Path]
) -> Optional[Dict[str, float]]:
    if not orig_centroids_path:
        return None
    if not orig_centroids_path.exists():
        print(
            f"[warn] Original centroids file not found: {orig_centroids_path}",
            file=sys.stderr,
        )
        return None
    try:
        orig = np.load(orig_centroids_path)
    except Exception as e:
        print(f"[warn] Failed loading original centroids: {e}", file=sys.stderr)
        return None
    if orig.shape != new_centroids.shape:
        print(
            f"[warn] Shape mismatch original {orig.shape} vs new {new_centroids.shape}",
            file=sys.stderr,
        )
        return None
    diffs = new_centroids - orig
    l2_per_row = np.sqrt((diffs**2).sum(axis=1))
    return {
        "mean_l2_row_diff": float(l2_per_row.mean()),
        "median_l2_row_diff": float(np.median(l2_per_row)),
        "max_l2_row_diff": float(l2_per_row.max()),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def save_centroid_grid_png(
    centroids: np.ndarray,
    out_path: Path,
    cols: int = 32,
    threshold: float = 0.5,
    scale: int = 16,
    skip_empty_row: bool = False,
):
    """
    Render centroids as a grid of thumbnails (binarized for clarity).
    """
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        print("[warn] pillow not installed; skipping viz grid.", file=sys.stderr)
        return
    K = centroids.shape[0]
    rows = math.ceil(K / cols)
    cell_pix = 8
    W = cols * cell_pix
    H = rows * cell_pix
    # Binarize
    bin_c = (centroids >= threshold).astype(np.uint8)
    canvas = np.zeros((H, W), dtype=np.uint8)
    for idx in range(K):
        if skip_empty_row and idx == 0:
            continue
        r = idx // cols
        c = idx % cols
        patch = (bin_c[idx].reshape(cell_pix, cell_pix) * 255).astype(np.uint8)
        canvas[r * cell_pix : (r + 1) * cell_pix, c * cell_pix : (c + 1) * cell_pix] = (
            patch
        )

    img = Image.fromarray(canvas, mode="L")
    if scale != 1:
        img = img.resize((W * scale, H * scale), Image.NEAREST)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"[info] Saved centroid grid PNG: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recompute pixel-space primitive centroids (Path A)."
    )
    p.add_argument(
        "--cells-dir",
        required=True,
        type=Path,
        help="Directory containing cells.npy or shard_*.npy",
    )
    p.add_argument(
        "--assignments-file",
        required=True,
        type=Path,
        help="Assignments mapping (cell_id -> primitive_id). Parquet / JSONL / CSV.",
    )
    p.add_argument(
        "--output-centroids",
        required=True,
        type=Path,
        help="Output .npy path for recomputed centroids (float32, shape (K,64)).",
    )
    p.add_argument(
        "--output-stats",
        type=Path,
        default=None,
        help="Optional JSON stats path (if omitted, derives from centroids filename).",
    )
    p.add_argument(
        "--viz-grid",
        type=Path,
        default=None,
        help="Optional PNG path for centroid thumbnails (binarized).",
    )
    p.add_argument(
        "--k-total",
        type=int,
        default=None,
        help="Total number of primitive classes (including empty). "
        "If omitted, inferred as (max primitive_id + 1).",
    )
    p.add_argument(
        "--empty-id",
        type=int,
        default=0,
        help="Primitive id reserved for empty cell (kept as-is).",
    )
    p.add_argument(
        "--limit-cells",
        type=int,
        default=None,
        help="Debug: limit number of cell records processed.",
    )
    p.add_argument(
        "--orig-centroids",
        type=Path,
        default=None,
        help="Optional original centroid file to compare divergence.",
    )
    p.add_argument(
        "--nn-stats",
        action="store_true",
        help="Compute nearest-neighbor duplicate candidate statistics.",
    )
    p.add_argument(
        "--viz-threshold",
        type=float,
        default=0.5,
        help="Threshold for binarizing centroid visualization.",
    )
    p.add_argument(
        "--viz-scale",
        type=int,
        default=16,
        help="Scale factor for visualization grid thumbnails.",
    )
    p.add_argument(
        "--skip-empty-in-viz",
        action="store_true",
        help="Do not render centroid 0 in visualization grid.",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=500_000,
        help="Print progress after this many processed cells (0 disables).",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    try:
        centroids, counts = compute_centroids(
            cells_dir=args.cells_dir,
            assignments_file=args.assignments_file,
            k_total=args.k_total,
            empty_id=args.empty_id,
            limit_cells=args.limit_cells,
            progress_every=args.progress_every,
        )
    except Exception as e:
        print(f"[error] Failed computing centroids: {e}", file=sys.stderr)
        return 2

    # Save centroids
    out_path = args.output_centroids
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, centroids.astype(np.float32), allow_pickle=False)
    print(
        f"[info] Saved recomputed centroids: {out_path} (shape={centroids.shape}, total_cells={int(counts.sum()):,})"
    )

    # Build stats
    occupied = int((counts > 0).sum())
    stats = {
        "k_total": int(centroids.shape[0]),
        "occupied_classes": occupied,
        "empty_id": int(args.empty_id),
        "total_cells_counted": int(counts.sum()),
        "counts": counts.tolist(),
        "mean_active_pixels_per_centroid": float(
            (centroids.sum(axis=1).mean()) if occupied else 0.0
        ),
    }

    # Optional original centroid comparison
    diverge = compare_with_original(centroids, args.orig_centroids)
    if diverge:
        stats["orig_comparison"] = diverge

    # Optional NN stats
    if args.nn_stats:
        stats["nearest_neighbor"] = nearest_neighbor_stats(
            centroids, counts, skip_id=args.empty_id
        )

    # Stats output
    stats_path = (
        args.output_stats if args.output_stats else out_path.with_suffix(".stats.json")
    )
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[info] Wrote stats JSON: {stats_path}")

    # Visualization
    if args.viz_grid:
        save_centroid_grid_png(
            centroids,
            args.viz_grid,
            cols=32,
            threshold=args.viz_threshold,
            scale=args.viz_scale,
            skip_empty_row=args.skip_empty_in_viz,
        )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print("\n[info] Aborted by user.", file=sys.stderr)
        sys.exit(130)
