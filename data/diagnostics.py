#!/usr/bin/env python3
"""
diagnostics.py

Purpose:
  Provide integrity diagnostics for the glyph rasterization pipeline:
    - Sample metadata records (by random, interval, or explicit ids)
    - Verify raster PNG and grid NPY existence
    - Recompute occupancy grid from raster and compare with stored grid
    - Compute occupancy / density statistics
    - Cross-check diacritic flag consistency with ratio + (optional) advance_width from DB
    - Summarize discrepancies / anomalies

Usage examples:
  # Basic random sample of 20 glyphs
  python -m data.diagnostics --random 20

  # Sample every 1000th glyph
  python -m data.diagnostics --every 1000

  # Specific glyph IDs
  python -m data.diagnostics --ids 604 605 9999

  # Include DB for hybrid rule cross-check (advance_width)
  python -m data.diagnostics --random 50 --glyph-db dataset/glyphs.db

  # Specify metadata path and config thresholds manually
  python -m data.diagnostics --metadata data/rasters/metadata.jsonl \
      --grid-rows 16 --grid-cols 16 --cell-px 8 \
      --ratio-threshold 0.15 --adv-threshold 100

Exit codes:
  0 = no anomalies
  1 = anomalies found

Notes:
  - Assumes a binary 128x128 raster (0 or 255).
  - Occupancy grid stored as (rows, cols) uint8 (0/1).
  - Recomputed grid: cell considered active if any pixel >= 128 in its block.
  - Hybrid rule: is_diacritic = (advance_width < adv_threshold) OR (ratio < ratio_threshold)
    If advance_width is unavailable (no DB provided), only ratio is validated.

Limitations:
  - Does not currently verify scale_factor correctness (would require re-scaling vectors).
  - Stochastic sampling reproduces with --seed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Any

try:
    import numpy as np
except ImportError:
    print("[ERROR] numpy is required for diagnostics.", file=sys.stderr)
    sys.exit(2)

try:
    from PIL import Image
except ImportError:
    print("[ERROR] Pillow (PIL) is required for diagnostics.", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class Thresholds:
    ratio: float
    advance: int


@dataclass
class SampleRecord:
    glyph_id: int
    label: str
    is_diacritic: bool
    ratio: float
    target_dim: int
    raster_filename: str
    orig_glyph_id: int


@dataclass
class FileCheckResult:
    glyph_id: int
    raster_ok: bool
    grid_ok: bool
    raster_path: Path
    grid_path: Path


@dataclass
class GridValidation:
    glyph_id: int
    cells_on: int
    match: bool
    expected_shape: Tuple[int, int]
    recomputed_cells_on: int


@dataclass
class HybridCheck:
    glyph_id: int
    reported_diacritic: bool
    ratio: float
    advance_width: Optional[int]
    expected_diacritic: Optional[bool]
    consistent: Optional[bool]
    reason: str


@dataclass
class Anomaly:
    glyph_id: int
    kind: str
    detail: str


# ---------------------------------------------------------------------------
# Metadata Loading
# ---------------------------------------------------------------------------


def load_metadata(path: Path) -> List[SampleRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    records: List[SampleRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                records.append(
                    SampleRecord(
                        glyph_id=int(j["glyph_id"]),
                        label=str(j.get("label", "")),
                        is_diacritic=bool(j.get("is_diacritic", False)),
                        ratio=float(j.get("major_dim_ratio", 0.0)),
                        target_dim=int(j.get("target_dim", 128)),
                        raster_filename=str(
                            j.get("raster_filename", f"{j['glyph_id']}.png")
                        ),
                        orig_glyph_id=int(j.get("orig_glyph_id", -1)),
                    )
                )
            except KeyError:
                # Skip malformed line
                continue
    return records


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_records(
    records: List[SampleRecord],
    *,
    random_count: Optional[int],
    every: Optional[int],
    explicit_ids: Optional[Sequence[int]],
    seed: int,
) -> List[SampleRecord]:
    if explicit_ids:
        id_set = set(explicit_ids)
        return [r for r in records if r.glyph_id in id_set]

    if every and every > 0:
        return [r for idx, r in enumerate(records) if idx % every == 0]

    if random_count is not None:
        rng = random.Random(seed)
        if random_count >= len(records):
            return list(records)
        return rng.sample(records, random_count)

    # Default: first 10 if nothing specified
    return records[:10]


# ---------------------------------------------------------------------------
# File Existence & Grid Validation
# ---------------------------------------------------------------------------


def check_files(
    samples: List[SampleRecord],
    rasters_dir: Path,
    grids_dir: Path,
) -> List[FileCheckResult]:
    results: List[FileCheckResult] = []
    for r in samples:
        raster_path = rasters_dir / r.raster_filename
        grid_path = grids_dir / f"{r.glyph_id}.npy"
        results.append(
            FileCheckResult(
                glyph_id=r.glyph_id,
                raster_ok=raster_path.exists(),
                grid_ok=grid_path.exists(),
                raster_path=raster_path,
                grid_path=grid_path,
            )
        )
    return results


def recompute_grid(
    raster_path: Path,
    *,
    rows: int,
    cols: int,
    cell_px: int,
) -> Tuple[Any, int]:
    """
    Returns:
      grid: np.ndarray shape (rows, cols) uint8
      cells_on: int
    """
    img = Image.open(raster_path)
    arr = np.array(img)
    if arr.shape != (rows * cell_px, cols * cell_px):
        raise ValueError(
            f"Unexpected raster shape {arr.shape} (expected {(rows * cell_px, cols * cell_px)})"
        )
    grid = np.zeros((rows, cols), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            block = arr[
                r * cell_px : (r + 1) * cell_px, c * cell_px : (c + 1) * cell_px
            ]
            if (block >= 128).any():  # binary 0/255
                grid[r, c] = 1
    return grid, int(grid.sum())


def validate_grids(
    samples: List[SampleRecord],
    file_checks: List[FileCheckResult],
    *,
    rows: int,
    cols: int,
    cell_px: int,
) -> List[GridValidation]:
    check_map = {c.glyph_id: c for c in file_checks}
    validations: List[GridValidation] = []
    for s in samples:
        c = check_map[s.glyph_id]
        if not (c.raster_ok and c.grid_ok):
            continue
        try:
            stored = np.load(c.grid_path, allow_pickle=False)
            if stored.shape != (rows, cols):
                raise ValueError(
                    f"Stored grid shape {stored.shape} != expected {(rows, cols)}"
                )
            recomputed, recomputed_on = recompute_grid(
                c.raster_path, rows=rows, cols=cols, cell_px=cell_px
            )
            match = bool((recomputed == stored).all())
            validations.append(
                GridValidation(
                    glyph_id=s.glyph_id,
                    cells_on=int(stored.sum()),
                    match=match,
                    expected_shape=(rows, cols),
                    recomputed_cells_on=recomputed_on,
                )
            )
        except Exception as e:
            validations.append(
                GridValidation(
                    glyph_id=s.glyph_id,
                    cells_on=-1,
                    match=False,
                    expected_shape=(rows, cols),
                    recomputed_cells_on=-1,
                )
            )
    return validations


# ---------------------------------------------------------------------------
# Hybrid Rule Consistency
# ---------------------------------------------------------------------------


def fetch_advance_widths(
    glyph_db: Path, glyph_ids: Sequence[int]
) -> Dict[int, Optional[int]]:
    """
    Fetch advance_width for given glyph primary key IDs.
    """
    result: Dict[int, Optional[int]] = {gid: None for gid in glyph_ids}
    if not glyph_db or not glyph_db.exists():
        return result
    conn = sqlite3.connect(str(glyph_db))
    try:
        q_marks = ",".join(["?"] * len(glyph_ids))
        cur = conn.execute(
            f"SELECT id, advance_width FROM glyphs WHERE id IN ({q_marks})",
            list(glyph_ids),
        )
        for row in cur:
            gid, adv = row
            result[int(gid)] = int(adv) if adv is not None else None
    finally:
        conn.close()
    return result


def check_hybrid_consistency(
    samples: List[SampleRecord],
    advance_map: Dict[int, Optional[int]],
    thresholds: Thresholds,
) -> List[HybridCheck]:
    results: List[HybridCheck] = []
    for s in samples:
        adv = advance_map.get(s.glyph_id)
        expected = None
        consistent = None
        reason = ""
        if adv is not None:
            expected = (adv < thresholds.advance) or (s.ratio < thresholds.ratio)
            consistent = expected == s.is_diacritic
            if not consistent:
                reason = (
                    f"Mismatch: reported={s.is_diacritic} expected={expected} "
                    f"(adv={adv}, ratio={s.ratio:.4f})"
                )
        else:
            # Only partial validation: ratio side
            if s.ratio < thresholds.ratio and not s.is_diacritic:
                reason = (
                    "ratio<threshold but flagged non-diacritic (advance unavailable)"
                )
            elif s.ratio >= thresholds.ratio and s.is_diacritic:
                reason = "ratio>=threshold but flagged diacritic (advance unavailable)"
            if reason:
                expected = None
                consistent = None
        results.append(
            HybridCheck(
                glyph_id=s.glyph_id,
                reported_diacritic=s.is_diacritic,
                ratio=s.ratio,
                advance_width=adv,
                expected_diacritic=expected,
                consistent=consistent,
                reason=reason,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Aggregation / Reporting
# ---------------------------------------------------------------------------


def summarize(
    samples: List[SampleRecord],
    file_checks: List[FileCheckResult],
    grid_validations: List[GridValidation],
    hybrid_checks: List[HybridCheck],
) -> Tuple[List[Anomaly], Dict[str, Any]]:
    anomalies: List[Anomaly] = []

    file_map = {f.glyph_id: f for f in file_checks}
    grid_map = {g.glyph_id: g for g in grid_validations}
    hybrid_map = {h.glyph_id: h for h in hybrid_checks}

    diac_cells = []
    non_cells = []
    for s in samples:
        gv = grid_map.get(s.glyph_id)
        if gv and gv.cells_on >= 0:
            if s.is_diacritic:
                diac_cells.append(gv.cells_on)
            else:
                non_cells.append(gv.cells_on)

    # File anomalies
    for fc in file_checks:
        if not fc.raster_ok:
            anomalies.append(
                Anomaly(fc.glyph_id, "missing_raster", str(fc.raster_path))
            )
        if not fc.grid_ok:
            anomalies.append(Anomaly(fc.glyph_id, "missing_grid", str(fc.grid_path)))

    # Grid mismatch anomalies
    for gv in grid_validations:
        if not gv.match:
            anomalies.append(
                Anomaly(
                    gv.glyph_id,
                    "grid_mismatch",
                    f"cells_on={gv.cells_on} recomputed={gv.recomputed_cells_on}",
                )
            )

    # Hybrid mismatches
    for hc in hybrid_checks:
        if hc.reason and hc.consistent is False:
            anomalies.append(Anomaly(hc.glyph_id, "hybrid_mismatch", hc.reason))
        elif hc.reason and hc.consistent is None:
            # ratio-only inconsistency (advance missing)
            anomalies.append(Anomaly(hc.glyph_id, "hybrid_ratio_warning", hc.reason))

    def stats(arr: List[int]) -> Dict[str, Any]:
        if not arr:
            return {}
        a = sorted(arr)
        return {
            "count": len(a),
            "min": int(a[0]),
            "max": int(a[-1]),
            "mean": float(sum(a) / len(a)),
            "p50": float(a[len(a) // 2]),
            "p90": float(a[int(len(a) * 0.9) - 1]),
            "p95": float(a[int(len(a) * 0.95) - 1]),
        }

    summary = {
        "sample_size": len(samples),
        "files_checked": len(file_checks),
        "grids_validated": len(grid_validations),
        "hybrid_checks": len(hybrid_checks),
        "diacritic_cell_stats": stats(diac_cells),
        "non_diacritic_cell_stats": stats(non_cells),
        "anomaly_count": len(anomalies),
    }
    return anomalies, summary


def print_report(
    samples: List[SampleRecord],
    file_checks: List[FileCheckResult],
    grid_validations: List[GridValidation],
    hybrid_checks: List[HybridCheck],
    anomalies: List[Anomaly],
    summary: Dict[str, Any],
    *,
    json_lines: bool,
    top_n: int,
) -> None:
    if json_lines:
        for s in samples[:top_n]:
            print(
                json.dumps(
                    {
                        "glyph_id": s.glyph_id,
                        "label": s.label,
                        "is_diacritic": s.is_diacritic,
                        "ratio": s.ratio,
                        "target_dim": s.target_dim,
                        "raster": next(
                            (
                                f.raster_path.name
                                for f in file_checks
                                if f.glyph_id == s.glyph_id
                            ),
                            None,
                        ),
                    },
                    ensure_ascii=False,
                )
            )
        print(json.dumps({"summary": summary}, ensure_ascii=False))
        for a in anomalies:
            print(
                json.dumps(
                    {
                        "anomaly": {
                            "glyph_id": a.glyph_id,
                            "kind": a.kind,
                            "detail": a.detail,
                        }
                    },
                    ensure_ascii=False,
                )
            )
        return

    print("=== SAMPLE (truncated) ===")
    for s in samples[:top_n]:
        print(
            f"[SAMPLE] id={s.glyph_id} label={s.label} "
            f"diac={s.is_diacritic} ratio={s.ratio:.4f} target_dim={s.target_dim}"
        )

    gv_map = {g.glyph_id: g for g in grid_validations}
    print("\n=== GRID VALIDATION (sample order) ===")
    for s in samples[:top_n]:
        gv = gv_map.get(s.glyph_id)
        if gv:
            print(
                f"[GRID] id={gv.glyph_id} match={gv.match} cells_on={gv.cells_on} "
                f"recomputed={gv.recomputed_cells_on}"
            )

    hc_map = {h.glyph_id: h for h in hybrid_checks}
    print("\n=== HYBRID CHECKS (sample order) ===")
    for s in samples[:top_n]:
        hc = hc_map.get(s.glyph_id)
        if hc:
            print(
                f"[HYBRID] id={hc.glyph_id} reported={hc.reported_diacritic} "
                f"ratio={hc.ratio:.4f} adv={hc.advance_width} consistent={hc.consistent} "
                f"{'reason=' + hc.reason if hc.reason else ''}"
            )

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if anomalies:
        print("\n=== ANOMALIES ===")
        for a in anomalies:
            print(f"[ANOMALY] id={a.glyph_id} kind={a.kind} detail={a.detail}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Raster/Grid integrity diagnostics")
    p.add_argument("--metadata", type=Path, default=Path("data/rasters/metadata.jsonl"))
    p.add_argument("--rasters-dir", type=Path, default=Path("data/rasters"))
    p.add_argument("--grids-dir", type=Path, default=Path("data/grids"))
    p.add_argument(
        "--glyph-db",
        type=Path,
        default=None,
        help="sqlite glyphs.db for advance_width checks",
    )

    group = p.add_mutually_exclusive_group()
    group.add_argument("--random", type=int, help="Random sample size")
    group.add_argument("--every", type=int, help="Take every Nth record")
    p.add_argument("--ids", type=int, nargs="+", help="Explicit glyph ids")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rows", type=int, default=16)
    p.add_argument("--cols", type=int, default=16)
    p.add_argument("--cell-px", type=int, default=8)

    p.add_argument("--ratio-threshold", type=float, default=0.15)
    p.add_argument("--adv-threshold", type=int, default=100)

    p.add_argument(
        "--json-lines",
        action="store_true",
        help="Emit JSON lines for downstream ingestion",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Max sample lines to display in human-readable mode",
    )
    p.add_argument(
        "--reconstruct",
        action="store_true",
        help="Attempt coarse raster reconstruction from occupancy grid and report similarity stats",
    )
    p.add_argument(
        "--recons-dir",
        type=Path,
        default=Path("data/recons"),
        help="Directory to write reconstructed raster PNGs (used with --reconstruct)",
    )

    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    try:
        records = load_metadata(args.metadata)
    except Exception as e:
        print(f"[ERROR] Failed loading metadata: {e}", file=sys.stderr)
        return 2

    if not records:
        print("[WARN] No records in metadata.", file=sys.stderr)
        return 1

    samples = sample_records(
        records,
        random_count=args.random,
        every=args.every,
        explicit_ids=args.ids,
        seed=args.seed,
    )
    if not samples:
        print("[WARN] No samples selected.", file=sys.stderr)
        return 1

    file_checks = check_files(samples, args.rasters_dir, args.grids_dir)
    grid_validations = validate_grids(
        samples,
        file_checks,
        rows=args.rows,
        cols=args.cols,
        cell_px=args.cell_px,
    )

    advance_map: Dict[int, Optional[int]] = {}
    if args.glyph_db:
        try:
            advance_map = fetch_advance_widths(
                args.glyph_db, [s.glyph_id for s in samples]
            )
        except Exception as e:
            print(f"[WARN] Could not fetch advance widths: {e}", file=sys.stderr)

    thresholds = Thresholds(ratio=args.ratio_threshold, advance=args.adv_threshold)
    hybrid_checks = check_hybrid_consistency(samples, advance_map, thresholds)

    anomalies, summary = summarize(
        samples, file_checks, grid_validations, hybrid_checks
    )

    print_report(
        samples,
        file_checks,
        grid_validations,
        hybrid_checks,
        anomalies,
        summary,
        json_lines=args.json_lines,
        top_n=args.top_n,
    )

    return 0 if not anomalies else 1


# ---------------------------------------------------------------------------
# (Optional) Raster Reconstruction Utilities
# ---------------------------------------------------------------------------


def reconstruct_raster_from_grid(grid, cell_px: int = 8, on_value: int = 255):
    """
    Reconstruct a coarse raster from a (rows, cols) occupancy grid by
    filling each active cell with a solid block of on_value.

    NOTE:
      This cannot recreate the original glyph shape, only a blocky proxy.
      Information lost inside each 8x8 cell (the precise stroke pattern)
      is unrecoverable. Therefore a byte-level equality check will almost
      always fail unless the original glyph cells were either fully empty
      or fully filled.
    """
    import numpy as _np

    rows, cols = grid.shape
    canvas = _np.zeros((rows * cell_px, cols * cell_px), dtype=_np.uint8)
    for r in range(rows):
        for c in range(cols):
            if grid[r, c]:
                canvas[
                    r * cell_px : (r + 1) * cell_px, c * cell_px : (c + 1) * cell_px
                ] = on_value
    return canvas


def compare_reconstruction(raster_path, grid, cell_px=8):
    """
    Returns dict with:
      exact_match: bool
      differing_pixels: int
      original_on_pixels: int
      reconstructed_on_pixels: int
      jaccard: float  (intersection / union over on-pixel sets)
    """
    from PIL import Image
    import numpy as _np

    recon = reconstruct_raster_from_grid(grid, cell_px=cell_px)
    orig = _np.array(Image.open(raster_path))
    if orig.shape != recon.shape:
        raise ValueError(f"Shape mismatch orig={orig.shape} recon={recon.shape}")

    orig_on = orig >= 128
    recon_on = recon >= 128
    intersection = (orig_on & recon_on).sum()
    union = (orig_on | recon_on).sum()
    differing = (orig != recon).sum()
    return {
        "exact_match": differing == 0,
        "differing_pixels": int(differing),
        "original_on_pixels": int(orig_on.sum()),
        "reconstructed_on_pixels": int(recon_on.sum()),
        "jaccard": float(intersection / union) if union else 1.0,
    }


# Extend report functions to optionally display reconstruction stats.
# We inject a lightweight hook without restructuring existing logic:
def _augment_with_reconstruction(
    samples,
    file_checks,
    grid_validations,
    cell_px=8,
    limit=20,
    recons_dir: Path | None = None,
):
    """
    For the first `limit` samples with valid grids, compute reconstruction similarity.
    Returns list of tuples (glyph_id, stats_dict).
    """
    import numpy as _np

    stats = []
    gv_map = {g.glyph_id: g for g in grid_validations}
    fc_map = {f.glyph_id: f for f in file_checks}
    shown = 0
    if recons_dir:
        recons_dir.mkdir(parents=True, exist_ok=True)
    for s in samples:
        if shown >= limit:
            break
        gv = gv_map.get(s.glyph_id)
        fc = fc_map.get(s.glyph_id)
        if not gv or not fc or not (gv.match and fc.raster_ok and fc.grid_ok):
            continue
        try:
            grid = __import__("numpy").load(fc.grid_path, allow_pickle=False)
            recon_stats = compare_reconstruction(fc.raster_path, grid, cell_px=cell_px)
            # Save coarse reconstructed PNG if requested
            if recons_dir:
                try:
                    from PIL import Image as _Image

                    recon_img = reconstruct_raster_from_grid(grid, cell_px=cell_px)
                    _Image.fromarray(recon_img, mode="L").save(
                        recons_dir / fc.raster_path.name
                    )
                except Exception as _e:
                    recon_stats["save_error"] = str(_e)
            stats.append((s.glyph_id, recon_stats))
            shown += 1
        except Exception as e:
            stats.append((s.glyph_id, {"error": str(e)}))
            shown += 1
    return stats


# Patch main to include reconstruction comparison when --reconstruct is passed.
def _patched_parse_args(original_parse):
    def wrapper(argv):
        ns = original_parse(argv)
        # Add dynamic attribute (default False) if not present
        if not hasattr(ns, "reconstruct"):
            setattr(ns, "reconstruct", False)
        return ns

    return wrapper


# Monkey-patch parse_args to include new flag if not already defined.
# (Safer than modifying earlier parse section in this appended patch.)
if "parse_args" in globals():
    import argparse as _argparse

    real_parse_args = parse_args

    def parse_args(argv):
        # Rebuild parser with the original, then add new flag.
        ns = real_parse_args(argv)
        return ns

    # We cannot re-open the original parser easily without re-defining;
    # instead we accept --reconstruct via manual scan:
    import sys as _sys

    if "--reconstruct" in _sys.argv:
        # Inject flag manually after initial parse
        # (since original parser will have rejected unknown args, user must add it last).
        pass


def _has_flag(name: str) -> bool:
    import sys as _sys

    return name in _sys.argv


# Wrap original main to add reconstruction phase if requested.
_original_main = main


def main(argv=None):
    rc = 0
    # Run original main logic but intercept printed output by re-running
    # core pieces (simpler: re-execute original and then re-open for reconstruction).
    # For minimal intrusion, we re-run parsing here.
    if argv is None:
        argv = sys.argv[1:]
    # Detect reconstruct flag (duck typed)
    reconstruct = _has_flag("--reconstruct")
    # Execute original main (prints base report)
    rc = _original_main(argv)
    if not reconstruct:
        return rc
    # If reconstruction requested, re-parse to get sample selection deterministically.
    args = parse_args(argv)
    try:
        records = load_metadata(args.metadata)
    except Exception as e:
        print(f"[RECON][ERROR] cannot reload metadata: {e}", file=sys.stderr)
        return rc
    samples = sample_records(
        records,
        random_count=args.random if hasattr(args, "random") else None,
        every=args.every if hasattr(args, "every") else None,
        explicit_ids=args.ids if hasattr(args, "ids") else None,
        seed=getattr(args, "seed", 42),
    )
    file_checks = check_files(samples, args.rasters_dir, args.grids_dir)
    grid_validations = validate_grids(
        samples,
        file_checks,
        rows=getattr(args, "rows", 16),
        cols=getattr(args, "cols", 16),
        cell_px=getattr(args, "cell_px", 8),
    )
    recon_stats = _augment_with_reconstruction(
        samples,
        file_checks,
        grid_validations,
        cell_px=getattr(args, "cell_px", 8),
        limit=getattr(args, "top_n", 20),
        recons_dir=Path(getattr(args, "recons_dir", "data/recons"))
        if getattr(args, "reconstruct", False)
        else None,
    )
    print("\n=== RECONSTRUCTION (coarse grid fill) ===")
    for gid, st in recon_stats:
        if "error" in st:
            print(f"[RECON] id={gid} error={st['error']}")
            continue
        print(
            f"[RECON] id={gid} exact={st['exact_match']} "
            f"diff_pixels={st['differing_pixels']} "
            f"orig_on={st['original_on_pixels']} "
            f"recon_on={st['reconstructed_on_pixels']} "
            f"jaccard={st['jaccard']:.4f}"
        )
    print(
        "\n[INFO] Reconstruction uses block fill per active cell; "
        "exact_match is expected to be False for most glyphs because "
        "intra-cell shape detail is irretrievably lost in the occupancy grid."
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
