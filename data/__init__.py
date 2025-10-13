"""
glyph-grid.data package initialization.

This module provides high-level entry points and scaffolding for the
data preprocessing steps described in NEW_PLAN.md:

Plan Alignment:
  Sections 3.x : Data model concepts (glyph contours, labels)
  Section 4    : Rasterization spec (128×128 main glyph, diacritic heuristic)
  Section 5    : 16×16 grid extraction (8×8 cells)
  Section 6.1  : Primitive vocabulary initialization via K-Means on ~1M non-empty cells
  Section 6.2  : (Downstream Phase 1 model consumes primitive IDs; not implemented here)
  Section 9.1  : Preprocessing tasks list

Scope of this file:
  - Define lightweight dataclasses representing glyph vector data, raster results, and config fragments.
  - Provide public API functions you can call from scripts (e.g. data/rasterize.py to be added) for:
        rasterize_glyphs()
        sample_cells_for_kmeans()
        compute_primitive_kmeans()
        assign_primitives_to_cells()
  - Offer reference implementations / placeholders with emphasis on determinism and clarity.
  - Keep heavy logic (polygon fill, DB iteration) to be implemented in forthcoming dedicated modules.
    This keeps __init__ import fast and avoids circular dependencies.

Design Notes:
  - Rasterization: Final binary mask 128x128 (uint8 {0,255}); diacritic scaling rule (largest dim=64).
  - Grid extraction: Row-major 16x16 cells each 8x8; empty cell → primitive 0.
  - Primitive vocabulary: K = 1023 (excluding empty). Cells flattened to 64D vectors (uint8 -> float32).
  - Determinism: Force a fixed numpy random seed for sampling & K-Means init.

External Dependencies (expected downstream, not strictly required at import time):
  - numpy
  - pillow (PIL) for PNG writing (in rasterization module, not here)
  - scikit-learn for KMeans (optional fallback: MiniBatchKMeans if memory)
  - sqlite3 (standard library) for glyph DB access

The functions below are intentionally conservative: they validate shapes and
avoid silent coercions. Actual raster polygon filling, oversampling, and downsampling
will reside in data/rasterize.py (to be implemented in step 1).

Author: Automated scaffold (expert engineering template)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Tuple, Optional, Dict, Any, Sequence, Callable
import json
import math
import os
import random
import logging
import numpy as np

try:
    from sklearn.cluster import KMeans, MiniBatchKMeans
except Exception:  # pragma: no cover - sklearn may not be installed yet
    KMeans = MiniBatchKMeans = None  # type: ignore

logger = logging.getLogger("glyph_grid.data")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# --------------------------------------------------------------------------------------
# Dataclasses / Type Models
# --------------------------------------------------------------------------------------


@dataclass
class RasterizerConfig:
    """
    Parsed subset of configs/rasterizer.yaml relevant to raster/glyph → grid pipeline.
    Only includes fields required by code in this stage; extended validation lives in
    the future rasterize.py module.
    """

    canvas_size: int = 128
    supersample_factor: int = 2
    curve_subdivisions: int = 8
    binarize_threshold: float = 0.5
    main_target_dim: int = 128
    diacritic_target_dim: int = 64
    diacritic_ratio_threshold: float = 0.25
    grid_rows: int = 16
    grid_cols: int = 16
    cell_px: int = 8
    empty_primitive_id: int = 0

    @property
    def supersampled_size(self) -> int:
        return self.canvas_size * self.supersample_factor

    def validate(self) -> None:
        assert self.canvas_size == 128, "Plan requires 128 final canvas."
        assert self.grid_rows == 16 and self.grid_cols == 16, (
            "Plan requires 16x16 grid."
        )
        assert self.cell_px * self.grid_rows == self.canvas_size, (
            "cell_px * rows must equal canvas_size."
        )
        assert self.cell_px * self.grid_cols == self.canvas_size, (
            "cell_px * cols must equal canvas_size."
        )
        assert 0.0 < self.diacritic_ratio_threshold < 1.0, (
            "ratio threshold must be (0,1)."
        )
        assert self.main_target_dim == 128 and self.diacritic_target_dim == 64, (
            "Scaling targets fixed by plan."
        )


@dataclass
class GlyphVector:
    """
    Represents one glyph's vector data as loaded from the sqlite DB.

    contours: List of subpaths; each subpath is a list of commands:
        command tuple structure examples:
            ("moveTo", (x, y))
            ("lineTo", (x, y))
            ("qCurveTo", ( (cx, cy), (x, y) ))  # Intermediate control for quadratic
            ("cubicTo", ( (c1x,c1y), (c2x,c2y), (x,y) ))
            ("closePath", None)

    The plan's "Vector Contours JSON" section indicates arrays of arrays.
    We canonicalize into tuples for internal use.
    """

    glyph_id: int
    label: str
    font_hash: str
    contours: List[List[Tuple[str, Any]]]  # list of subpaths, each list of (op, data)
    original_bbox: Tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)
    used_ascent: float
    used_descent: float

    def major_dimension(self) -> float:
        min_x, min_y, max_x, max_y = self.original_bbox
        return max(max_x - min_x, max_y - min_y)

    def em_height(self) -> float:
        return self.used_ascent - self.used_descent


@dataclass
class RasterResult:
    """
    Output of glyph rasterization BEFORE grid slicing.

    mask: (H,W) uint8 binary (0 or 255)
    is_diacritic: bool per ratio rule
    scale_factor: float used to scale original vector units to pixel space
    major_dim_ratio: major_dim / em_height (stored for metadata)
    target_dim: int (128 main or 64 diacritic)
    """

    glyph_id: int
    label: str
    font_hash: str
    mask: np.ndarray
    is_diacritic: bool
    scale_factor: float
    major_dim_ratio: float
    target_dim: int
    original_bbox: Tuple[float, float, float, float]


@dataclass
class CellGrid:
    """
    16x16 grid extracted from a rasterized glyph.
    cells: numpy array shape (16,16,8,8) uint8 binary
    """

    glyph_id: int
    cells: np.ndarray  # (16,16,8,8) uint8
    label: str
    is_diacritic: bool


# --------------------------------------------------------------------------------------
# Configuration Loader
# --------------------------------------------------------------------------------------


def load_rasterizer_config(path: str) -> RasterizerConfig:
    """
    Parse the YAML config (without pulling full dependency if not required).
    Minimal fallback parser using 'yaml' if installed, else a strict subset via json after preprocessing.

    Raises:
        FileNotFoundError, ValueError
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyYAML required to load rasterizer config.") from e

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Map keys defensively
    raster = raw.get("raster", {})
    grid = raw.get("grid", {})

    cfg = RasterizerConfig(
        canvas_size=raster.get("canvas_size", 128),
        supersample_factor=raster.get("supersample_factor", 2),
        curve_subdivisions=raster.get("curve_subdivisions", 8),
        binarize_threshold=raster.get("binarize_threshold", 0.5),
        main_target_dim=raster.get("main_target_dim", 128),
        diacritic_target_dim=raster.get("diacritic_target_dim", 64),
        diacritic_ratio_threshold=raster.get("diacritic_ratio_threshold", 0.25),
        grid_rows=grid.get("rows", 16),
        grid_cols=grid.get("cols", 16),
        cell_px=grid.get("cell_px", 8),
        empty_primitive_id=grid.get("empty_primitive_id", 0),
    )
    cfg.validate()
    return cfg


# --------------------------------------------------------------------------------------
# Public Rasterization API (High-Level)
# --------------------------------------------------------------------------------------


def rasterize_glyphs(
    glyphs: Iterable[GlyphVector],
    config: RasterizerConfig,
    *,
    progress: Optional[Callable[[int], None]] = None,
) -> Iterable[Tuple[RasterResult, CellGrid]]:
    """
    High-level generator that:
      1. Applies diacritic heuristic
      2. Scales + translates vector contours
      3. Renders supersampled binary mask (placeholder implementation here)
      4. Downsamples to 128x128, thresholds → binary
      5. Extracts 16x16x(8x8) cells

    NOTE: This is a SCAFFOLD. The actual polygon rasterization (fill rule, curve subdivision)
    should be implemented in a dedicated module. For now, this function raises
    NotImplementedError to signal downstream script to import the concrete implementation
    once added (data/rasterize.py).

    Yields:
        (RasterResult, CellGrid)

    Plan references: Sections 4.1–4.4, 5.
    """
    raise NotImplementedError(
        "rasterize_glyphs scaffold: implement full polygon raster + supersampling "
        "in data/rasterize.py and re-export or override this function."
    )


# --------------------------------------------------------------------------------------
# Primitive Vocabulary (K-Means) – Section 6.1
# --------------------------------------------------------------------------------------


def sample_cells_for_kmeans(
    cell_iterator: Iterable[np.ndarray],
    sample_size: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Randomly sample up to 'sample_size' non-empty 8x8 cells from an iterable of cells.

    Args:
        cell_iterator: iterable yielding (8,8) uint8 arrays (binary {0,255} or {0,1})
        sample_size: target number of samples (~1_000_000 per plan)
        seed: RNG seed for reproducibility

    Returns:
        samples: (N, 64) float32 array normalized to [0,1]
    """
    rng = random.Random(seed)
    reservoir: List[np.ndarray] = []
    n_seen = 0

    for cell in cell_iterator:
        if cell.shape != (8, 8):
            raise ValueError(f"Cell shape must be (8,8), got {cell.shape}")
        # Consider empty cell (all zeros) – skip for K=1023 clustering
        if not np.any(cell):
            continue
        flat = cell.reshape(-1)
        if len(reservoir) < sample_size:
            reservoir.append(flat.copy())
        else:
            # Reservoir sampling
            j = rng.randint(0, n_seen)
            if j < sample_size:
                reservoir[j] = flat.copy()
        n_seen += 1

    if not reservoir:
        raise ValueError("No non-empty cells encountered for sampling.")
    arr = np.stack(reservoir, axis=0).astype(np.float32)
    # Normalize: binary {0,255} → {0,1}
    if arr.max() > 1.0:
        arr /= 255.0
    return arr


def compute_primitive_kmeans(
    samples: np.ndarray,
    k: int = 1023,
    seed: int = 42,
    batch: bool = True,
    max_iter: int = 300,
) -> np.ndarray:
    """
    Run K-Means on sampled cell vectors.

    Args:
        samples: (N,64) float32 in [0,1]
        k: number of clusters (1023 per plan)
        seed: random state
        batch: whether to use MiniBatchKMeans if available
        max_iter: max iterations (full KMeans) or max_iter for MiniBatch loops

    Returns:
        centroids: (k,64) float32 in [0,1]

    Raises:
        RuntimeError if sklearn unavailable.
    """
    if KMeans is None:
        raise RuntimeError("scikit-learn is required for K-Means but is not installed.")

    if samples.ndim != 2 or samples.shape[1] != 64:
        raise ValueError("samples must have shape (N,64)")

    if batch and MiniBatchKMeans is not None:
        logger.info("Using MiniBatchKMeans for primitive vocabulary (k=%d)", k)
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            batch_size=4096,
            max_iter=max_iter,
            reassignment_ratio=0.01,
            n_init="auto",
        )
    else:
        logger.info("Using full KMeans for primitive vocabulary (k=%d)", k)
        model = KMeans(
            n_clusters=k,
            random_state=seed,
            max_iter=max_iter,
            n_init="auto",
        )
    model.fit(samples)
    centroids = model.cluster_centers_.astype(np.float32)
    return centroids


def assign_primitives_to_cells(
    cell_array: np.ndarray,
    centroids: np.ndarray,
    empty_id: int = 0,
) -> np.ndarray:
    """
    Assign primitive IDs to each cell via nearest centroid (L2).

    Args:
        cell_array: (H,W,8,8) uint8 or bool or {0,255}. Typically (16,16,8,8).
        centroids: (K,64) float32 cluster centers in [0,1].
        empty_id: ID reserved for all-empty cells (0 per plan).

    Returns:
        ids: (H,W) int32 array where empty cells map to empty_id and non-empty to 1..K.
             NOTE: The caller must ensure consistent mapping: cluster index i -> primitive_id (i+1)
                   since 0 is reserved for empty.

    Steps:
        1. Flatten each cell to 64D float32 (0/1).
        2. Identify empty cells (all zeros) → empty_id.
        3. Compute squared distances to centroids; pick argmin.
        4. Add +1 offset so cluster 0 becomes primitive 1.
    """
    if cell_array.ndim != 4:
        raise ValueError("cell_array must have shape (H,W,8,8)")
    H, W, h, w = cell_array.shape
    if (h, w) != (8, 8):
        raise ValueError("Inner cell spatial size must be (8,8).")

    # Flatten
    flat = cell_array.reshape(H * W, 64).astype(np.float32)
    if flat.max() > 1.0:
        flat /= 255.0

    # Empty mask
    empty_mask = flat.sum(axis=1) == 0.0

    # Prepare output
    ids = np.empty(H * W, dtype=np.int32)
    ids[empty_mask] = empty_id

    non_empty_idx = np.where(~empty_mask)[0]
    if non_empty_idx.size:
        # Distances: (N_non_empty, K)
        feats = flat[non_empty_idx]  # (N_non_empty,64)
        # Efficient distance computation: ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x·c
        x2 = np.sum(feats * feats, axis=1, keepdims=True)  # (N,1)
        c2 = np.sum(centroids * centroids, axis=1)[None, :]  # (1,K)
        dots = feats @ centroids.T  # (N,K)
        dists = x2 + c2 - 2.0 * dots
        cluster_ids = np.argmin(dists, axis=1)  # 0..K-1
        # Map cluster i -> primitive (i+1) to leave 0 for empty
        ids[non_empty_idx] = cluster_ids + 1
    primitive_grid = ids.reshape(H, W)
    return primitive_grid


# --------------------------------------------------------------------------------------
# Helper Utilities
# --------------------------------------------------------------------------------------


def save_centroids(path: str, centroids: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, centroids)
    logger.info("Saved primitive centroids: %s (shape=%s)", path, centroids.shape)


def load_centroids(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 64:
        raise ValueError(f"Centroid file {path} must be (K,64), got {arr.shape}")
    return arr.astype(np.float32)


def write_label_map(path: str, mapping: Dict[str, int]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    logger.info("Wrote label map (%d entries) to %s", len(mapping), path)


def load_label_map(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Label map JSON must be an object.")
    return {str(k): int(v) for k, v in data.items()}


# --------------------------------------------------------------------------------------
# Public API Exports
# --------------------------------------------------------------------------------------

__all__ = [
    # Dataclasses
    "RasterizerConfig",
    "GlyphVector",
    "RasterResult",
    "CellGrid",
    # Config
    "load_rasterizer_config",
    # Rasterization placeholder
    "rasterize_glyphs",
    # Primitive vocabulary functions
    "sample_cells_for_kmeans",
    "compute_primitive_kmeans",
    "assign_primitives_to_cells",
    # Helpers
    "save_centroids",
    "load_centroids",
    "write_label_map",
    "load_label_map",
]
