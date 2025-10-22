#!/usr/bin/env python3
"""
Sequence-Aware Glyph Grid Dataset
==================================

Extends GlyphGridDataset to load context windows of neighboring glyphs
for sequence-aware glyph classification.

Key Features:
- Loads center glyph + k neighbors before/after (based on glyph_id sequence)
- Computes glyph_id deltas for positional encoding
- Handles boundary cases (start/end of dataset) gracefully
- Compatible with memmap and per-file modes
- Returns context_mask for variable-length sequences

Usage:
------
from train.dataset_sequence import GlyphGridSequenceDataset

dataset = GlyphGridSequenceDataset(
    glyph_ids=[100, 101, 102, ...],
    grids_dir=Path("data/grids_memmap"),
    label_map=label_map,
    glyph_to_label=glyph_to_label,
    context_window=2,  # 2 before + 2 after = 4 neighbors
)

# Returns: (center_grid, context_grids, context_deltas, label_idx, gid, is_diacritic)
# center_grid: (16, 16)
# context_grids: (K, 16, 16) where K = 2*context_window
# context_deltas: (K,) - glyph_id deltas (e.g., [-2, -1, +1, +2])
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class GlyphGridSequenceDataset(Dataset):
    """
    Loads primitive ID grids with sequential context windows.

    For each center glyph, loads K neighboring glyphs (K = 2*context_window):
      - context_window glyphs before (glyph_id - k, ..., glyph_id - 1)
      - context_window glyphs after (glyph_id + 1, ..., glyph_id + k)

    Boundary Handling:
    - If neighbors don't exist, fills with zeros (padding token assumed to be 0)
    - Returns context_mask indicating valid neighbors

    Args:
        glyph_ids: List of glyph_ids for this split
        grids_dir: Directory containing grids (memmap or per-file)
        label_map: Dict mapping label_string -> class_index
        glyph_to_label: Dict mapping glyph_id -> label_string
        context_window: Number of neighbors on each side (default: 2)
        diacritic_flags: Optional dict glyph_id -> bool
        cache: If True, cache loaded grids
        cast_dtype: Data type for grid tensors
        memmap_grids_file: Optional path to memmap grids file
        memmap_row_ids_file: Optional path to memmap row_ids file
    """

    MEMMAP_GRIDS = "grids_uint16.npy"
    MEMMAP_ROW_IDS = "glyph_row_ids.npy"

    def __init__(
        self,
        glyph_ids: List[int],
        grids_dir: Path,
        label_map: Dict[str, int],
        glyph_to_label: Dict[int, str],
        context_window: int = 2,
        diacritic_flags: Optional[Dict[int, bool]] = None,
        cache: bool = False,
        cast_dtype: torch.dtype = torch.int64,
        memmap_grids_file: Optional[Path] = None,
        memmap_row_ids_file: Optional[Path] = None,
    ):
        self.glyph_ids = glyph_ids
        self.grids_dir = grids_dir
        self.label_map = label_map
        self.glyph_to_label = glyph_to_label
        self.context_window = context_window
        self.diacritic_flags = diacritic_flags or {}
        self.cache_enabled = cache
        self.cast_dtype = cast_dtype
        self._cache: Dict[int, torch.Tensor] = {}

        # Memmap detection and setup
        self._memmap_mode = False
        mm_grid_path = memmap_grids_file or (grids_dir / self.MEMMAP_GRIDS)
        mm_row_ids_path = memmap_row_ids_file or (grids_dir / self.MEMMAP_ROW_IDS)

        if mm_grid_path.exists() and mm_row_ids_path.exists():
            row_ids = np.load(mm_row_ids_path)
            mm = np.memmap(mm_grid_path, dtype=np.uint16, mode="r")
            if mm.size % 256 != 0:
                raise RuntimeError(f"Memmap size not divisible by 256: {mm.size}")
            total = mm.size // 256
            mm = mm.reshape(total, 16, 16)
            if len(row_ids) != total:
                raise RuntimeError("row_ids length mismatch memmap shape.")
            self._memmap_mode = True
            self._mm = mm
            self._row_ids = row_ids
            self._id_to_pos = {int(row_ids[i]): i for i in range(len(row_ids))}

        # Filter out glyphs without labels
        missing = [gid for gid in self.glyph_ids if gid not in self.glyph_to_label]
        if missing:
            print(
                f"[warn] {len(missing)} glyph_ids missing label mapping; skipped.",
                file=sys.stderr,
            )
            self.glyph_ids = [
                gid for gid in self.glyph_ids if gid in self.glyph_to_label
            ]

        # Filter out glyphs not in memmap
        if self._memmap_mode:
            filtered = [gid for gid in self.glyph_ids if gid in self._id_to_pos]
            dropped = len(self.glyph_ids) - len(filtered)
            if dropped:
                print(
                    f"[warn] {dropped} glyph_ids not present in memmap index; dropped.",
                    file=sys.stderr,
                )
            self.glyph_ids = filtered

        if len(self.glyph_ids) == 0:
            raise RuntimeError("GlyphGridSequenceDataset has zero usable glyph ids.")

        # Build sorted list of all available glyph_ids for context lookup
        if self._memmap_mode:
            self._all_glyph_ids_sorted = sorted(self._id_to_pos.keys())
        else:
            # For per-file mode, scan directory
            self._all_glyph_ids_sorted = sorted(self.glyph_to_label.keys())

        self._glyph_id_set = set(self._all_glyph_ids_sorted)

    def __len__(self):
        return len(self.glyph_ids)

    def _load_grid_u16(self, path: Path) -> np.ndarray:
        raw = np.fromfile(path, dtype=np.uint16)
        if raw.size != 256:
            raise ValueError(f"Corrupt .u16 grid {path} size={raw.size}")
        return raw.reshape(16, 16)

    def _load_grid_file(self, gid: int) -> torch.Tensor:
        u16_path = self.grids_dir / f"{gid}.u16"
        npy_path = self.grids_dir / f"{gid}.npy"
        if u16_path.exists():
            arr = self._load_grid_u16(u16_path)
        elif npy_path.exists():
            arr = np.load(npy_path)
            if arr.shape != (16, 16):
                raise ValueError(f"Grid shape mismatch {npy_path}: {arr.shape}")
            arr = arr.astype(np.uint16)
        else:
            # Return zeros if file doesn't exist (for context padding)
            return torch.zeros((16, 16), dtype=torch.int64)
        return torch.from_numpy(arr.astype("int64"))

    def _load_grid_memmap(self, gid: int) -> torch.Tensor:
        pos = self._id_to_pos.get(gid)
        if pos is None:
            # Return zeros for padding
            return torch.zeros((16, 16), dtype=torch.int64)
        arr = self._mm[pos]
        return torch.from_numpy(arr.astype("int64"))

    def _load_grid(self, gid: int) -> torch.Tensor:
        if self.cache_enabled and gid in self._cache:
            return self._cache[gid]
        t = (
            self._load_grid_memmap(gid)
            if self._memmap_mode
            else self._load_grid_file(gid)
        )
        if self.cache_enabled:
            self._cache[gid] = t
        return t

    def _get_context_neighbors(self, center_gid: int) -> Tuple[List[int], List[int]]:
        """
        Get context neighbor glyph_ids and their deltas relative to center.

        Returns:
            neighbor_gids: List of K glyph_ids (may include 0 for padding)
            deltas: List of K deltas (positive for after, negative for before)
        """
        K = 2 * self.context_window
        neighbor_gids = []
        deltas = []

        # Get neighbors before center
        for offset in range(-self.context_window, 0):
            neighbor_gid = center_gid + offset
            if neighbor_gid > 0 and neighbor_gid in self._glyph_id_set:
                neighbor_gids.append(neighbor_gid)
                deltas.append(offset)
            else:
                # Padding
                neighbor_gids.append(0)
                deltas.append(offset)

        # Get neighbors after center
        for offset in range(1, self.context_window + 1):
            neighbor_gid = center_gid + offset
            if neighbor_gid in self._glyph_id_set:
                neighbor_gids.append(neighbor_gid)
                deltas.append(offset)
            else:
                # Padding
                neighbor_gids.append(0)
                deltas.append(offset)

        return neighbor_gids, deltas

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        torch.Tensor,  # center_grid
        torch.Tensor,  # context_grids
        torch.Tensor,  # context_deltas
        int,  # label_idx
        int,  # glyph_id
        bool,  # is_diacritic
    ]:
        gid = self.glyph_ids[idx]

        # Get label
        label_str = self.glyph_to_label.get(gid)
        if label_str is None:
            raise KeyError(f"Missing label for glyph_id={gid}")
        label_idx = self.label_map.get(label_str)
        if label_idx is None:
            raise KeyError(
                f"Label '{label_str}' not found in label_map.json (glyph_id={gid})"
            )

        # Load center glyph
        center_grid = self._load_grid(gid)

        # Load context neighbors
        neighbor_gids, deltas = self._get_context_neighbors(gid)
        K = len(neighbor_gids)

        # Load all neighbor grids
        context_grids = []
        for ngid in neighbor_gids:
            if ngid == 0:
                # Padding token
                context_grids.append(torch.zeros((16, 16), dtype=torch.int64))
            else:
                context_grids.append(self._load_grid(ngid))

        # Stack context grids: (K, 16, 16)
        context_grids_tensor = torch.stack(context_grids, dim=0)

        # Context deltas: (K,)
        context_deltas_tensor = torch.tensor(deltas, dtype=torch.long)

        # Diacritic flag
        is_diacritic = bool(self.diacritic_flags.get(gid, False))

        return (
            center_grid.to(self.cast_dtype),
            context_grids_tensor.to(self.cast_dtype),
            context_deltas_tensor,
            label_idx,
            gid,
            is_diacritic,
        )


def collate_sequence_batch(batch):
    """
    Custom collate function for sequence dataset.

    Args:
        batch: List of tuples from __getitem__

    Returns:
        Batched tensors:
        - center_grids: (B, 16, 16)
        - context_grids: (B, K, 16, 16)
        - context_deltas: (B, K)
        - labels: (B,)
        - glyph_ids: (B,)
        - diacritic_flags: (B,)
    """
    center_grids = []
    context_grids = []
    context_deltas = []
    labels = []
    glyph_ids = []
    diacritic_flags = []

    for item in batch:
        center_grid, ctx_grids, ctx_deltas, label, gid, is_dia = item
        center_grids.append(center_grid)
        context_grids.append(ctx_grids)
        context_deltas.append(ctx_deltas)
        labels.append(label)
        glyph_ids.append(gid)
        diacritic_flags.append(is_dia)

    return (
        torch.stack(center_grids, dim=0),  # (B, 16, 16)
        torch.stack(context_grids, dim=0),  # (B, K, 16, 16)
        torch.stack(context_deltas, dim=0),  # (B, K)
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(glyph_ids, dtype=torch.long),
        torch.tensor(diacritic_flags, dtype=torch.bool),
    )


__all__ = [
    "GlyphGridSequenceDataset",
    "collate_sequence_batch",
]
