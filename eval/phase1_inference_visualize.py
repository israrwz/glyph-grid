#!/usr/bin/env python3
"""
Phase 1 Inference & Visualization (Prototype)

Generates a composite diagnostic image for one or more glyph rasters by:
  1. Loading the latest Phase 1 CNN checkpoint (unless a specific file supplied).
  2. Loading primitive centroids (with EMPTY at index 0).
  3. For each provided 128x128 glyph raster:
       - Splitting into 16x16 grid of 8x8 cells.
       - For each cell, computing nearest centroid (skipping class 0 if cell empty).
       - Overlaying the centroid's bitmap (thresholded) inside that cell region
         in semi‑transparent red (alpha=128) on top of the original glyph raster.
  4. Producing a single output image (side‑by‑side panel) OR individual images.

Intended Usage (example):
  python -m eval.phase1_inference_visualize \
      --checkpoint-dir checkpoints/phase1 \
      --centroids assets/centroids/primitive_centroids.npy \
      --glyphs examples/glyph_*.png \
      --out output/phase1_overlay.png

Accepted glyph inputs:
  - .png (8-bit grayscale; auto-thresholded at >0)
  - .npy (shape (128,128) uint8 or bool)
  - (Optional future) .pt tensor files (single 128x128)

Dependencies:
  - torch
  - numpy
  - pillow (PIL)
  - pyyaml (only if loading phase1.yaml to locate assets automatically)

Assumptions:
  - Centroids stored as float32 array shape (K, 64) with row 0 = EMPTY (all zeros).
  - Model architecture matches models/phase1_cnn.BaselinePhase1CNN (config-aligned).
  - Cells are strictly 8x8; glyph raster is 128x128 (plan-compliant).
  - Input glyphs are "unseen" (not required, but typical for evaluation).

NOTE:
  This is a standalone visualization helper; it does not modify training artifacts.
  For large batch evaluation, adapt into a batched dataloader path.

Future Enhancements (planned in progress.md):
  - Optionally show predicted centroid ID text overlay per cell.
  - Add XOR(cell, centroid) heatmap channel.
  - Export JSON of per-cell primitive assignments.

Author: Auto-generated scaffold.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import torch
except ImportError as e:
    raise RuntimeError("PyTorch is required to run this script.") from e

# ---------------------------------------------------------------------------
# Model Import (lazy to allow use outside project root if PYTHONPATH set)
# ---------------------------------------------------------------------------
try:
    from models.phase1_cnn import build_phase1_model
except ImportError:
    build_phase1_model = None  # We fallback with a clearer error later


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def find_latest_checkpoint(ckpt_dir: Path, pattern: str = "*.pt") -> Optional[Path]:
    matches = list(ckpt_dir.glob(pattern))
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def load_centroids(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 64:
        raise ValueError(f"Centroids shape mismatch: expected (K,64), got {arr.shape}")
    return arr.astype(np.float32)


def load_model(checkpoint_path: Path, num_classes: int = 1024) -> torch.nn.Module:
    """
    Load a Phase 1 CNN checkpoint, robust to different wrapping conventions:
      - {'model_state': ...}
      - {'model_state_dict': ...}
      - raw state_dict (keys like 'features.0.weight', 'classifier.4.bias', ...)
    Falls back to raising a descriptive error if none of these patterns match.
    """
    if build_phase1_model is None:
        raise RuntimeError(
            "Could not import build_phase1_model. Ensure PYTHONPATH includes project root."
        )

    cfg = {
        "in_channels": 1,
        "num_classes": num_classes,
        "conv_blocks": [
            {
                "out_channels": 32,
                "kernel": 3,
                "stride": 1,
                "padding": 1,
                "batchnorm": True,
                "pool": 2,
            },
            {
                "out_channels": 64,
                "kernel": 3,
                "stride": 1,
                "padding": 1,
                "batchnorm": True,
                "pool": 2,
            },
        ],
        "flatten_dim": 256,
        "fc_hidden": 128,
        "fc_dropout": 0.2,
        "weight_init": "none",  # We will overwrite with checkpoint weights
    }
    model = build_phase1_model(cfg)

    raw = torch.load(checkpoint_path, map_location="cpu")

    # Determine actual state_dict
    if isinstance(raw, dict):
        if "model_state" in raw and isinstance(raw["model_state"], dict):
            state_dict = raw["model_state"]
        elif "model_state_dict" in raw and isinstance(raw["model_state_dict"], dict):
            state_dict = raw["model_state_dict"]
        else:
            # Heuristic: does this look like a raw state_dict (contains parameter tensors)?
            param_like_keys = [
                k
                for k in raw.keys()
                if isinstance(k, str) and ("features." in k or "classifier." in k)
            ]
            if param_like_keys:
                state_dict = raw  # treat as direct state_dict
            else:
                raise RuntimeError(
                    f"Unrecognized checkpoint format. Keys: {list(raw.keys())[:10]}. "
                    "Expected 'model_state', 'model_state_dict', or direct parameter tensors."
                )
    else:
        raise RuntimeError(
            f"Unsupported checkpoint object type: {type(raw)} (expected dict)."
        )

    # Load (strict True to catch architecture mismatches; relax if user requests)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def load_glyph(path: Path) -> np.ndarray:
    """
    Returns binary numpy array shape (128,128), dtype uint8 {0,255}.
    """
    if path.suffix.lower() == ".png":
        img = Image.open(path).convert("L").resize((128, 128), Image.NEAREST)
        arr = np.array(img, dtype=np.uint8)
        bin_arr = (arr > 0).astype(np.uint8) * 255
        return bin_arr
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        if arr.shape != (128, 128):
            raise ValueError(f"Unexpected npy raster shape {arr.shape} for {path}")
        if arr.dtype != np.uint8:
            if arr.dtype == bool:
                arr = arr.astype(np.uint8) * 255
            else:
                # Normalize / threshold
                arr = (arr > (arr.max() / 2)).astype(np.uint8) * 255
        # Ensure binary
        arr = (arr > 0).astype(np.uint8) * 255
        return arr
    raise ValueError(f"Unsupported glyph file extension: {path}")


def iter_cells(raster: np.ndarray) -> Sequence[Tuple[int, int, np.ndarray]]:
    """
    Yield (row_idx, col_idx, 8x8 patch) for a 128x128 raster.
    """
    if raster.shape != (128, 128):
        raise ValueError(f"Expected raster (128,128); got {raster.shape}")
    cells = []
    cell_size = 8
    for r in range(0, 128, cell_size):
        for c in range(0, 128, cell_size):
            patch = raster[r : r + cell_size, c : c + cell_size]
            cells.append((r // cell_size, c // cell_size, patch))
    return cells


def cell_to_tensor(cell: np.ndarray) -> torch.Tensor:
    # Convert 8x8 binary {0,255} to float tensor (1,8,8) with 0/1
    t = torch.from_numpy((cell > 0).astype("float32")).unsqueeze(0)
    return t  # (1,8,8)


def nearest_centroid_id(cell_patch: np.ndarray, centroids: np.ndarray) -> int:
    """
    Find nearest centroid by L2 distance.
    centroids: (K,64), index 0 = EMPTY
    Returns centroid ID (int).
    """
    flat = (cell_patch > 0).astype(np.float32).reshape(-1)  # (64,)
    # Early empty detection
    if flat.sum() == 0:
        return 0
    # Compute distances to non-empty only (skip index 0)
    non_empty = centroids[1:]
    # (flat - c)^2 = flat^2 - 2 flat·c + c^2 => we just brute force; small K (<2k)
    diffs = non_empty - flat
    dists = np.einsum("ij,ij->i", diffs, diffs)
    nearest = int(np.argmin(dists)) + 1
    return nearest


def overlay_centroids(
    raster: np.ndarray,
    centroid_ids: np.ndarray,
    centroids: np.ndarray,
    alpha: int = 128,
    skip_empty: bool = True,
) -> Image.Image:
    """
    Compose overlay image: original glyph in grayscale + centroid patterns in red.
    centroid_ids: (16,16) int array of predicted primitive IDs.
    """
    base = Image.fromarray(raster, mode="L").convert("RGBA")
    overlay = Image.new("RGBA", (128, 128), (0, 0, 0, 0))

    # Precompute centroid bitmaps (thresholded)
    centroid_bitmaps = {}
    for cid in np.unique(centroid_ids):
        if skip_empty and cid == 0:
            continue
        if cid < 0 or cid >= len(centroids):
            continue
        vec = centroids[cid]
        if vec.shape[0] != 64:
            continue
        bmp = (vec.reshape(8, 8) >= 0.5).astype(np.uint8)  # boolean mask
        centroid_bitmaps[cid] = bmp

    cell_size = 8
    for cell_r in range(16):
        for cell_c in range(16):
            cid = centroid_ids[cell_r, cell_c]
            if skip_empty and cid == 0:
                continue
            bmp = centroid_bitmaps.get(cid)
            if bmp is None:
                continue
            # Create RGBA patch
            rgba = np.zeros((8, 8, 4), dtype=np.uint8)
            # Red channel where centroid mask is 1
            rgba[..., 0] = bmp * 255
            rgba[..., 3] = bmp * alpha  # alpha
            patch_img = Image.fromarray(rgba, mode="RGBA")
            x0 = cell_c * cell_size
            y0 = cell_r * cell_size
            overlay.paste(patch_img, (x0, y0), patch_img)

    composite = Image.alpha_composite(base, overlay)
    return composite


def process_glyph(
    glyph_path: Path,
    model: torch.nn.Module,
    centroids: np.ndarray,
    device: torch.device,
) -> Image.Image:
    raster = load_glyph(glyph_path)  # (128,128) uint8 {0,255}
    # Build centroid id grid (16x16)
    ids = np.zeros((16, 16), dtype=np.int32)
    with torch.no_grad():
        for r, c, patch in iter_cells(raster):
            tensor = cell_to_tensor(patch).unsqueeze(0).to(device)  # (1,1,8,8)
            logits = model(tensor)
            pred = int(torch.argmax(logits, dim=1).item())
            # Optional: cross-check with direct nearest centroid:
            # (Uncomment below if you want to verify mapping)
            # nn_id = nearest_centroid_id(patch, centroids)
            ids[r, c] = pred
    # Overlay predicted centroid shapes
    composite = overlay_centroids(raster, ids, centroids, alpha=128)
    return composite


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class Args:
    checkpoint_dir: Path
    checkpoint_file: Optional[Path]
    centroids: Path
    glyphs: List[Path]
    out: Path
    panel: bool
    width_cells: int
    verbose: bool


def parse_args(argv: Sequence[str]) -> Args:
    ap = argparse.ArgumentParser(
        description="Phase 1 inference visualization (centroid overlays)"
    )
    ap.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/phase1"),
        help="Directory containing *.pt checkpoints; latest is used if --checkpoint-file omitted.",
    )
    ap.add_argument(
        "--checkpoint-file",
        type=Path,
        default=None,
        help="Explicit checkpoint file. Overrides --checkpoint-dir if provided.",
    )
    ap.add_argument(
        "--centroids",
        type=Path,
        default=Path("assets/centroids/primitive_centroids.npy"),
        help="Path to centroids with EMPTY row at index 0.",
    )
    ap.add_argument(
        "--glyphs",
        type=str,
        nargs="+",
        required=True,
        help="List of glyph raster paths or glob patterns (PNG or NPY).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("output/phase1_overlay.png"),
        help="Output image path (if --panel). If not panel, used as directory.",
    )
    ap.add_argument(
        "--no-panel",
        dest="panel",
        action="store_false",
        help="If set, saves individual images instead of a combined panel.",
    )
    ap.add_argument(
        "--width-cells",
        type=int,
        default=4,
        help="Number of glyph composites per row in panel mode.",
    )
    ap.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose logging to stderr."
    )
    ap.set_defaults(panel=True)
    ns = ap.parse_args(argv)

    # Expand glob patterns
    glyph_paths: List[Path] = []
    for g in ns.glyphs:
        if any(ch in g for ch in "*?[]"):
            for match in glob.glob(g):
                glyph_paths.append(Path(match))
        else:
            glyph_paths.append(Path(g))
    glyph_paths = [p for p in glyph_paths if p.exists()]
    if not glyph_paths:
        ap.error("No glyph files matched / found.")

    return Args(
        checkpoint_dir=ns.checkpoint_dir,
        checkpoint_file=ns.checkpoint_file,
        centroids=ns.centroids,
        glyphs=glyph_paths,
        out=ns.out,
        panel=ns.panel,
        width_cells=ns.width_cells,
        verbose=ns.verbose,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.verbose:
        print(f"[info] Device: {device}", file=sys.stderr)

    if args.checkpoint_file:
        ckpt = args.checkpoint_file
    else:
        ckpt = find_latest_checkpoint(args.checkpoint_dir)
    if not ckpt or not ckpt.exists():
        print(
            f"[error] No checkpoint found (searched: {args.checkpoint_dir}). "
            "Use --checkpoint-file to specify explicitly.",
            file=sys.stderr,
        )
        return 2
    if args.verbose:
        print(f"[info] Using checkpoint: {ckpt}", file=sys.stderr)

    if not args.centroids.exists():
        print(f"[error] Centroids not found: {args.centroids}", file=sys.stderr)
        return 3

    centroids = load_centroids(args.centroids)
    model = load_model(ckpt, num_classes=centroids.shape[0]).to(device)

    composites: List[Image.Image] = []
    for gpath in args.glyphs:
        try:
            comp = process_glyph(gpath, model, centroids, device)
            composites.append(comp)
            if not args.panel:
                # Save individually
                out_dir = args.out
                out_dir.mkdir(parents=True, exist_ok=True)
                save_path = out_dir / f"{gpath.stem}_overlay.png"
                comp.save(save_path)
                if args.verbose:
                    print(f"[ok] Saved {save_path}", file=sys.stderr)
        except Exception as e:
            print(f"[warn] Failed processing {gpath}: {e}", file=sys.stderr)

    if args.panel:
        # Arrange composites in a grid
        if not composites:
            print("[error] No composites generated.", file=sys.stderr)
            return 4
        per_row = max(1, args.width_cells)
        cell_w, cell_h = composites[0].size
        rows = (len(composites) + per_row - 1) // per_row
        panel = Image.new("RGBA", (cell_w * per_row, cell_h * rows), color=(0, 0, 0, 0))
        for idx, img in enumerate(composites):
            r = idx // per_row
            c = idx % per_row
            panel.paste(img, (c * cell_w, r * cell_h))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        panel.save(args.out)
        if args.verbose:
            print(f"[ok] Panel saved to {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
