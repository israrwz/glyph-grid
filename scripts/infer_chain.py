#!/usr/bin/env python3
"""
Two-Phase Glyph Inference Script
================================

Performs hierarchical inference:
  Phase 1: primitive cell classification (8x8 -> primitive ID 0..1023)
  Phase 2: glyph classification from 16x16 primitive ID grid

Features:
  - Supports CNN or Transformer Phase 2 architectures.
  - Optional external phase2 YAML config to reconstruct architecture.
  - Maps predicted glyph label -> base_unicode (from data/chars.csv).
  - Top‑K prediction output.
  - Optional JSONL export of full results.
  - Batched Phase 1 cell inference for speed.
  - Graceful fallbacks if checkpoint lacks embedded config.

Usage Example:
  python scripts/infer_chain.py \
      --rasters_dir data/rasters \
      --phase1_ckpt checkpoints/phase1/best.pt \
      --phase2_ckpt checkpoints/phase2/best.pt \
      --label_map data/grids_memmap/label_map.json \
      --chars_csv data/chars.csv \
      --arch cnn \
      --limit 5 \
      --topk 5 \
      --output_json predictions.jsonl

To auto-reconstruct Phase 2 model from a YAML config:
  python scripts/infer_chain.py ... --config_yaml configs/phase2.yaml

Output format (console):
  <filename>: top1_label=<label> top1_base=<char> prob=<p> | topk=[(label,base,p_float), ...]

JSONL (if --output_json provided):
  {"file":"...","topk":[{"rank":1,"label":"...","base":"...", "prob":0.9876}, ...]}

Requirements:
  - Project root on PYTHONPATH or run from repository root.
  - Checkpoints produced by training scripts (model_state dict).

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
from torch import nn
import random


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Project-local imports (factories)
import os, sys

# Add repo root to sys.path so "models" package resolves even when script is run from a nested working directory (e.g., Kaggle).
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from models.phase1_cnn import build_phase1_model
from models.phase2_cnn import build_phase2_cnn_model
from models.phase2_transformer import build_phase2_model

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def load_label_map(path: Path) -> Tuple[Dict[str, int], List[str]]:
    """
    label_map.json: { "label_string": class_index, ... }
    Returns:
      map (label->idx), inverse list (idx->label)
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    inv = [None] * len(payload)
    for lbl, idx in payload.items():
        if 0 <= idx < len(inv):
            inv[idx] = lbl
    # Fill any gaps with placeholder
    for i, v in enumerate(inv):
        if v is None:
            inv[i] = f"<UNK_{i}>"
    return payload, inv


def load_chars_csv(path: Path) -> Dict[str, str]:
    """
    Build mapping label -> base_unicode (first occurrence wins).
    chars.csv columns include: codepoint, base_unicode, joining_group, char_class, label
    """
    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl = row.get("label")
            base = row.get("base_unicode")
            if not lbl or not base:
                continue
            if lbl not in mapping:
                mapping[lbl] = base
    return mapping


def safe_load_yaml(path: Optional[Path]) -> Dict[str, Any]:
    if not path:
        return {}
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Phase 1 (Primitive) Model Loader
# ---------------------------------------------------------------------------
def load_phase1_model(ckpt_path: Path) -> nn.Module:
    """
    Reconstruct baseline Phase 1 CNN with default config; load weights.
    """
    base_cfg = {
        "in_channels": 1,
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
        "num_classes": 1024,
        "weight_init": "kaiming_normal",
    }
    model = build_phase1_model(base_cfg)
    payload = torch.load(str(ckpt_path), map_location="cpu")
    state = payload.get("model_state") or payload
    missing = model.load_state_dict(state, strict=False)
    if missing.missing_keys:
        print(f"[warn] Phase1 missing keys: {missing.missing_keys}", file=sys.stderr)
    model.to(DEVICE).eval()
    return model


# ---------------------------------------------------------------------------
# Phase 2 Model Loader
# ---------------------------------------------------------------------------
def infer_arch_arg_or_yaml(arch_arg: Optional[str], yaml_cfg: Dict[str, Any]) -> str:
    if arch_arg:
        return arch_arg.lower()
    model_root = yaml_cfg.get("model", {}) if isinstance(yaml_cfg, dict) else {}
    arch = model_root.get("architecture")
    return (arch or "cnn").lower()


def build_phase2_from_yaml(yaml_cfg: Dict[str, Any], num_labels: int) -> nn.Module:
    """
    Attempt to reconstruct model from YAML config dict.
    """
    arch = infer_arch_arg_or_yaml(None, yaml_cfg)
    if arch == "cnn":
        return build_phase2_cnn_model(
            yaml_cfg, num_labels=num_labels, primitive_centroids=None
        )
    return build_phase2_model(yaml_cfg, num_labels=num_labels, primitive_centroids=None)


def build_phase2_fallback(arch: str, num_labels: int) -> nn.Module:
    """
    Fallback minimal config if YAML absent.
    Mirrors upgraded CNN config (embedding_dim=96, larger stages).
    """
    if arch == "cnn":
        cfg = {
            "input": {
                "primitive_vocab_size": 1024,
                "embedding_dim": 96,
                "normalize_embeddings": False,
            },
            "model": {
                "architecture": "cnn",
                "cnn": {
                    "stages": [96, 192, 256],
                    "blocks_per_stage": [3, 3, 3],
                    "kernel_size": 3,
                    "stem_kernel_size": 3,
                    "stem_stride": 1,
                    "downsample": "conv",
                    "activation": "gelu",
                    "dropout": 0.15,
                    "classifier_hidden_dim": 256,
                    "classifier_dropout": 0.30,
                },
                "init": {
                    "embedding_from_centroids": False,
                    "centroid_requires_grad": True,
                    "weight_init": "xavier_uniform",
                },
            },
        }
        return build_phase2_cnn_model(
            cfg, num_labels=num_labels, primitive_centroids=None
        )
    # Transformer fallback (baseline)
    cfg_t = {
        "input": {
            "primitive_vocab_size": 1024,
            "embedding_dim": 64,
            "positional_encoding": "sinusoidal_2d",
            "combine_mode": "add",
            "patch_grouping": {"enabled": True, "patch_rows": 4, "patch_cols": 4},
            "token_pooling": "cls",
            "use_cls_token": True,
            "normalize_embeddings": False,
        },
        "model": {
            "architecture": "transformer",
            "transformer": {
                "num_layers": 5,
                "d_model": 256,
                "num_heads": 8,
                "mlp_hidden_dim": 512,
                "dropout": 0.1,
                "attention_dropout": 0.1,
                "layer_norm_eps": 1e-5,
                "pre_norm": True,
            },
            "classifier": {"hidden_dim": 256, "dropout": 0.1, "activation": "gelu"},
            "init": {
                "embedding_from_centroids": False,
                "centroid_requires_grad": True,
                "weight_init": "xavier_uniform",
                "cls_init": "normal",
            },
        },
    }
    return build_phase2_model(cfg_t, num_labels=num_labels, primitive_centroids=None)


def load_phase2_model(
    ckpt_path: Path,
    num_labels: int,
    arch_arg: Optional[str],
    yaml_cfg: Dict[str, Any],
) -> nn.Module:
    """
    Build model (from YAML if provided) then load weights.
    """
    arch = infer_arch_arg_or_yaml(arch_arg, yaml_cfg)
    if yaml_cfg:
        model = build_phase2_from_yaml(yaml_cfg, num_labels)
    else:
        model = build_phase2_fallback(arch, num_labels)

    payload = torch.load(str(ckpt_path), map_location="cpu")
    state = payload.get("model_state") or payload
    missing = model.load_state_dict(state, strict=False)
    if missing.missing_keys:
        # Suppress verbose missing key dump; silent here to allow higher-level fallback logic.
        pass
    model.to(DEVICE).eval()
    return model


# ---------------------------------------------------------------------------
# Raster -> Primitive Grid
# ---------------------------------------------------------------------------
def raster_to_primitive_grid(
    img_path: Path,
    phase1_model: nn.Module,
    normalize_uint8: bool = True,
    empty_threshold: float = 0.0,
) -> torch.Tensor:
    """
    Converts a 128x128 raster to a (16,16) primitive ID grid using Phase 1 model.

    Empty cell masking:
      Any 8x8 patch whose raw pixel max == 0 (fully empty) OR whose normalized
      sum <= empty_threshold (default 0.0) is forced to primitive ID 0 after prediction.

    Returns:
      Tensor (16,16) int64 of primitive IDs.
    """
    img = Image.open(img_path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    if arr.shape != (128, 128):
        raise ValueError(f"Expected 128x128 image; got {arr.shape} for {img_path.name}")

    patches: List[torch.Tensor] = []
    empties: List[bool] = []
    for gy in range(16):
        for gx in range(16):
            patch_raw = arr[gy * 8 : (gy + 1) * 8, gx * 8 : (gx + 1) * 8]
            is_empty = patch_raw.max() == 0
            t = torch.from_numpy(patch_raw)  # (8,8)
            if normalize_uint8 and t.max() > 1:
                t = (t.float() / 255.0).to(torch.float32)
            else:
                t = t.to(torch.float32)
            # Optional secondary emptiness check after normalization
            if (
                not is_empty
                and empty_threshold > 0.0
                and t.sum().item() <= empty_threshold
            ):
                is_empty = True
            empties.append(is_empty)
            t = t.unsqueeze(0).unsqueeze(0)  # (1,1,8,8)
            patches.append(t)
    batch = torch.cat(patches, dim=0).to(DEVICE)  # (256,1,8,8)

    with torch.no_grad():
        logits = phase1_model(batch)  # (256, num_primitives)
        preds = torch.argmax(logits, dim=1).view(16, 16).cpu()

    # Apply empty cell masking: force primitive ID 0 for empty patches
    empty_mask = torch.tensor(empties, dtype=torch.bool).view(16, 16)
    preds[empty_mask] = 0

    return preds.to(torch.int64)


# ---------------------------------------------------------------------------
# Phase 2 Prediction
# ---------------------------------------------------------------------------
def predict_glyph_grid(
    grid: torch.Tensor,
    phase2_model: nn.Module,
    topk: int = 5,
) -> Tuple[List[int], List[float]]:
    """
    grid: (16,16) int64 primitive IDs
    Returns:
      topk_indices, topk_probs
    """
    if grid.shape != (16, 16):
        raise ValueError(f"Expected grid shape (16,16); got {tuple(grid.shape)}")
    input_tensor = grid.unsqueeze(0).to(torch.long).to(DEVICE)  # (1,16,16)

    with torch.no_grad():
        logits = phase2_model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        values, indices = torch.topk(probs, k=topk, dim=1)

    return indices.squeeze(0).cpu().tolist(), values.squeeze(0).cpu().tolist()


# ---------------------------------------------------------------------------
# JSONL Writing
# ---------------------------------------------------------------------------
def write_jsonl(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    append: bool = False,
) -> None:
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Two-phase glyph inference")
    ap.add_argument(
        "--rasters_dir",
        type=Path,
        required=True,
        help="Directory of 128x128 raster PNG files.",
    )
    ap.add_argument(
        "--phase1_ckpt",
        type=Path,
        required=True,
        help="Phase 1 primitive classifier checkpoint.",
    )
    ap.add_argument(
        "--phase2_ckpt",
        type=Path,
        required=True,
        help="Phase 2 glyph classifier checkpoint.",
    )
    ap.add_argument(
        "--label_map", type=Path, required=True, help="Path to label_map.json."
    )
    ap.add_argument(
        "--chars_csv",
        type=Path,
        required=True,
        help="Path to chars.csv for label->base_unicode mapping.",
    )
    ap.add_argument(
        "--arch",
        type=str,
        default=None,
        choices=["cnn", "transformer"],
        help="Override Phase 2 architecture (ignored if --baseline_phase2 set).",
    )
    ap.add_argument(
        "--config_yaml",
        type=Path,
        default=None,
        help="Optional YAML config to reconstruct Phase 2 model.",
    )
    ap.add_argument(
        "--limit", type=int, default=0, help="Limit number of rasters (0=all)."
    )
    ap.add_argument("--topk", type=int, default=5, help="Top-K predictions to output.")
    ap.add_argument(
        "--output_json", type=Path, default=None, help="Optional JSONL output path."
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file console output (JSON only).",
    )
    ap.add_argument(
        "--show_grid",
        action="store_true",
        help="Print primitive grid (text) for debugging.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (shuffling, torch, numpy).",
    )
    ap.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle raster processing order after sorting.",
    )
    ap.add_argument(
        "--baseline_phase2",
        action="store_true",
        help="Force loading Phase 2 checkpoint with baseline CNN architecture (embedding_dim=64, stages [64,128,192]).",
    )
    ap.add_argument(
        "--all-classes",
        action="store_true",
        help="Iterate over all glyph classes; for each class sample up to --limit rasters matching that class label base. Uses filename prefix before last underscore for matching.",
    )
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # Validate paths
    for p_attr in (
        "rasters_dir",
        "phase1_ckpt",
        "phase2_ckpt",
        "label_map",
        "chars_csv",
    ):
        p: Path = getattr(args, p_attr)
        if not p.exists():
            print(f"[error] Missing required path: {p_attr}={p}", file=sys.stderr)
            sys.exit(2)

    # Load label map and chars mapping
    label_map, inv_labels = load_label_map(args.label_map)
    base_unicode_map = load_chars_csv(args.chars_csv)

    # Set seed
    set_seed(args.seed)

    # Build models
    phase1 = load_phase1_model(args.phase1_ckpt)
    yaml_cfg = safe_load_yaml(args.config_yaml)

    if args.baseline_phase2:
        # Baseline CNN spec matching original smaller architecture (to avoid missing keys)
        baseline_cfg = {
            "input": {
                "primitive_vocab_size": 1024,
                "embedding_dim": 64,
                "normalize_embeddings": False,
            },
            "model": {
                "architecture": "cnn",
                "cnn": {
                    "stages": [64, 128, 192],
                    "blocks_per_stage": [2, 2, 2],
                    "kernel_size": 3,
                    "stem_kernel_size": 3,
                    "stem_stride": 1,
                    "downsample": "conv",
                    "activation": "gelu",
                    "dropout": 0.1,
                    "classifier_hidden_dim": 0,
                    "classifier_dropout": 0.2,
                },
                "init": {
                    "embedding_from_centroids": False,
                    "centroid_requires_grad": True,
                    "weight_init": "xavier_uniform",
                },
            },
        }
        phase2 = build_phase2_cnn_model(
            baseline_cfg, num_labels=len(inv_labels), primitive_centroids=None
        )
        payload = torch.load(str(args.phase2_ckpt), map_location="cpu")
        state = payload.get("model_state") or payload
        phase2.load_state_dict(state, strict=False)
        phase2.to(DEVICE).eval()
    else:
        phase2 = None
        missing_after = None

        # Utility: strip common prefixes from checkpoint keys
        def _sanitize_state_keys(raw_state: dict) -> dict:
            cleaned = {}
            for k, v in raw_state.items():
                nk = k
                for prefix in ("model.", "module.", "net.", "_orig_mod."):
                    if nk.startswith(prefix):
                        nk = nk[len(prefix) :]
                cleaned[nk] = v
            return cleaned

        # Infer embedding_dim from checkpoint if possible
        def _infer_embedding_dim(state: dict) -> int:
            for cand in (
                "embedding.weight",
                "primitive_embedding.weight",
                "_orig_mod.embedding.weight",
            ):
                if cand in state and state[cand].dim() == 2:
                    return int(state[cand].shape[1])
            # Fallback: search any 2D tensor whose first dimension looks like vocab (900..1300)
            candidates = [
                t for t in state.values() if t.dim() == 2 and 900 <= t.shape[0] <= 1300
            ]
            if candidates:
                candidates.sort(key=lambda x: x.shape[1])
                return int(candidates[0].shape[1])
            return 64  # safe default

        # Infer CNN stages & blocks from conv weights; fallback to heuristic
        def _infer_cnn_stages(
            state: dict, embed_dim: int
        ) -> tuple[list[int], list[int]]:
            # Accept both sanitized keys (after _orig_mod removal) or raw prefixed
            stem_key_candidates = [k for k in state if k.endswith("stem.conv.weight")]
            stem_out = None
            for sk in stem_key_candidates:
                if state[sk].dim() == 4:
                    stem_out = int(state[sk].shape[0])
                    break

            # Collect conv1 weights inside residual blocks
            block_channels = []
            for k, v in state.items():
                if (
                    k.endswith("conv1.conv.weight")
                    and ".features." in k
                    and v.dim() == 4
                ):
                    block_channels.append(int(v.shape[0]))

            if stem_out is not None and block_channels:
                stages = [stem_out]
                counts = [0]
                for ch in block_channels:
                    if ch != stages[-1]:
                        stages.append(ch)
                        counts.append(1)
                    else:
                        counts[-1] += 1
                return stages, counts

            # Fallback heuristics based on embedding_dim (capacity vs baseline)
            if embed_dim >= 96:
                return [96, 192, 256], [3, 3, 3]
            return [64, 128, 192], [2, 2, 2]

        # Infer classifier hidden dim (any Linear whose out_features != num_labels but appears in classifier)
        def _infer_classifier_hidden_dim(state: dict, num_labels: int | None) -> int:
            if num_labels is None:
                num_labels = -1
            hidden = None
            for k, v in state.items():
                if "classifier" in k and k.endswith(".weight") and v.dim() == 2:
                    out_f, in_f = v.shape
                    if out_f != num_labels:
                        # Exclude embedding weights mistakenly matched
                        hidden = out_f
                        # Prefer the largest candidate (later layers)
            return hidden or 0

        # Primary load (baseline override or dynamic)
        if args.baseline_phase2:
            payload = torch.load(str(args.phase2_ckpt), map_location="cpu")
            raw_state = payload.get("model_state") or payload
            state = _sanitize_state_keys(raw_state)
            emb_dim = _infer_embedding_dim(state)
            stages, blocks = _infer_cnn_stages(state, emb_dim)
            baseline_cfg = {
                "input": {
                    "primitive_vocab_size": 1024,
                    "embedding_dim": emb_dim,
                    "normalize_embeddings": False,
                },
                "model": {
                    "architecture": "cnn",
                    "cnn": {
                        "stages": stages,
                        "blocks_per_stage": blocks,
                        "kernel_size": 3,
                        "stem_kernel_size": 3,
                        "stem_stride": 1,
                        "downsample": "conv",
                        "activation": "gelu",
                        "dropout": 0.1,
                        "classifier_hidden_dim": _infer_classifier_hidden_dim(
                            state, len(inv_labels)
                        ),
                        "classifier_dropout": 0.2
                        if _infer_classifier_hidden_dim(state, len(inv_labels)) == 0
                        else 0.30,
                    },
                    "init": {
                        "embedding_from_centroids": False,
                        "centroid_requires_grad": True,
                        "weight_init": "xavier_uniform",
                    },
                },
            }
            phase2 = build_phase2_cnn_model(
                baseline_cfg, num_labels=len(inv_labels), primitive_centroids=None
            )
            # Remap state keys to match model
            phase2.load_state_dict(state, strict=False)
            phase2.to(DEVICE).eval()
        else:
            # Attempt capacity/dynamic loader first
            phase2 = load_phase2_model(
                ckpt_path=args.phase2_ckpt,
                num_labels=len(inv_labels),
                arch_arg=args.arch,
                yaml_cfg=yaml_cfg,
            )
            try:
                payload = torch.load(str(args.phase2_ckpt), map_location="cpu")
                raw_state = payload.get("model_state") or payload
                state = _sanitize_state_keys(raw_state)
                model_keys = set(phase2.state_dict().keys())
                missing_after = [k for k in model_keys if k not in state]
                if "embedding.weight" in missing_after and args.arch != "transformer":
                    # Fallback to inferred baseline using checkpoint's actual embedding/stage structure
                    emb_dim = _infer_embedding_dim(state)
                    stages, blocks = _infer_cnn_stages(state, emb_dim)
                    inferred_cfg = {
                        "input": {
                            "primitive_vocab_size": 1024,
                            "embedding_dim": emb_dim,
                            "normalize_embeddings": False,
                        },
                        "model": {
                            "architecture": "cnn",
                            "cnn": {
                                "stages": stages,
                                "blocks_per_stage": blocks,
                                "kernel_size": 3,
                                "stem_kernel_size": 3,
                                "stem_stride": 1,
                                "downsample": "conv",
                                "activation": "gelu",
                                "dropout": 0.1,
                                "classifier_hidden_dim": _infer_classifier_hidden_dim(
                                    state, len(inv_labels)
                                ),
                                "classifier_dropout": 0.2
                                if _infer_classifier_hidden_dim(state, len(inv_labels))
                                == 0
                                else 0.30,
                            },
                            "init": {
                                "embedding_from_centroids": False,
                                "centroid_requires_grad": True,
                                "weight_init": "xavier_uniform",
                            },
                        },
                    }
                    phase2 = build_phase2_cnn_model(
                        inferred_cfg,
                        num_labels=len(inv_labels),
                        primitive_centroids=None,
                    )
                    phase2.load_state_dict(state, strict=False)
                    phase2.to(DEVICE).eval()
                    missing_after = None
            except Exception:
                pass

    rasters = sorted([p for p in args.rasters_dir.glob("*.png")])
    if args.all_classes:
        # Build mapping from input label prefix to list of paths
        per_label: Dict[str, List[Path]] = {}
        for rp in rasters:
            stem = rp.stem
            input_label_prefix = stem.rsplit("_", 1)[0] if "_" in stem else stem
            per_label.setdefault(input_label_prefix, []).append(rp)
        # Sample up to limit per class (randomized if --shuffle else first N)
        selected: List[Path] = []
        rng = random.Random(args.seed)
        for lbl, paths in per_label.items():
            if args.shuffle:
                rng.shuffle(paths)
            take = paths[: args.limit] if args.limit > 0 else paths
            selected.extend(take)
        rasters = selected
    else:
        if args.shuffle:
            random.shuffle(rasters)
        if args.limit > 0:
            rasters = rasters[: args.limit]

    if not rasters:
        print("[warn] No raster PNG files found.", file=sys.stderr)
        return

    print(
        f"[INFO] Starting inference | rasters={len(rasters)} | device={DEVICE.type} "
        f"| phase1_params={sum(p.numel() for p in phase1.parameters())} "
        f"| phase2_params={sum(p.numel() for p in phase2.parameters())}",
        flush=True,
    )

    json_rows: List[Dict[str, Any]] = []

    for idx, raster_path in enumerate(rasters, start=1):
        try:
            grid = raster_to_primitive_grid(raster_path, phase1)
            if args.show_grid and not args.quiet:
                # Print grid as small ASCII matrix (primitive IDs)
                grid_str = "\n".join(
                    " ".join(f"{int(v):04d}" for v in row.tolist())
                    for row in grid.tolist()
                )
                print(f"[GRID] {raster_path.name}\n{grid_str}")

            label_indices, probs = predict_glyph_grid(grid, phase2, topk=args.topk)
            labels = [inv_labels[i] for i in label_indices]
            bases = [base_unicode_map.get(lbl, "?") for lbl in labels]

            if not args.quiet:
                top1_label = labels[0]
                top1_base = bases[0]
                # Derive input label (everything before last underscore) to map its base char (if available)
                stem = raster_path.stem
                input_label = stem.rsplit("_", 1)[0] if "_" in stem else stem
                input_base = base_unicode_map.get(input_label, "?")
                # Determine match status emoji:
                # ✅ if top1 label matches input_label
                # ❗ if input_label appears in remaining top-K
                # ❌ otherwise
                if input_label == top1_label:
                    status_emoji = "✅"
                elif input_label in labels[1 : args.topk]:
                    status_emoji = "❗"
                else:
                    status_emoji = "❌"
                # Concise log format:
                # {input unicode} -> {top match unicode} [top5 unicodes] (filename) {emoji}
                top5_bases = ", ".join(bases[: args.topk])
                print(
                    f"{input_base} -> {top1_base} [{top5_bases}] ({raster_path.name}) {status_emoji}",
                    flush=True,
                )

            json_rows.append(
                {
                    "file": raster_path.name,
                    "topk": [
                        {
                            "rank": r + 1,
                            "label": labels[r],
                            "base": bases[r],
                            "prob": probs[r],
                        }
                        for r in range(len(labels))
                    ],
                }
            )

        except Exception as e:
            print(f"[error] Failed on {raster_path.name}: {e}", file=sys.stderr)

    if args.output_json:
        write_jsonl(args.output_json, json_rows)
        if not args.quiet:
            print(f"[INFO] Wrote JSONL predictions to {args.output_json}", flush=True)

    print("[DONE] Inference complete.", flush=True)


if __name__ == "__main__":
    main()
