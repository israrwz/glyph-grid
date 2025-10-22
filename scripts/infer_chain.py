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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
from torch import nn
import random

# OpenVINO imports (optional)
try:
    from openvino import Core

    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False


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
from models.phase2_cnn_sequence import build_phase2_sequence_model

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
    elif arch == "cnn_sequence":
        return build_phase2_sequence_model(
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
    Build model (from YAML if provided, else embedded checkpoint config, else fallback)
    then load weights. Supports both original capacity config and phase2_light variant.

    Precedence:
      1. --config_yaml (yaml_cfg arg)
      2. Embedded checkpoint payload["config"]
      3. Fallback heuristic (arch_arg or default cnn)
    """
    payload = torch.load(str(ckpt_path), map_location="cpu")
    embedded_cfg = payload.get("config") or {}
    # Merge: yaml overrides embedded (shallow)
    merged_cfg: Dict[str, Any] = {}
    if embedded_cfg and isinstance(embedded_cfg, dict):
        merged_cfg = embedded_cfg.copy()
    if yaml_cfg and isinstance(yaml_cfg, dict):
        # Shallow key override
        for k, v in yaml_cfg.items():
            if (
                isinstance(v, dict)
                and k in merged_cfg
                and isinstance(merged_cfg[k], dict)
            ):
                # Second-level override (model/input/etc.)
                merged_cfg[k] = {**merged_cfg[k], **v}
            else:
                merged_cfg[k] = v

    use_cfg = merged_cfg if merged_cfg else yaml_cfg
    arch = infer_arch_arg_or_yaml(
        arch_arg, use_cfg if isinstance(use_cfg, dict) else {}
    )

    if use_cfg:
        try:
            model = build_phase2_from_yaml(use_cfg, num_labels)
        except Exception as e:
            print(
                f"[warn] Failed building Phase2 from merged config ({e}); using fallback.",
                file=sys.stderr,
            )
            model = build_phase2_fallback(arch, num_labels)
    else:
        model = build_phase2_fallback(arch, num_labels)

    state = payload.get("model_state") or payload
    load_result = model.load_state_dict(state, strict=False)
    if load_result.missing_keys:
        # Attempt adaptive rebuild for CNN width/block mismatches (e.g., light vs capacity)
        if arch == "cnn":
            # Infer embedding/stages from state
            def _infer_embed(state_dict: dict) -> int:
                for k, v in state_dict.items():
                    if k.endswith("embedding.weight") and v.dim() == 2:
                        return v.shape[1]
                return int(model.embedding.weight.shape[1])  # fallback to current

            def _infer_stages(state_dict: dict) -> tuple[list[int], list[int]]:
                stem = None
                for k, v in state_dict.items():
                    if k.endswith("stem.conv.weight") and v.dim() == 4:
                        stem = v.shape[0]
                        break
                block_channels = []
                for k, v in state_dict.items():
                    if (
                        "features" in k
                        and k.endswith("conv1.conv.weight")
                        and v.dim() == 4
                    ):
                        block_channels.append(v.shape[0])
                if stem and block_channels:
                    stages = [stem]
                    counts = [0]
                    for ch in block_channels:
                        if ch != stages[-1]:
                            stages.append(ch)
                            counts.append(1)
                        else:
                            counts[-1] += 1
                    return stages, counts
                # Heuristic fallback
                embed_dim = _infer_embed(state_dict)
                if embed_dim >= 96:
                    return [96, 192, 256], [3, 3, 3]
                return [64, 128, 192], [2, 2, 2]

            try:
                stages, blocks = _infer_stages(state)
                embed_dim = _infer_embed(state)
                hidden_dim = (
                    256  # preserve classifier_hidden_dim for both capacity & light
                )
                inferred_cfg = {
                    "input": {
                        "primitive_vocab_size": 1024,
                        "embedding_dim": embed_dim,
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
                            "dropout": 0.10 if stages[0] == 64 else 0.15,
                            "classifier_hidden_dim": hidden_dim,
                            "classifier_dropout": 0.25 if stages[0] == 64 else 0.30,
                        },
                        "init": {
                            "embedding_from_centroids": False,
                            "centroid_requires_grad": True,
                            "weight_init": "xavier_uniform",
                            "cls_init": "normal",
                        },
                        "classifier": {
                            "hidden_dim": hidden_dim,
                            "dropout": 0.15 if stages[0] == 64 else 0.30,
                            "activation": "gelu",
                        },
                    },
                }
                model = build_phase2_cnn_model(
                    inferred_cfg, num_labels=num_labels, primitive_centroids=None
                )
                model.load_state_dict(state, strict=False)
                print(
                    "[INFO] Rebuilt Phase2 CNN adaptively (light/capacity reconciliation).",
                    flush=True,
                )
            except Exception as e:
                print(f"[warn] Adaptive rebuild failed: {e}", file=sys.stderr)
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
    context_grids: Optional[torch.Tensor] = None,
    context_deltas: Optional[torch.Tensor] = None,
) -> Tuple[List[int], List[float]]:
    """
    grid: (16,16) int64 primitive IDs
    context_grids: Optional (K, 16, 16) for sequence-aware models
    context_deltas: Optional (K,) relative glyph_id deltas
    Returns:
      topk_indices, topk_probs
    """
    if grid.shape != (16, 16):
        raise ValueError(f"Expected grid shape (16,16); got {tuple(grid.shape)}")
    input_tensor = grid.unsqueeze(0).to(torch.long).to(DEVICE)  # (1,16,16)

    with torch.no_grad():
        # Check if model supports sequence context
        if context_grids is not None and context_deltas is not None:
            ctx_tensor = (
                context_grids.unsqueeze(0).to(torch.long).to(DEVICE)
            )  # (1,K,16,16)
            delta_tensor = (
                context_deltas.unsqueeze(0).to(torch.long).to(DEVICE)
            )  # (1,K)
            logits, _ = phase2_model(input_tensor, ctx_tensor, delta_tensor)
        else:
            # Visual-only mode (works for both regular and sequence models)
            output = phase2_model(input_tensor)
            # Handle both tuple and tensor returns
            logits = output[0] if isinstance(output, tuple) else output

        probs = torch.softmax(logits, dim=1)
        values, indices = torch.topk(probs, k=topk, dim=1)

    return indices.squeeze(0).cpu().tolist(), values.squeeze(0).cpu().tolist()


def predict_glyph_grid_batch(
    grids: torch.Tensor,
    phase2_model: nn.Module,
    topk: int = 5,
    context_grids: Optional[torch.Tensor] = None,
    context_deltas: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched version of predict_glyph_grid.

    grids: (B, 16, 16) int64 primitive IDs
    context_grids: Optional (B, K, 16, 16) for sequence-aware models
    context_deltas: Optional (B, K) relative glyph_id deltas
    Returns:
      topk_indices: (B, topk) int
      topk_probs: (B, topk) float
    """
    if grids.ndim != 3 or grids.shape[1:] != (16, 16):
        raise ValueError(f"Expected grid shape (B, 16, 16); got {tuple(grids.shape)}")

    input_tensor = grids.to(torch.long).to(DEVICE)  # (B, 16, 16)

    with torch.no_grad():
        # Check if model supports sequence context
        if context_grids is not None and context_deltas is not None:
            ctx_tensor = context_grids.to(torch.long).to(DEVICE)  # (B,K,16,16)
            delta_tensor = context_deltas.to(torch.long).to(DEVICE)  # (B,K)
            logits, _ = phase2_model(input_tensor, ctx_tensor, delta_tensor)
        else:
            # Visual-only mode
            output = phase2_model(input_tensor)
            logits = output[0] if isinstance(output, tuple) else output

        probs = torch.softmax(logits, dim=1)
        values, indices = torch.topk(probs, k=topk, dim=1)

    return indices.cpu(), values.cpu()


# ---------------------------------------------------------------------------
# OpenVINO Inference
# ---------------------------------------------------------------------------
def export_phase2_to_onnx(
    model: nn.Module,
    onnx_path: Path,
    batch_size: int = 1,
) -> None:
    """Export Phase 2 model to ONNX format."""
    model.eval()
    dummy_input = torch.randint(0, 1024, (batch_size, 16, 16), dtype=torch.long).to(
        DEVICE
    )

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=11,
    )
    print(f"[INFO] Exported Phase 2 model to ONNX: {onnx_path}", flush=True)


def convert_onnx_to_openvino(onnx_path: Path, output_dir: Path) -> Path:
    """Convert ONNX model to OpenVINO IR format."""
    if not OPENVINO_AVAILABLE:
        raise RuntimeError(
            "OpenVINO is not available. Install with: pip install openvino"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    xml_path = output_dir / "phase2_model.xml"

    # Use direct Python API for conversion
    try:
        from openvino import Core

        ie = Core()
        model = ie.read_model(model=str(onnx_path))
        from openvino import serialize

        serialize(model, str(xml_path))
        print(f"[INFO] Converted ONNX to OpenVINO IR: {xml_path}", flush=True)
    except Exception as e:
        raise RuntimeError(f"Failed to convert ONNX to OpenVINO: {e}")

    return xml_path


class OpenVINOInferenceEngine:
    """OpenVINO inference engine for Phase 2 model."""

    def __init__(self, model_path: Path, device: str = "CPU"):
        if not OPENVINO_AVAILABLE:
            raise RuntimeError(
                "OpenVINO is not available. Install with: pip install openvino"
            )

        self.ie = Core()
        self.model = self.ie.read_model(model=str(model_path))
        self.compiled_model = self.ie.compile_model(self.model, device)
        self.infer_request = self.compiled_model.create_infer_request()

        # Get input/output info
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        print(f"[INFO] OpenVINO model loaded on {device}", flush=True)

    def predict_batch(
        self,
        grids: torch.Tensor,
        topk: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched inference using OpenVINO.

        grids: (B, 16, 16) int64 primitive IDs
        Returns:
            topk_indices: (B, topk) int
            topk_probs: (B, topk) float
        """
        if grids.ndim != 3 or grids.shape[1:] != (16, 16):
            raise ValueError(
                f"Expected grid shape (B, 16, 16); got {tuple(grids.shape)}"
            )

        # Convert to numpy and run inference
        input_data = grids.cpu().numpy().astype(np.int64)
        results = self.infer_request.infer({self.input_layer: input_data})
        logits = results[self.output_layer]

        # Compute softmax and top-k
        logits_tensor = torch.from_numpy(logits)
        probs = torch.softmax(logits_tensor, dim=1)
        values, indices = torch.topk(probs, k=topk, dim=1)

        return indices, values


def setup_openvino_model(
    phase2_model: nn.Module,
    cache_dir: Path,
    device: str = "CPU",
) -> OpenVINOInferenceEngine:
    """
    Setup OpenVINO inference engine for Phase 2 model.
    Exports to ONNX, converts to OpenVINO IR, and loads the model.
    """
    if not OPENVINO_AVAILABLE:
        raise RuntimeError(
            "OpenVINO is not available. Install with: pip install openvino openvino-dev"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = cache_dir / "phase2_model.onnx"
    ir_dir = cache_dir / "openvino_ir"

    # Export to ONNX if not cached
    if not onnx_path.exists():
        print("[INFO] Exporting Phase 2 model to ONNX...", flush=True)
        export_phase2_to_onnx(phase2_model, onnx_path, batch_size=1)
    else:
        print(f"[INFO] Using cached ONNX model: {onnx_path}", flush=True)

    # Convert to OpenVINO IR if not cached
    xml_path = ir_dir / "phase2_model.xml"
    if not xml_path.exists():
        print("[INFO] Converting ONNX to OpenVINO IR...", flush=True)
        xml_path = convert_onnx_to_openvino(onnx_path, ir_dir)
    else:
        print(f"[INFO] Using cached OpenVINO IR: {xml_path}", flush=True)

    # Load OpenVINO model
    return OpenVINOInferenceEngine(xml_path, device=device)


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
    ap.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for Phase 2 inference (default: 32).",
    )
    ap.add_argument(
        "--phase1_batch_size",
        type=int,
        default=256,
        help="Batch size for Phase 1 cell inference (default: 256 = all cells at once).",
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
    ap.add_argument(
        "--use-test-split",
        action="store_true",
        help="Restrict inference to glyph_ids listed in phase2_test_ids.txt (derived from label_map parent / splits or --splits-dir).",
    )
    ap.add_argument(
        "--splits-dir",
        type=Path,
        default=None,
        help="Optional explicit splits directory containing phase2_test_ids.txt (defaults to <label_map_dir>/splits).",
    )
    ap.add_argument(
        "--use-memmap-grids",
        action="store_true",
        help="Use memmap grids_uint16.npy + glyph_row_ids.npy directly (bypass Phase 1 cell inference). Applies test split filtering if --use-test-split is set.",
    )
    ap.add_argument(
        "--memmap-grids-file",
        type=Path,
        default=Path("data/grids_memmap/grids_uint16.npy"),
        help="Path to memmap grids file when --use-memmap-grids is enabled.",
    )
    ap.add_argument(
        "--memmap-row-ids-file",
        type=Path,
        default=Path("data/grids_memmap/glyph_row_ids.npy"),
        help="Path to memmap glyph_row_ids.npy when --use-memmap-grids is enabled.",
    )
    ap.add_argument(
        "--glyph-labels-file",
        type=Path,
        default=Path("data/grids_memmap/glyph_labels.jsonl"),
        help="Optional glyph_labels.jsonl mapping (glyph_id -> label string); if absent will attempt to derive from label_map keys.",
    )
    ap.add_argument(
        "--summary",
        action="store_true",
        help="Print summary metrics (top1 accuracy, top5 contains rate) after inference (raster mode only).",
    )
    ap.add_argument(
        "--use-openvino",
        action="store_true",
        help="Use OpenVINO for Phase 2 inference acceleration (requires openvino package).",
    )
    ap.add_argument(
        "--openvino-device",
        type=str,
        default="CPU",
        choices=["CPU", "GPU", "AUTO"],
        help="OpenVINO device target (default: CPU).",
    )
    ap.add_argument(
        "--openvino-cache-dir",
        type=Path,
        default=Path("cache/openvino"),
        help="Directory for OpenVINO model cache (default: cache/openvino).",
    )
    ap.add_argument(
        "--use-sequence-context",
        action="store_true",
        help="Enable sequence context for sequence-aware models (loads neighbor glyphs).",
    )
    ap.add_argument(
        "--context-window",
        type=int,
        default=2,
        help="Context window size for sequence-aware models (default: 2, meaning 2 before + 2 after).",
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
                    "classifier_hidden_dim": 256,
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

    # ------------------------------------------------------------------
    # OpenVINO Setup (if requested)
    # ------------------------------------------------------------------
    openvino_engine = None
    if args.use_openvino:
        if not OPENVINO_AVAILABLE:
            print(
                "[error] OpenVINO requested but not available. Install with: pip install openvino openvino-dev",
                file=sys.stderr,
            )
            sys.exit(1)
        if phase2 is None:
            print(
                "[error] Phase 2 model not loaded; cannot setup OpenVINO.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(
            f"[INFO] Setting up OpenVINO inference engine on {args.openvino_device}...",
            flush=True,
        )
        try:
            openvino_engine = setup_openvino_model(
                phase2,
                cache_dir=args.openvino_cache_dir,
                device=args.openvino_device,
            )
            print("[INFO] OpenVINO engine ready.", flush=True)
        except Exception as e:
            print(f"[error] Failed to setup OpenVINO: {e}", file=sys.stderr)
            sys.exit(1)

    # ------------------------------------------------------------------
    # Test split loading (glyph_id list) if requested
    # ------------------------------------------------------------------
    test_ids_set: Optional[set[int]] = None
    if args.use_test_split:
        splits_dir = args.splits_dir or (args.label_map.parent / "splits")
        test_file = splits_dir / "phase2_test_ids.txt"
        if not test_file.exists():
            print(f"[error] Test split file not found: {test_file}", file=sys.stderr)
            return
        try:
            test_ids = [
                int(line.strip())
                for line in test_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            test_ids_set = set(test_ids)
            print(f"[INFO] Loaded test split IDs: {len(test_ids_set)}", flush=True)
        except Exception as e:
            print(f"[error] Failed reading test split file: {e}", file=sys.stderr)
            return

    # ------------------------------------------------------------------
    # MEMMAP GRID MODE
    # ------------------------------------------------------------------
    # Enforce raster-only when user requests test split evaluation with Phase 1 reconstruction
    # (Ignore memmap mode if --use-test-split is set to honor user preference for raster inference.)
    if args.use_test_split and args.use_memmap_grids:
        print(
            "[INFO] Ignoring --use-memmap-grids because --use-test-split requested (raster + Phase 1 inference).",
            flush=True,
        )
        memmap_mode = False
    else:
        memmap_mode = args.use_memmap_grids
    memmap_grids: Optional[Any] = None
    memmap_row_ids: Optional[Any] = None
    glyph_id_to_label: Dict[int, str] = {}

    if memmap_mode:
        import numpy as np

        if not args.memmap_grids_file.exists() or not args.memmap_row_ids_file.exists():
            print(
                "[error] Memmap files missing; disable --use-memmap-grids or provide correct paths.",
                file=sys.stderr,
            )
            return
        try:
            row_ids_arr = np.load(args.memmap_row_ids_file)
            mm = np.memmap(args.memmap_grids_file, dtype=np.uint16, mode="r")
            if mm.size % 256 != 0:
                raise RuntimeError("Memmap size not divisible by 256.")
            total = mm.size // 256
            mm = mm.reshape(total, 16, 16)
            if len(row_ids_arr) != total:
                raise RuntimeError("glyph_row_ids length mismatch memmap grid count.")
            memmap_grids = mm
            memmap_row_ids = row_ids_arr
        except Exception as e:
            print(f"[error] Failed loading memmap grids: {e}", file=sys.stderr)
            return
        # Optional glyph_labels_file mapping (glyph_id -> label string)
        if args.glyph_labels_file.exists():
            try:
                for line in args.glyph_labels_file.read_text(
                    encoding="utf-8"
                ).splitlines():
                    if not line.strip():
                        continue
                    import json

                    rec = json.loads(line)
                    gid = int(rec.get("glyph_id"))
                    lbl = (
                        rec.get("label")
                        or rec.get("glyph_label")
                        or rec.get("label_string")
                    )
                    if lbl is not None:
                        glyph_id_to_label[gid] = str(lbl)
            except Exception:
                pass
        # Fallback: derive label string from label_map keys pattern "<glyphid>_<rest>"
        if not glyph_id_to_label:
            for k in label_map.keys():
                try:
                    # if key starts with integer id followed by underscore
                    parts = k.split("_", 1)
                    if parts and parts[0].isdigit():
                        gid = int(parts[0])
                        glyph_id_to_label.setdefault(gid, k)
                except Exception:
                    continue

        # Build list of glyph indices respecting test split
        chosen_positions: List[int] = []
        for pos, gid in enumerate(memmap_row_ids.tolist()):
            if test_ids_set and gid not in test_ids_set:
                continue
            chosen_positions.append(pos)

        if not chosen_positions:
            print(
                "[warn] No glyph positions selected (check test split filtering).",
                file=sys.stderr,
            )

        # If --limit applies as samples per class, we need grouping by label base
        if args.all_classes:
            # Group by label string
            per_label_positions: Dict[str, List[int]] = {}
            for pos in chosen_positions:
                gid = int(memmap_row_ids[pos])
                lbl = glyph_id_to_label.get(gid, f"{gid}")
                per_label_positions.setdefault(lbl, []).append(pos)
            rng = random.Random(args.seed)
            selected_positions: List[int] = []
            for lbl, plist in per_label_positions.items():
                if args.shuffle:
                    rng.shuffle(plist)
                take = plist[: args.limit] if args.limit > 0 else plist
                selected_positions.extend(take)
            chosen_positions = selected_positions
        else:
            if args.shuffle:
                rng = random.Random(args.seed)
                rng.shuffle(chosen_positions)
            if args.limit > 0:
                chosen_positions = chosen_positions[: args.limit]

        if not chosen_positions:
            print("[warn] No glyphs after sampling; aborting.", file=sys.stderr)
            return

        print(
            f"[INFO] Memmap inference | selected_grids={len(chosen_positions)}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # RASTER MODE
    # ------------------------------------------------------------------
    rasters: List[Path] = []
    if not memmap_mode:
        rasters = sorted([p for p in args.rasters_dir.glob("*.png")])
        # Test split filtering (extract trailing numeric id)
        if test_ids_set:
            filtered = []
            for rp in rasters:
                stem = rp.stem
                # Expect last underscore component to be numeric glyph_id
                parts = stem.rsplit("_", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    gid = int(parts[1])
                    if gid in test_ids_set:
                        filtered.append(rp)
            rasters = filtered
        if args.all_classes:
            per_label: Dict[str, List[Path]] = {}
            for rp in rasters:
                stem = rp.stem
                input_label_prefix = stem.rsplit("_", 1)[0] if "_" in stem else stem
                per_label.setdefault(input_label_prefix, []).append(rp)
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

    # Start inference timer (after models loaded and ready)
    inference_start_time = time.time()

    json_rows: List[Dict[str, Any]] = []

    if memmap_mode:
        # Direct grid inference loop
        for pos in chosen_positions:
            gid = int(memmap_row_ids[pos])
            raw_grid = memmap_grids[pos]  # numpy (16,16)
            grid = torch.from_numpy(raw_grid.astype("int64"))
            # Load context for sequence-aware models
            ctx_grids = None
            ctx_deltas = None
            if args.use_sequence_context and hasattr(phase2, "context_window"):
                print(
                    f"[DEBUG] Loading sequence context for glyph_id={gid}, window={args.context_window}",
                    flush=True,
                )
                # Find neighbors
                window = args.context_window
                neighbors = []
                deltas = []
                for offset in range(-window, 0):
                    neighbor_gid = gid + offset
                    neighbor_pos = None
                    for p, rid in enumerate(memmap_row_ids):
                        if rid == neighbor_gid:
                            neighbor_pos = p
                            break
                    if neighbor_pos is not None:
                        neighbors.append(
                            torch.from_numpy(memmap_grids[neighbor_pos].astype("int64"))
                        )
                        deltas.append(offset)
                    else:
                        neighbors.append(torch.zeros((16, 16), dtype=torch.int64))
                        deltas.append(offset)
                for offset in range(1, window + 1):
                    neighbor_gid = gid + offset
                    neighbor_pos = None
                    for p, rid in enumerate(memmap_row_ids):
                        if rid == neighbor_gid:
                            neighbor_pos = p
                            break
                    if neighbor_pos is not None:
                        neighbors.append(
                            torch.from_numpy(memmap_grids[neighbor_pos].astype("int64"))
                        )
                        deltas.append(offset)
                    else:
                        neighbors.append(torch.zeros((16, 16), dtype=torch.int64))
                        deltas.append(offset)
                if neighbors:
                    ctx_grids = torch.stack(neighbors, dim=0)  # (K, 16, 16)
                    ctx_deltas = torch.tensor(deltas, dtype=torch.long)  # (K,)
                    print(
                        f"[DEBUG] Context loaded: {len(neighbors)} neighbors, deltas={deltas}",
                        flush=True,
                    )
                else:
                    print(f"[DEBUG] No neighbors found for glyph_id={gid}", flush=True)
            else:
                if not args.use_sequence_context:
                    print(
                        f"[DEBUG] Sequence context disabled (--use-sequence-context not set)",
                        flush=True,
                    )
                elif not hasattr(phase2, "context_window"):
                    print(
                        f"[DEBUG] Model does not have context_window attribute (not a sequence model)",
                        flush=True,
                    )

            with torch.no_grad():
                print(
                    f"[DEBUG] Calling predict_glyph_grid with context={'YES' if ctx_grids is not None else 'NO'}",
                    flush=True,
                )
                label_indices, probs = predict_glyph_grid(
                    grid,
                    phase2,
                    topk=args.topk,
                    context_grids=ctx_grids,
                    context_deltas=ctx_deltas,
                )
                print(
                    f"[DEBUG] Prediction: top1_idx={label_indices[0]}, prob={probs[0]:.4f}",
                    flush=True,
                )
            labels = [inv_labels[i] for i in label_indices]
            bases = [base_unicode_map.get(lbl, "?") for lbl in labels]
            top1_label = labels[0]
            top1_base = bases[0]
            input_label = glyph_id_to_label.get(gid, str(gid))
            # Input base unicode (derive similarly to raster mode)
            input_base = base_unicode_map.get(input_label, "?")
            if input_label == top1_label:
                status_emoji = "✅"
            elif input_label in labels[1 : args.topk]:
                status_emoji = "❗"
            else:
                status_emoji = "❌"
            top5_bases = ", ".join(bases[: args.topk])
            print(
                f"{input_base} -> {top1_base} [{top5_bases}] (glyph_{gid}.memmap) {status_emoji}",
                flush=True,
            )
            json_rows.append(
                {
                    "glyph_id": gid,
                    "file": f"glyph_{gid}.memmap",
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
    else:
        # Batched inference mode
        batch_size = args.batch_size
        num_rasters = len(rasters)

        for batch_start in range(0, num_rasters, batch_size):
            batch_end = min(batch_start + batch_size, num_rasters)
            batch_rasters = rasters[batch_start:batch_end]

            # Phase 1: Convert rasters to grids
            batch_grids = []
            valid_indices = []
            for i, raster_path in enumerate(batch_rasters):
                try:
                    grid = raster_to_primitive_grid(raster_path, phase1)
                    batch_grids.append(grid)
                    valid_indices.append(i)
                except Exception as e:
                    print(
                        f"[error] Phase 1 failed for {raster_path.name}: {e}",
                        file=sys.stderr,
                    )
                    continue

            if not batch_grids:
                continue

            # Phase 2: Batch prediction
            try:
                grids_tensor = torch.stack(batch_grids)  # (B, 16, 16)

                # Use OpenVINO or PyTorch
                if args.use_openvino and openvino_engine:
                    batch_indices, batch_probs = openvino_engine.predict_batch(
                        grids_tensor, topk=args.topk
                    )
                else:
                    batch_indices, batch_probs = predict_glyph_grid_batch(
                        grids_tensor, phase2, topk=args.topk
                    )

                # Process results
                for i, (grid_idx, raster_path) in enumerate(
                    zip(valid_indices, [batch_rasters[vi] for vi in valid_indices])
                ):
                    label_indices = batch_indices[i].tolist()
                    probs = batch_probs[i].tolist()
                    labels = [inv_labels[idx] for idx in label_indices]
                    bases = [base_unicode_map.get(lbl, "?") for lbl in labels]

                    if args.show_grid and not args.quiet:
                        grid = batch_grids[i]
                        grid_str = "\n".join(
                            " ".join(f"{int(v):04d}" for v in row.tolist())
                            for row in grid.tolist()
                        )
                        print(f"[GRID] {raster_path.name}\n{grid_str}")

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
                print(
                    f"[error] Batch inference failed: {e}",
                    file=sys.stderr,
                )
                continue

    # End inference timer
    inference_end_time = time.time()
    inference_duration = inference_end_time - inference_start_time

    if not args.quiet:
        num_glyphs = len(json_rows)
        avg_time_per_glyph = (
            (inference_duration / num_glyphs * 1000) if num_glyphs > 0 else 0
        )
        print(
            f"[TIMING] Inference time: {inference_duration:.3f}s for {num_glyphs} glyphs ({avg_time_per_glyph:.2f}ms per glyph)",
            flush=True,
        )

    if args.output_json:
        write_jsonl(args.output_json, json_rows)
        if not args.quiet:
            print(f"[INFO] Wrote JSONL predictions to {args.output_json}", flush=True)

    # Summary metrics (raster mode only)
    if not memmap_mode and args.summary and json_rows:
        total = len(json_rows)
        top1_matches = 0
        top5_contains = 0
        for rec in json_rows:
            # Reconstruct input label prefix from filename
            fname = rec["file"]
            stem = Path(fname).stem
            input_label = stem.rsplit("_", 1)[0] if "_" in stem else stem
            topk_labels = [entry["label"] for entry in rec["topk"]]
            if topk_labels and input_label == topk_labels[0]:
                top1_matches += 1
                top5_contains += 1  # top1 is also in top5
            elif input_label in topk_labels[1:]:
                top5_contains += 1
        top1_acc = top1_matches / total if total else 0.0
        top5_contain_rate = top5_contains / total if total else 0.0
        print(
            f"[SUMMARY] samples={total} top1_acc={top1_acc:.4f} top5_contains_rate={top5_contain_rate:.4f} "
            f"top1_matches={top1_matches} top5_contains={top5_contains}",
            flush=True,
        )
    print("[DONE] Inference complete.", flush=True)


if __name__ == "__main__":
    main()
