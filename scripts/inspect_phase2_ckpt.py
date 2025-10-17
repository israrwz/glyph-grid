#!/usr/bin/env python3
"""
inspect_phase2_ckpt.py
======================

Purpose:
  Inspect a Phase 2 glyph classification checkpoint to infer the model
  architecture (CNN vs Transformer) and structural hyperparameters
  without needing the original YAML config. This helps reconcile
  inference-time mismatches (e.g., missing keys in infer_chain.py).

What it does:
  1. Loads a PyTorch checkpoint (Path to .pt).
  2. Extracts the underlying state_dict (handles payload["model_state"]).
  3. Heuristically determines whether it's a CNN or Transformer:
     - Transformer indicators: keys containing "encoder.0", "primitive_embedding",
       "cls_token", "positional".
     - CNN indicators: keys like "stem.conv.weight", "features.0.conv1.conv.weight".
  4. Derives key structural attributes:
     - Common: vocab_size, embedding_dim, num_labels.
     - CNN: stages (channel widths), blocks_per_stage, classifier head size.
     - Transformer: d_model, mlp_hidden_dim, num_layers, inferred heads (best-effort),
       classifier hidden dim (if present).
  5. Prints a human-readable summary and optionally writes JSON output.

Usage:
  python scripts/inspect_phase2_ckpt.py --ckpt checkpoints/phase2/best.pt
  python scripts/inspect_phase2_ckpt.py --ckpt some.pt --json_out structure.json
  python scripts/inspect_phase2_ckpt.py --ckpt some.pt --verbose

Return codes:
  0 on success, non-zero on fatal errors (e.g., file missing / torch absent).

Limitations:
  - If state_dict naming changed or is highly pruned, some fields may be 'unknown'.
  - Head count for a transformer is inferred heuristically (may fail).
  - Does not validate tensor contents; only shapes & key patterns.

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# ---------------------------------------------------------------------------
# Safe torch import
# ---------------------------------------------------------------------------
try:
    import torch
except ImportError as e:  # pragma: no cover
    print("[error] PyTorch not installed; cannot inspect checkpoint.", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_state(ckpt_path: Path):
    payload = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(payload, dict) and "model_state" in payload:
        return payload["model_state"], payload
    # If this looks like a plain state_dict
    return payload, {"_raw": True}


def classify_arch(keys: List[str]) -> str:
    """Return 'transformer', 'cnn', or 'unknown'.

    Enhanced:
      - Handles optional leading prefixes: 'model.', 'module.', 'net.'.
      - More robust CNN detection via presence of any 'features.' residual pattern.
      - More robust transformer detection via any 'encoder.' or multi-head attn pattern.
    """
    # Strip common wrappers
    normalized = []
    for k in keys:
        nk = k
        for prefix in ("model.", "module.", "net."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        normalized.append(nk)

    # Transformer indicators
    if any(
        ("encoder." in k)
        or ("cls_token" in k)
        or ("primitive_embedding.weight" in k)
        or ("positional" in k)
        or ("multihead" in k)
        for k in normalized
    ):
        return "transformer"

    # CNN indicators (stem + features residual blocks)
    if any(
        ("stem.conv" in k)
        or ("features." in k and ".conv1.conv.weight" in k)
        or ("features." in k and ".conv.weight" in k and ".conv1." not in k)
        for k in normalized
    ):
        return "cnn"

    return "unknown"


def extract_common(state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    # Normalize keys (remove common prefixes)
    remapped: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        nk = k
        for prefix in ("model.", "module.", "net."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        remapped[nk] = v

    # Candidate embedding keys (CNN / Transformer)
    emb_key = None
    for candidate in (
        "embedding.weight",
        "primitive_embedding.weight",
        "embed.embedding.weight",
    ):
        if candidate in remapped:
            emb_key = candidate
            break

    if emb_key:
        w = remapped[emb_key]
        out["vocab_size"] = w.shape[0]
        out["embedding_dim"] = w.shape[1] if w.dim() == 2 else "unknown"
    else:
        # Fallback heuristic: find any weight with shape[0] in typical vocab range (900..1300)
        vocab_like = [
            (k, t)
            for k, t in remapped.items()
            if t.dim() == 2 and 900 <= t.shape[0] <= 1300
        ]
        if vocab_like:
            # Choose tensor with smallest second dim as embedding (usual embedding_dim < hidden_dim)
            vocab_like.sort(key=lambda x: x[1].shape[1])
            k_sel, t_sel = vocab_like[0]
            out["vocab_size"] = t_sel.shape[0]
            out["embedding_dim"] = t_sel.shape[1]
            out["embedding_inferred_from"] = k_sel
        else:
            out["vocab_size"] = "unknown"
            out["embedding_dim"] = "unknown"

    # Try to infer number of labels from classifier last linear layer
    classifier_candidates = [
        k for k in state.keys() if k.endswith(".weight") and "classifier" in k
    ]
    # Heuristic: final classifier weight usually has shape (num_labels, hidden)
    num_labels = None
    for ck in sorted(classifier_candidates):
        tensor = state[ck]
        if tensor.dim() == 2:
            rows, cols = tensor.shape
            # If rows is large (e.g., ~1000) treat that as num_labels
            if rows >= 50:  # threshold
                num_labels = rows
    out["num_labels"] = num_labels if num_labels is not None else "unknown"
    return out


# ----------------------------- CNN Extraction -------------------------------
def extract_cnn_details(state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Attempt to reconstruct stages (channel widths) & blocks per stage.

    Strategy:
      - Stem out channels from stem.conv.weight shape[0].
      - Residual blocks have keys like features.X.conv1.conv.weight (out_ch, in_ch,...)
        where out_ch == in_ch for block conv1/conv2; downsample transitions often appear
        as features.<i>.conv.weight with different in/out.
      - We partition sequential features.* modules into stages when an in/out channel
        size increases (after a downsample).
    """
    info: Dict[str, Any] = {
        "stem_out_channels": "unknown",
        "stages": [],
        "blocks_per_stage": [],
        "classifier_hidden_dim": 0,
    }

    stem_key = "stem.conv.weight"
    if stem_key in state:
        stem_w = state[stem_key]
        if stem_w.dim() == 4:
            info["stem_out_channels"] = int(stem_w.shape[0])

    # Gather feature conv1 weights
    feature_block_entries: List[Tuple[int, int, int]] = []  # (index, out_ch, in_ch)
    downsample_entries: List[Tuple[int, int, int]] = []  # (index, out_ch, in_ch)

    for k in state.keys():
        # Residual block conv1: features.<i>.conv1.conv.weight
        if k.startswith("features.") and k.endswith("conv1.conv.weight"):
            try:
                idx = int(k.split(".")[1])
                w = state[k]
                if w.dim() == 4:
                    out_ch, in_ch = int(w.shape[0]), int(w.shape[1])
                    feature_block_entries.append((idx, out_ch, in_ch))
            except Exception:
                pass
        # Downsample (stage transition) conv: features.<i>.conv.weight
        elif (
            k.startswith("features.")
            and k.endswith("conv.weight")
            and ".conv1." not in k
        ):
            try:
                idx = int(k.split(".")[1])
                w = state[k]
                if w.dim() == 4:
                    out_ch, in_ch = int(w.shape[0]), int(w.shape[1])
                    downsample_entries.append((idx, out_ch, in_ch))
            except Exception:
                pass

    # Sort by index
    feature_block_entries.sort(key=lambda x: x[0])
    downsample_entries.sort(key=lambda x: x[0])

    # Build stage boundaries
    # Approach: start with stem_out, then when a downsample entry appears we
    # treat that as the start of a new stage with out_ch channels.
    stages: List[int] = []
    blocks_per_stage: List[int] = []

    if info["stem_out_channels"] != "unknown":
        stages.append(info["stem_out_channels"])
        blocks_per_stage.append(0)

    # Map block index -> (out_ch, in_ch)
    for idx, out_ch, in_ch in feature_block_entries:
        # If we have a downsample at or before this idx with out_ch different from last stage
        # check if this block belongs to a new stage
        # simpler heuristic: if out_ch != current stage channel width, begin new stage.
        current_stage_ch = stages[-1] if stages else None
        if current_stage_ch is None or out_ch != current_stage_ch:
            stages.append(out_ch)
            blocks_per_stage.append(1)
        else:
            blocks_per_stage[-1] += 1

    # If classifier hidden layer exists
    # Seek classifier.1.weight or classifier.0.weight shapes
    classifier_weight_keys = [
        k for k in state if k.startswith("classifier.") and k.endswith(".weight")
    ]
    # If we find >1 weight before final classification layer, deduce hidden dim
    # The final layer is likely the one whose out_features == num_labels (already in common).
    common = extract_common(state)
    num_labels = common.get("num_labels")
    hidden_dim_detected = None
    for ck in classifier_weight_keys:
        w = state[ck]
        if w.dim() == 2:
            out_f, in_f = w.shape
            if num_labels != "unknown" and out_f == num_labels:
                continue  # likely final classifier
            # Otherwise treat out_f as hidden dim candidate
            if out_f != num_labels:
                hidden_dim_detected = out_f
    if hidden_dim_detected is not None:
        info["classifier_hidden_dim"] = hidden_dim_detected

    if stages:
        info["stages"] = stages
        info["blocks_per_stage"] = blocks_per_stage

    return info


# -------------------------- Transformer Extraction -------------------------
def extract_transformer_details(state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Extract transformer structural info.

    Heuristics:
      - d_model: from a feedforward input or output projection weight shape.
      - mlp_hidden_dim: largest linear weight in encoder layer where dim0 > dim1.
      - num_layers: count distinct encoder.<layer_idx>. prefixes.
      - heads: try to infer from multi-head projection weight shapes if combined qkv present.
    """
    info: Dict[str, Any] = {
        "d_model": "unknown",
        "mlp_hidden_dim": "unknown",
        "num_layers": 0,
        "classifier_hidden_dim": "unknown",
        "num_heads_inferred": "unknown",
    }

    # Count layers
    encoder_prefixes = set()
    for k in state.keys():
        if k.startswith("encoder."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                encoder_prefixes.add(int(parts[1]))
    if encoder_prefixes:
        info["num_layers"] = len(encoder_prefixes)

    # Feedforward linear weights pattern guess
    # Look for something like encoder.0.ffn.linear1.weight or encoder.0.mlp.fc1.weight
    # Fallback: pick any encoder.0.*weight with shape[0] > shape[1]
    candidate_ff = []
    for k, v in state.items():
        if not k.startswith("encoder.0") or not k.endswith("weight"):
            continue
        if v.dim() == 2:
            out_f, in_f = v.shape
            candidate_ff.append((k, out_f, in_f))
    # Choose the one with out_f > in_f as mlp hidden, in_f as d_model
    for k, out_f, in_f in candidate_ff:
        if out_f > in_f:
            info["d_model"] = in_f
            info["mlp_hidden_dim"] = out_f
            break
    # If still unknown, fallback to largest in_f encountered
    if info["d_model"] == "unknown":
        if candidate_ff:
            # pick the last tuple
            _, out_f, in_f = candidate_ff[-1]
            info["d_model"] = in_f
            info["mlp_hidden_dim"] = out_f

    # Infer classifier hidden dim: look for classifier.*.weight where out_f != num_labels
    common = extract_common(state)
    num_labels = common.get("num_labels")
    classifier_weight_keys = [
        k for k in state if k.startswith("classifier.") and k.endswith(".weight")
    ]
    hidden = None
    for ck in classifier_weight_keys:
        w = state[ck]
        if w.dim() == 2:
            out_f, in_f = w.shape
            if num_labels != "unknown" and out_f == num_labels:
                continue
            hidden = out_f
    if hidden is not None:
        info["classifier_hidden_dim"] = hidden

    # Multi-head attention inference attempt:
    # Search for a weight where dimension0 == d_model and dimension1 == d_model (linear proj),
    # and maybe a fused qkv with dimension0 == 3*d_model.
    if info["d_model"] != "unknown":
        d_model = info["d_model"]
        qkv_like = [
            v
            for k, v in state.items()
            if v.dim() == 2 and v.shape[0] == 3 * d_model and v.shape[1] == d_model
        ]
        if qkv_like:
            # Without direct head dimension, attempt factoring d_model
            # Try common head divisors
            for h in (16, 12, 8, 6, 4, 2):
                if d_model % h == 0:
                    info["num_heads_inferred"] = h
                    break

    return info


# ---------------------------------------------------------------------------
# Main inspection routine
# ---------------------------------------------------------------------------
def inspect(ckpt_path: Path, verbose: bool = False) -> Dict[str, Any]:
    state, raw_payload = load_state(ckpt_path)
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a dict-like state.")
    keys = list(state.keys())
    arch = classify_arch(keys)
    # Rebuild remapped dict for downstream detail extraction (prefix stripping)
    cleaned_state: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        nk = k
        for prefix in ("model.", "module.", "net."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        cleaned_state[nk] = v
    # Use cleaned_state for detail extraction to increase robustness

    common = extract_common(state)
    details: Dict[str, Any] = {}
    if arch == "cnn":
        details = extract_cnn_details(cleaned_state)
    elif arch == "transformer":
        details = extract_transformer_details(cleaned_state)

    result = {
        "checkpoint": str(ckpt_path),
        "architecture": arch,
        "common": common,
        "details": details,
        "total_param_tensors": len(keys),
    }

    if verbose:
        print("=== Phase 2 Checkpoint Inspection ===")
        print(f"File: {ckpt_path}")
        print(f"Architecture: {arch}")
        print(f"Vocab Size: {common.get('vocab_size')}")
        print(f"Embedding Dim: {common.get('embedding_dim')}")
        print(f"Num Labels: {common.get('num_labels')}")
        if arch == "cnn":
            print("-- CNN Details --")
            print(f"Stem Out Channels: {details.get('stem_out_channels')}")
            print(f"Stages (channels): {details.get('stages')}")
            print(f"Blocks/Stage: {details.get('blocks_per_stage')}")
            print(f"Classifier Hidden Dim: {details.get('classifier_hidden_dim')}")
        elif arch == "transformer":
            print("-- Transformer Details --")
            print(f"d_model: {details.get('d_model')}")
            print(f"mlp_hidden_dim: {details.get('mlp_hidden_dim')}")
            print(f"num_layers: {details.get('num_layers')}")
            print(f"classifier_hidden_dim: {details.get('classifier_hidden_dim')}")
            print(f"num_heads_inferred: {details.get('num_heads_inferred')}")
        else:
            print("No further structural details (unknown architecture).")
        if verbose:
            print(f"Total parameter tensors: {len(keys)}")
        if verbose and arch == "unknown":
            print(
                "Hint: Missing expected key patterns; maybe checkpoint is incomplete or saved with a custom wrapper."
            )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Inspect Phase 2 checkpoint structure.")
    ap.add_argument(
        "--ckpt", type=Path, required=True, help="Path to Phase 2 checkpoint (.pt)"
    )
    ap.add_argument(
        "--json_out",
        type=Path,
        default=None,
        help="Optional path to write JSON summary.",
    )
    ap.add_argument(
        "--verbose", action="store_true", help="Print human-readable summary."
    )
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.ckpt.exists():
        print(f"[error] Checkpoint not found: {args.ckpt}", file=sys.stderr)
        sys.exit(1)

    try:
        result = inspect(args.ckpt, verbose=args.verbose)
    except Exception as e:
        print(f"[error] Failed to inspect checkpoint: {e}", file=sys.stderr)
        sys.exit(3)

    if args.json_out:
        try:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            with args.json_out.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            if args.verbose:
                print(f"[info] JSON summary written to {args.json_out}")
        except Exception as e:
            print(f"[warn] Could not write JSON output: {e}", file=sys.stderr)

    if not args.verbose and not args.json_out:
        # Minimal default output
        print(json.dumps(result, ensure_ascii=False))

    sys.exit(0)


if __name__ == "__main__":
    main()
