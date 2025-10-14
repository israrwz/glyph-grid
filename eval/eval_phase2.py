#!/usr/bin/env python3
"""
Phase 2 Evaluation Script
=========================

Evaluates a trained Phase 2 transformer model on a specified split and produces:
  - Overall metrics (loss, top1, top5, macro-F1, diacritic subset accuracy if available)
  - Confusion matrix (saved as .npz + optional rendered image)
  - Per-class accuracy JSON
  - Top-N most confused class pairs
  - Optional per-class support summary
  - Optional CSV dump of per-class accuracies

Data / Artifacts Expected
------------------------
From prior pipeline steps:
  data/processed/
    grids/<glyph_id>.u16        # 16x16 primitive ID grid (uint16 raw) OR <glyph_id>.npy
    label_map.json              # { label_string: class_index, ... }
    splits/phase2_{train,val,test}_ids.txt
  data/rasters/metadata.jsonl   # (optional) contains glyph_id, label, is_diacritic
  checkpoints/phase2/*.pt       # phase 2 model checkpoints
  assets/centroids/primitive_centroids.npy (optional, for embedding init if required)

Usage
-----
  python -m eval.eval_phase2 \
      --config configs/phase2.yaml \
      --checkpoint checkpoints/phase2/epoch09-val0.8123.pt \
      --split val \
      --out-dir output/phase2_eval_val

  # Auto-pick latest checkpoint in directory:
  python -m eval.eval_phase2 --config configs/phase2.yaml --checkpoint-dir checkpoints/phase2 --split test

Key Outputs (out-dir)
---------------------
  metrics.json
  per_class_accuracy.json
  confusion_matrix.npz          # confusion (C,C) int64
  confusion_matrix.png          # (if matplotlib available and --render-matrix)
  confusion_top_pairs.json      # top-N confused class pairs
  per_class_accuracy.csv        # (if --csv)
  summary.txt                   # human-readable high-level summary

Confusion Pair Ranking
----------------------
A "confused pair" (i,j) is ranked by min(conf[i,j], conf[j,i]) * (conf[i,j]+conf[j,i]) for stability.
This highlights mutual confusions over one-sided dominance.

Performance Notes
-----------------
Memory: confusion matrix is C^2; for 2k classes it's ~32MB (int64). Acceptable here.

Future Extensions
-----------------
- Support soft mixture embeddings
- Support class weighting / filtering
- Export attention attribution summaries

License: Project root license.

"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
except ImportError as e:
    raise RuntimeError("PyTorch required (pip install torch).") from e

try:
    import yaml
except ImportError as e:
    raise RuntimeError("pyyaml required (pip install pyyaml).") from e

# Optional matplotlib for confusion matrix rendering
try:
    import matplotlib.pyplot as plt  # type: ignore

    _MPL = True
except Exception:  # pragma: no cover
    _MPL = False

# Import model factory
try:
    from models.phase2_transformer import build_phase2_model
except ImportError as e:
    raise RuntimeError(
        "Failed to import phase2 model factory. Ensure PYTHONPATH includes project root."
    ) from e


# ---------------------------------------------------------------------------
# Configuration & CLI
# ---------------------------------------------------------------------------


@dataclass
class Args:
    config: Path
    checkpoint: Optional[Path]
    checkpoint_dir: Optional[Path]
    split: str
    out_dir: Path
    batch_size: int
    num_workers: int
    pin_memory: bool
    render_matrix: bool
    top_confused: int
    csv: bool
    limit: Optional[int]
    device: str
    centroid_file: Optional[Path]


def parse_args(argv: Optional[Sequence[str]] = None) -> Args:
    p = argparse.ArgumentParser(description="Evaluate Phase 2 transformer.")
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Phase 2 YAML config (to reconstruct architecture).",
    )
    ckpt_group = p.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument(
        "--checkpoint",
        type=Path,
        help="Specific checkpoint file (.pt).",
    )
    ckpt_group.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Directory containing checkpoints (latest mtime picked).",
    )
    p.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Which split file to evaluate.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/phase2_eval"),
        help="Output directory for evaluation artifacts.",
    )
    p.add_argument("--batch-size", type=int, default=256, help="Evaluation batch size.")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    p.add_argument(
        "--pin-memory", action="store_true", help="Enable pin_memory in DataLoader."
    )
    p.add_argument(
        "--render-matrix",
        action="store_true",
        help="Render confusion matrix heatmap (requires matplotlib).",
    )
    p.add_argument(
        "--top-confused",
        type=int,
        default=50,
        help="Top-N most confused class pairs to export.",
    )
    p.add_argument(
        "--csv",
        action="store_true",
        help="Write per_class_accuracy.csv besides JSON.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of samples from split (debug).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="'auto' | 'cpu' | 'cuda'",
    )
    p.add_argument(
        "--centroid-file",
        type=Path,
        default=None,
        help="Override centroid file path if not in config (optional).",
    )
    ns = p.parse_args(argv)
    return Args(
        config=ns.config,
        checkpoint=ns.checkpoint,
        checkpoint_dir=ns.checkpoint_dir,
        split=ns.split,
        out_dir=ns.out_dir,
        batch_size=ns.batch_size,
        num_workers=ns.num_workers,
        pin_memory=ns.pin_memory,
        render_matrix=ns.render_matrix,
        top_confused=ns.top_confused,
        csv=ns.csv,
        limit=ns.limit,
        device=ns.device,
        centroid_file=ns.centroid_file,
    )


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    if not ckpt_dir.exists():
        return None
    cands = [p for p in ckpt_dir.glob("*.pt") if p.is_file()]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def load_label_map(path: Path) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def invert_label_map(label_map: Dict[str, int]) -> List[str]:
    inv = [""] * len(label_map)
    for k, v in label_map.items():
        if 0 <= v < len(inv):
            inv[v] = k
    return inv


def load_split_ids(path: Path) -> List[int]:
    ids: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.append(int(line))
            except ValueError:
                pass
    return ids


def load_metadata(metadata_path: Path) -> Tuple[Dict[int, str], Dict[int, bool]]:
    glyph_to_label: Dict[int, str] = {}
    diacritic_flags: Dict[int, bool] = {}
    if not metadata_path.exists():
        return glyph_to_label, diacritic_flags
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            gid = rec.get("glyph_id")
            lbl = rec.get("label")
            if gid is None or lbl is None:
                continue
            glyph_to_label[int(gid)] = str(lbl)
            diacritic_flags[int(gid)] = bool(rec.get("is_diacritic", False))
    return glyph_to_label, diacritic_flags


class GridDataset(Dataset):
    """
    Loads glyph grids (.u16 or .npy) and returns tensor + label_idx + glyph_id + diacritic_flag.
    """

    def __init__(
        self,
        glyph_ids: List[int],
        grids_dir: Path,
        label_map: Dict[str, int],
        glyph_to_label: Dict[int, str],
        diacritic: Dict[int, bool],
    ):
        self.glyph_ids = glyph_ids
        self.grids_dir = grids_dir
        self.label_map = label_map
        self.glyph_to_label = glyph_to_label
        self.diacritic = diacritic
        # Filter out glyphs without labels (safety)
        self.glyph_ids = [gid for gid in self.glyph_ids if gid in self.glyph_to_label]
        if not self.glyph_ids:
            raise RuntimeError("No glyph ids left after filtering label coverage.")

    def __len__(self):
        return len(self.glyph_ids)

    def _load_grid(self, gid: int) -> np.ndarray:
        u16 = self.grids_dir / f"{gid}.u16"
        npy = self.grids_dir / f"{gid}.npy"
        if u16.exists():
            arr = np.fromfile(u16, dtype=np.uint16)
            if arr.size != 256:
                raise ValueError(f"Corrupt u16 grid {u16}")
            return arr.reshape(16, 16)
        if npy.exists():
            arr = np.load(npy)
            if arr.shape != (16, 16):
                raise ValueError(f"Bad npy grid shape {npy}: {arr.shape}")
            return arr
        raise FileNotFoundError(f"Missing grid for glyph_id={gid}")

    def __getitem__(self, idx: int):
        gid = self.glyph_ids[idx]
        label_str = self.glyph_to_label[gid]
        label_idx = self.label_map[label_str]
        grid = self._load_grid(gid)
        t = torch.from_numpy(grid.astype("int64"))
        return (
            t,
            label_idx,
            gid,
            bool(self.diacritic.get(gid, False)),
        )


def collate(batch):
    grids, labels, gids, flags = zip(*batch)
    return (
        torch.stack(grids, dim=0),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(gids, dtype=torch.long),
        torch.tensor(flags, dtype=torch.bool),
    )


# ---------------------------------------------------------------------------
# Metrics & Helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def topk_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, ks=(1, 5)
) -> Dict[int, float]:
    out = {}
    maxk = max(ks)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    for k in ks:
        correct_k = correct[:k].reshape(-1).float().sum().item()
        out[k] = correct_k / targets.size(0)
    return out


def compute_macro_f1(conf: np.ndarray) -> float:
    # conf: (C,C)
    tp = np.diag(conf).astype(np.float64)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    valid_mask = (conf.sum(axis=1) > 0).astype(np.float64)
    return float((f1 * valid_mask).sum() / np.maximum(valid_mask.sum(), 1))


def compute_per_class_accuracy(conf: np.ndarray) -> np.ndarray:
    # correct / total (row-wise)
    correct = np.diag(conf)
    totals = conf.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.where(totals > 0, correct / totals, 0.0)
    return acc


def rank_confused_pairs(conf: np.ndarray, top_n: int) -> List[Dict[str, Any]]:
    C = conf.shape[0]
    pairs = []
    for i in range(C):
        for j in range(i + 1, C):
            a = conf[i, j]
            b = conf[j, i]
            if a == 0 and b == 0:
                continue
            # Score prioritizes mutuality & volume
            score = min(a, b) * (a + b)
            pairs.append(
                {
                    "i": i,
                    "j": j,
                    "i_to_j": int(a),
                    "j_to_i": int(b),
                    "total": int(a + b),
                    "score": int(score),
                }
            )
    pairs.sort(key=lambda x: (x["score"], x["total"]), reverse=True)
    return pairs[:top_n]


def load_centroids_if_needed(
    cfg: Dict, override: Optional[Path]
) -> Optional[np.ndarray]:
    if override and override.exists():
        return np.load(override)
    data_cfg = cfg.get("data", {}) or {}
    path = data_cfg.get("primitive_centroids", None)
    if path and Path(path).exists():
        try:
            return np.load(path)
        except Exception:
            return None
    return None


def build_model(
    cfg: Dict, num_labels: int, centroids: Optional[np.ndarray]
) -> nn.Module:
    return build_phase2_model(cfg, num_labels=num_labels, primitive_centroids=centroids)


def load_checkpoint(model: nn.Module, path: Path):
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict):
        for key in ("model_state", "model_state_dict"):
            if key in raw and isinstance(raw[key], dict):
                model.load_state_dict(raw[key], strict=True)
                return
        # maybe raw is direct state_dict
        state_like = [k for k, v in raw.items() if isinstance(v, torch.Tensor)]
        if state_like:
            model.load_state_dict(raw, strict=True)
            return
    raise RuntimeError(f"Unrecognized checkpoint format: {path}")


# ---------------------------------------------------------------------------
# Evaluation Core
# ---------------------------------------------------------------------------


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    track_diacritic: bool,
) -> Dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")  # accumulate manually
    total_loss = 0.0
    total_samples = 0
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    diac_total = 0
    diac_correct = 0
    with torch.no_grad():
        for grids, labels, gids, flags in loader:
            grids = grids.to(device)
            labels = labels.to(device)
            logits = model(grids)
            loss = criterion(logits, labels).item()
            total_loss += loss
            total_samples += grids.size(0)
            accs = topk_accuracy(logits, labels, ks=(1, 5))
            preds = torch.argmax(logits, dim=1)
            # confusion
            for t, p in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                conf[t, p] += 1
            if track_diacritic:
                flags = flags.to(device=device)
                mask = flags
                if mask.any():
                    diac_total += int(mask.sum().item())
                    diac_correct += int((preds[mask] == labels[mask]).sum().item())
    loss_avg = total_loss / max(1, total_samples)
    per_class_acc = compute_per_class_accuracy(conf)
    top1 = np.diag(conf).sum() / max(1, conf.sum())
    # For top5 we need to re-run (we stored only top1 above). We'll compute in-loop with topk logic.
    # Instead we re-aggregate quickly by storing top5 hits: re-pass? To keep memory low we just repeat logic:
    # Given we already computed per batch, we could accumulate but we didn't store; re-run only for top5 inside loop above (accs[5]).
    # Simpler: We captured accs each iteration but didn't accumulate. Adjust above to accumulate:
    # We'll patch: return also top5_sum, top5 denom. (Refactor below).
    # Instead just trust earlier specialization: We'll do a second pass computing top5. For large sets slight overhead is acceptable.
    # To keep performance reasonable, we accept a small overhead only if needed; for now we approximate by using confusion top1 only.
    metrics = {
        "loss": loss_avg,
        "accuracy_top1": top1,
        "confusion_matrix": conf,
        "per_class_accuracy": per_class_acc,
    }
    if track_diacritic and diac_total > 0:
        metrics["diacritic_subset_accuracy"] = diac_correct / diac_total
    return metrics


# Improved evaluation function that accumulates top1 & top5 precisely
def evaluate_full(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    track_diacritic: bool,
) -> Dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_samples = 0
    correct1 = 0.0
    correct5 = 0.0
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    diac_total = 0
    diac_correct = 0
    with torch.no_grad():
        for grids, labels, gids, flags in loader:
            grids = grids.to(device)
            labels = labels.to(device)
            logits = model(grids)
            loss = criterion(logits, labels).item()
            total_loss += loss
            bsz = grids.size(0)
            total_samples += bsz
            accs = topk_accuracy(logits, labels, ks=(1, 5))
            correct1 += accs[1] * bsz
            correct5 += accs[5] * bsz
            preds = torch.argmax(logits, dim=1)
            for t, p in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                conf[t, p] += 1
            if track_diacritic:
                mask = flags.to(device=device)
                if mask.any():
                    diac_total += int(mask.sum().item())
                    diac_correct += int((preds[mask] == labels[mask]).sum().item())
    loss_avg = total_loss / max(1, total_samples)
    per_class_acc = compute_per_class_accuracy(conf)
    metrics = {
        "loss": loss_avg,
        "accuracy_top1": correct1 / max(1, total_samples),
        "accuracy_top5": correct5 / max(1, total_samples),
        "confusion_matrix": conf,
        "per_class_accuracy": per_class_acc,
    }
    if track_diacritic and diac_total > 0:
        metrics["diacritic_subset_accuracy"] = diac_correct / diac_total
    return metrics


# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------


def run(args: Args) -> int:
    start_total = time.time()
    cfg = load_yaml(args.config)
    data_cfg = cfg.get("data", {}) or {}
    grids_dir = Path(data_cfg.get("grids_dir", "data/processed/grids"))
    label_map_path = Path(data_cfg.get("labels_file", "data/processed/label_map.json"))
    splits_dir = Path(data_cfg.get("root", "data/processed")) / "splits"
    split_file_map = {
        "train": data_cfg.get("index_train", splits_dir / "phase2_train_ids.txt"),
        "val": data_cfg.get("index_val", splits_dir / "phase2_val_ids.txt"),
        "test": data_cfg.get("index_test", splits_dir / "phase2_test_ids.txt"),
    }
    split_path = Path(split_file_map[args.split])
    metadata_path = Path("data/rasters/metadata.jsonl")
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    if not label_map_path.exists():
        raise FileNotFoundError(f"label_map.json missing: {label_map_path}")
    if not grids_dir.exists():
        raise FileNotFoundError(f"Grids directory missing: {grids_dir}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map = load_label_map(label_map_path)
    inv_labels = invert_label_map(label_map)
    num_classes = len(label_map)

    glyph_to_label, diacritic_flags = load_metadata(metadata_path)
    track_diacritic = "diacritic_subset_accuracy" in (cfg.get("metrics") or [])

    glyph_ids = load_split_ids(split_path)
    if args.limit and args.limit < len(glyph_ids):
        glyph_ids = glyph_ids[: args.limit]

    ds = GridDataset(glyph_ids, grids_dir, label_map, glyph_to_label, diacritic_flags)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate,
    )

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(
        f"[INFO] Evaluating Phase 2 model on split={args.split} | samples={len(ds)} | device={device}",
        flush=True,
    )

    # Checkpoint selection
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = find_latest_checkpoint(args.checkpoint_dir)  # type: ignore
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoints found in {args.checkpoint_dir}")
    print(f"[INFO] Using checkpoint: {ckpt_path}", flush=True)

    # Centroids (optional for embedding init)
    centroids = load_centroids_if_needed(cfg, args.centroid_file)

    # Build model
    model = build_model(cfg, num_labels=num_classes, centroids=centroids)
    load_checkpoint(model, ckpt_path)
    model.to(device)

    # Run evaluation
    metrics = evaluate_full(
        model,
        loader,
        device,
        num_classes=num_classes,
        track_diacritic=track_diacritic,
    )
    conf = metrics.pop("confusion_matrix")
    per_class_acc = metrics.pop("per_class_accuracy")

    macro_f1 = compute_macro_f1(conf)
    metrics["macro_f1"] = macro_f1

    # Confused pairs
    top_pairs = rank_confused_pairs(conf, args.top_confused)

    # Per-class structures
    per_class_records = []
    for idx, acc in enumerate(per_class_acc):
        support = int(conf[idx].sum())
        correct = int(conf[idx, idx])
        per_class_records.append(
            {
                "class_index": idx,
                "label": inv_labels[idx] if idx < len(inv_labels) else f"cls_{idx}",
                "accuracy": float(acc),
                "support": support,
                "correct": correct,
            }
        )

    # Write outputs
    (out_dir / "confusion_matrix.npz").write_bytes(_serialize_npz({"confusion": conf}))
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "per_class_accuracy.json", "w", encoding="utf-8") as f:
        json.dump(per_class_records, f, indent=2)
    with open(out_dir / "confusion_top_pairs.json", "w", encoding="utf-8") as f:
        json.dump(top_pairs, f, indent=2)

    if args.csv:
        with open(out_dir / "per_class_accuracy.csv", "w", encoding="utf-8") as f:
            f.write("class_index,label,accuracy,support,correct\n")
            for rec in per_class_records:
                f.write(
                    f"{rec['class_index']},{rec['label']},{rec['accuracy']:.6f},{rec['support']},{rec['correct']}\n"
                )

    # Render confusion matrix
    if args.render_matrix:
        if not _MPL:
            print(
                "[warn] matplotlib not available; skipping confusion matrix rendering.",
                file=sys.stderr,
            )
        else:
            render_confusion_matrix(
                conf,
                inv_labels,
                out_dir / "confusion_matrix.png",
                max_classes=100,  # if too many classes, show truncated
            )

    # Quick human summary
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Split: {args.split}\n")
        f.write(f"Samples: {len(ds)}\n")
        f.write(f"Checkpoint: {ckpt_path.name}\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write(f"macro_f1: {macro_f1:.6f}\n")
        if track_diacritic and "diacritic_subset_accuracy" in metrics:
            f.write(
                f"diacritic_subset_accuracy: {metrics['diacritic_subset_accuracy']:.6f}\n"
            )
        f.write(f"Top confused pairs (N={len(top_pairs)}):\n")
        for rec in top_pairs[:10]:
            f.write(
                f"  ({rec['i']}:{inv_labels[rec['i']]}) <-> ({rec['j']}:{inv_labels[rec['j']]}) "
                f"i->j={rec['i_to_j']} j->i={rec['j_to_i']} total={rec['total']} score={rec['score']}\n"
            )

    elapsed = time.time() - start_total
    print(
        f"[DONE] Evaluation complete in {elapsed:.2f}s | acc@1={metrics['accuracy_top1']:.4f} "
        f"acc@5={metrics['accuracy_top5']:.4f} macro_f1={macro_f1:.4f}",
        flush=True,
    )
    return 0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _serialize_npz(arr_dict: Dict[str, np.ndarray]) -> bytes:
    """Serialize arrays to bytes (np.savez to buffer)."""
    import io

    buf = io.BytesIO()
    np.savez(buf, **arr_dict)
    return buf.getvalue()


def render_confusion_matrix(
    conf: np.ndarray,
    labels: List[str],
    out_path: Path,
    max_classes: int = 100,
):
    """
    Render a truncated confusion matrix for large class counts.
    """
    C = conf.shape[0]
    if C > max_classes:
        idxs = list(range(max_classes))
        conf_plot = conf[:max_classes, :max_classes]
        labels_plot = [labels[i] for i in idxs]
        note = f"Showing first {max_classes}/{C} classes"
    else:
        conf_plot = conf
        labels_plot = labels
        note = None

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(
        np.log1p(conf_plot), interpolation="nearest", cmap="viridis"
    )  # log scale
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(conf_plot.shape[1]),
        yticks=np.arange(conf_plot.shape[0]),
        xticklabels=["" if len(l) > 10 else l for l in labels_plot],
        yticklabels=["" if len(l) > 10 else l for l in labels_plot],
        ylabel="True",
        xlabel="Predicted",
        title="Confusion Matrix (log1p scale)",
    )
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", rotation_mode="anchor")
    if note:
        ax.text(
            0.5,
            -0.12,
            note,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="gray",
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
