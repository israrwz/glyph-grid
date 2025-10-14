#!/usr/bin/env python3
"""
Phase 2 Training Script: Transformer over Primitive ID Grids
===========================================================

Trains the Phase 2 glyph classification model defined in:
  - configs/phase2.yaml
  - models/phase2_transformer.py (build_phase2_model)

Data Assumptions
----------------
Export pipeline (data/export_phase2_grids.py) produced:
  data/processed/
    grids/<glyph_id>.u16          # 512 bytes raw uint16 (16*16)
    label_map.json                # { label_string: class_index, ... }
    splits/phase2_train_ids.txt
           phase2_val_ids.txt
           phase2_test_ids.txt
  assets/centroids/primitive_centroids.npy (optional, prototype init)

Additionally we reuse raster metadata for auxiliary subset metrics:
  data/rasters/metadata.jsonl with records:
    { "glyph_id": int, "label": str, "is_diacritic": bool, ... }

Features
--------
- Config-driven (YAML).
- Dataset with lazy loading of grids (.u16 preferred, falls back to .npy).
- Optional centroids initialization for primitive embedding.
- Metrics: top-1 / top-5 accuracy, macro F1 (validation only), per-class accuracy (optional),
           diacritic subset accuracy (if flag present in metadata & requested).
- Early stopping + checkpointing (top-K + last).
- Mixed precision (AMP) optional.
- CSV logging + (optionally) TensorBoard (lightweight hook).
- Reproducibility (seed, deterministic flag).

Not Implemented (Future Extensions)
-----------------------------------
- Distributed training (DDP).
- Attention heatmap dumping mid-training (use eval script separately).
- Soft-grid / top-k primitive probability ingestion.
- Class weighting / focal loss (extend loss builder).
- Joint end-to-end fine-tuning (Phase 1 + Phase 2).

Usage
-----
  python -m train.train_phase2 --config configs/phase2.yaml

Environment Overrides (optional)
--------------------------------
  PHASE2_CFG=configs/phase2.yaml python -m train.train_phase2

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise RuntimeError("Missing dependency: pyyaml (pip install pyyaml)") from e

try:
    import numpy as np
except ImportError as e:  # pragma: no cover
    raise RuntimeError("Missing dependency: numpy (pip install numpy)") from e

try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
except ImportError as e:  # pragma: no cover
    raise RuntimeError("PyTorch is required (pip install torch).") from e

# Optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter  # pragma: no cover

    _TB_AVAILABLE = True
except Exception:  # pragma: no cover
    _TB_AVAILABLE = False

# Local model factory
try:
    from models.phase2_transformer import build_phase2_model
except ImportError as e:
    raise RuntimeError(
        "Could not import phase2 transformer model. Ensure PYTHONPATH includes project root."
    ) from e

# Optional CNN import (only needed if architecture == 'cnn'); delay errors until selected.
try:
    from models.phase2_cnn import build_phase2_cnn_model  # type: ignore

    _PHASE2_CNN_AVAILABLE = True
except Exception:
    _PHASE2_CNN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class Phase2Config:
    raw: Dict[str, Any]
    path: Path

    @staticmethod
    def from_yaml(path: Path) -> "Phase2Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return Phase2Config(raw=raw, path=path)

    def get(self, *keys, default=None):
        node = self.raw
        for k in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(k, None)
            if node is None:
                return default
        return node


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True  # pragma: no cover
        torch.backends.cudnn.benchmark = False
    else:  # pragma: no cover
        torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class GlyphGridDataset(Dataset):
    """
    Loads per-glyph primitive ID grids (16x16) and maps to label indices.

    File resolution order for each glyph_id:
      <grids_dir>/<id>.u16  (raw) else <grids_dir>/<id>.npy

    Args:
      glyph_ids: list[int] training or validation split.
      grids_dir: Path to directory with .u16 / .npy grids.
      label_map: dict label_string -> label_index
      glyph_to_label: dict glyph_id -> label_string
      diacritic_flags: dict glyph_id -> bool (optional)
      cache: if True, keep small grids in memory (useful if dataset small)
    """

    def __init__(
        self,
        glyph_ids: List[int],
        grids_dir: Path,
        label_map: Dict[str, int],
        glyph_to_label: Dict[int, str],
        diacritic_flags: Optional[Dict[int, bool]] = None,
        cache: bool = False,
        cast_dtype: torch.dtype = torch.int64,
    ):
        self.glyph_ids = glyph_ids
        self.grids_dir = grids_dir
        self.label_map = label_map
        self.glyph_to_label = glyph_to_label
        self.diacritic_flags = diacritic_flags or {}
        self.cache_enabled = cache
        self.cast_dtype = cast_dtype
        self._cache: Dict[int, torch.Tensor] = {}

        missing = [gid for gid in self.glyph_ids if gid not in self.glyph_to_label]
        if missing:
            # Skip silently? For now we remove them while warning.
            print(
                f"[warn] {len(missing)} glyph_ids missing label mapping; they will be skipped.",
                file=sys.stderr,
            )
            self.glyph_ids = [
                gid for gid in self.glyph_ids if gid in self.glyph_to_label
            ]

        if len(self.glyph_ids) == 0:
            raise RuntimeError("GlyphGridDataset has zero usable glyph ids.")

    def __len__(self):
        return len(self.glyph_ids)

    def _load_grid_u16(self, path: Path) -> np.ndarray:
        raw = np.fromfile(path, dtype=np.uint16)
        if raw.size != 256:
            raise ValueError(
                f"Corrupt .u16 grid {path} (expected 256 values, got {raw.size})"
            )
        return raw.reshape(16, 16)

    def _load_grid(self, gid: int) -> torch.Tensor:
        if self.cache_enabled and gid in self._cache:
            return self._cache[gid]
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
            raise FileNotFoundError(f"No grid file found for glyph_id={gid}")
        t = torch.from_numpy(arr.astype("int64"))
        if self.cache_enabled:
            self._cache[gid] = t
        return t

    def __getitem__(self, idx: int):
        gid = self.glyph_ids[idx]
        label_str = self.glyph_to_label.get(gid)
        if label_str is None:
            raise KeyError(f"Missing label string for glyph_id={gid}")
        label_idx = self.label_map.get(label_str)
        if label_idx is None:
            raise KeyError(
                f"Label '{label_str}' not found in label_map.json (glyph_id={gid})"
            )
        grid = self._load_grid(gid)  # (16,16) int64
        return (
            grid.to(self.cast_dtype),
            label_idx,
            gid,
            bool(self.diacritic_flags.get(gid, False)),
        )


# ---------------------------------------------------------------------------
# Data / Metadata Utilities
# ---------------------------------------------------------------------------


def load_label_map(path: Path) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def load_metadata_labels(metadata_path: Path) -> Tuple[Dict[int, str], Dict[int, bool]]:
    """
    Returns:
      glyph_id -> label_string
      glyph_id -> is_diacritic (bool)
    """
    glyph_to_label: Dict[int, str] = {}
    diacritic_flags: Dict[int, bool] = {}
    if not metadata_path.exists():
        print(
            f"[warn] metadata.jsonl not found at {metadata_path}; subset metrics may be unavailable.",
            file=sys.stderr,
        )
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
            label = rec.get("label")
            if gid is None or label is None:
                continue
            glyph_to_label[int(gid)] = str(label)
            diacritic_flags[int(gid)] = bool(rec.get("is_diacritic", False))
    return glyph_to_label, diacritic_flags


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@torch.no_grad()
def accuracy_topk(
    logits: torch.Tensor, targets: torch.Tensor, topk=(1,)
) -> List[float]:
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    res: List[float] = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum().item()
        res.append(correct_k / targets.size(0))
    return res


def compute_macro_f1(
    confusion: torch.Tensor, training_seen: "Optional[torch.Tensor]" = None
) -> float:
    """
    confusion: (C,C) tensor where rows = true, cols = pred.
    training_seen: Optional boolean mask (C,) indicating which classes were seen
                   in training (frequency > 0). Classes unseen in training are
                   excluded from the macro F1 denominator even if they appear
                   in validation.
    """
    tp = torch.diag(confusion)
    fp = confusion.sum(dim=0) - tp
    fn = confusion.sum(dim=1) - tp
    precision = tp / torch.clamp(tp + fp, min=1)
    recall = tp / torch.clamp(tp + fn, min=1)
    f1 = 2 * precision * recall / torch.clamp(precision + recall, min=1e-12)

    # Classes with at least one validation instance
    valid = (confusion.sum(dim=1) > 0).float()

    # Exclude classes not seen in training if mask provided
    if training_seen is not None:
        training_seen = training_seen.to(valid.device).float()
        valid = valid * (training_seen > 0)

    macro = (f1 * valid).sum().item() / max(1.0, valid.sum().item())
    return macro


def update_confusion(conf: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor):
    for p, t in zip(preds.tolist(), targets.tolist()):
        conf[t, p] += 1


# ---------------------------------------------------------------------------
# Checkpoint Utils
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, Any],
    scheduler: Optional[Any] = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }
    if scheduler and hasattr(scheduler, "state_dict"):
        payload["scheduler_state"] = scheduler.state_dict()
    torch.save(payload, path)


# ---------------------------------------------------------------------------
# Training / Validation
# ---------------------------------------------------------------------------


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    *,
    mixed_precision: bool = False,
    is_train: bool = True,
    compute_confusion: bool = False,
    num_classes: int = 0,
    track_subset_diacritic: bool = False,
) -> Dict[str, Any]:
    model.train(is_train)
    scaler = torch.cuda.amp.GradScaler(
        enabled=mixed_precision and is_train and device.type == "cuda"
    )

    total_loss = 0.0
    total_samples = 0
    total_correct1 = 0
    total_correct5 = 0

    if compute_confusion and num_classes > 0:
        confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    else:
        confusion = None

    subset_total = 0
    subset_correct = 0

    for batch in loader:
        grids, labels, glyph_ids, diacritic_flags = batch
        grids = grids.to(device, non_blocking=True)  # (B,16,16)
        labels = torch.as_tensor(labels, device=device, dtype=torch.long)

        with torch.cuda.amp.autocast(enabled=mixed_precision and device.type == "cuda"):
            logits = model(grids)  # (B,C)
            loss = loss_fn(logits, labels)

        if is_train:
            optimizer.zero_grad(set_to_none=True)  # type: ignore
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)  # type: ignore
                scaler.update()
            else:
                loss.backward()
                optimizer.step()  # type: ignore

        with torch.no_grad():
            bsz = grids.size(0)
            total_samples += bsz
            total_loss += loss.item() * bsz
            acc1, acc5 = accuracy_topk(logits, labels, topk=(1, 5))
            total_correct1 += acc1 * bsz
            total_correct5 += acc5 * bsz

            preds = torch.argmax(logits, dim=1)
            if confusion is not None:
                update_confusion(confusion, preds.cpu(), labels.cpu())

            if track_subset_diacritic:
                # diacritic_flags is a list[bool] or tensor; convert to mask
                if isinstance(diacritic_flags, torch.Tensor):
                    mask = diacritic_flags.to(device=device)
                else:
                    mask = torch.as_tensor(
                        diacritic_flags, dtype=torch.bool, device=device
                    )
                subset_count = mask.sum().item()
                if subset_count > 0:
                    subset_total += subset_count
                    subset_correct += (preds[mask] == labels[mask]).sum().item()

    metrics = {
        "loss": total_loss / max(1, total_samples),
        "accuracy_top1": total_correct1 / max(1, total_samples),
        "accuracy_top5": total_correct5 / max(1, total_samples),
    }
    if subset_total > 0:
        metrics["diacritic_subset_accuracy"] = subset_correct / subset_total
    if confusion is not None:
        metrics["confusion"] = confusion
    return metrics


# ---------------------------------------------------------------------------
# Optim + Scheduler + Loss
# ---------------------------------------------------------------------------


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    name = (cfg.get("name") or "adamw").lower()
    lr = float(cfg.get("lr", 5e-4))
    wd = float(cfg.get("weight_decay", 1e-4))
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
            betas=tuple(cfg.get("betas", (0.9, 0.999))),
            eps=float(cfg.get("eps", 1e-8)),
        )
    elif name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
            momentum=float(cfg.get("momentum", 0.9)),
            nesterov=bool(cfg.get("nesterov", True)),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    sched_cfg: Dict[str, Any],
    total_epochs: int,
):
    strat = (sched_cfg.get("strategy") or "cosine").lower()
    if strat == "cosine":
        warmup = int(sched_cfg.get("warmup_epochs", 0))
        min_scale = float(sched_cfg.get("min_lr_scale", 0.1))

        def lr_lambda(epoch: int):
            if epoch < warmup and warmup > 0:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / max(1, (total_epochs - warmup))
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return min_scale + (1 - min_scale) * cosine_factor

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif strat == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max"
            if sched_cfg.get("plateau", {}).get("mode", "max") == "max"
            else "min",
            factor=float(sched_cfg.get("plateau", {}).get("factor", 0.5)),
            patience=int(sched_cfg.get("plateau", {}).get("patience", 3)),
            min_lr=float(sched_cfg.get("plateau", {}).get("min_lr", 1e-6)),
            threshold=1e-4,
        )
    else:
        return None


def build_loss(cfg: Dict[str, Any]) -> nn.Module:
    smoothing = float(cfg.get("label_smoothing", 0.0) or 0.0)
    if smoothing <= 0:
        return nn.CrossEntropyLoss()
    return nn.CrossEntropyLoss(label_smoothing=smoothing)


# ---------------------------------------------------------------------------
# Global collate function (pickle-safe for DataLoader workers)
# ---------------------------------------------------------------------------
def collate_grids(batch):
    grids, labels, gids, dia = zip(*batch)
    return (
        torch.stack(grids, dim=0),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(gids, dtype=torch.long),
        torch.tensor(dia, dtype=torch.bool),
    )


# ---------------------------------------------------------------------------
# Main Training Orchestration
# ---------------------------------------------------------------------------


def train_phase2(cfg: Phase2Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Basic config values
    seed = int(cfg.get("seed", default=42))
    deterministic = bool(cfg.get("deterministic", default=True))
    set_seed(seed, deterministic=deterministic)

    data_cfg = cfg.get("data", default={}) or {}
    grids_dir = Path(data_cfg.get("grids_dir", "data/processed/grids"))
    label_map_path = Path(data_cfg.get("labels_file", "data/processed/label_map.json"))
    centroid_path = data_cfg.get("primitive_centroids", None)
    centroid_path = Path(centroid_path) if centroid_path else None
    splits_root = Path(label_map_path.parent / "splits")
    index_train = Path(
        data_cfg.get("index_train", splits_root / "phase2_train_ids.txt")
    )
    index_val = Path(data_cfg.get("index_val", splits_root / "phase2_val_ids.txt"))
    index_test = Path(data_cfg.get("index_test", splits_root / "phase2_test_ids.txt"))

    metadata_path = Path("data/rasters/metadata.jsonl")  # assumption

    if not label_map_path.exists():
        raise FileNotFoundError(f"Label map not found: {label_map_path}")
    if not grids_dir.exists():
        raise FileNotFoundError(f"Grids directory missing: {grids_dir}")

    label_map = load_label_map(label_map_path)
    num_labels = len(label_map)

    glyph_to_label, diacritic_flags = load_metadata_labels(metadata_path)

    train_ids = load_split_ids(index_train) if index_train.exists() else []
    val_ids = load_split_ids(index_val) if index_val.exists() else []
    test_ids = load_split_ids(index_test) if index_test.exists() else []

    if not train_ids or not val_ids:
        print(
            "[warn] One or more split files missing/empty; using all available glyphs as train, first 5% as val.",
            file=sys.stderr,
        )
        all_ids = sorted(
            int(p.stem) for p in grids_dir.glob("*.u16") if p.stem.isdigit()
        )
        if not all_ids:
            raise RuntimeError("No grid files found to auto-generate splits.")
        split_pt = max(1, int(0.95 * len(all_ids)))
        train_ids = all_ids[:split_pt]
        val_ids = all_ids[split_pt:]
        test_ids = val_ids

    # Datasets / loaders
    training_cfg = cfg.get("training", default={}) or {}
    batch_size = int(training_cfg.get("batch_size", 128))
    eval_batch_size = int(training_cfg.get("eval_batch_size", batch_size))
    num_workers = int(training_cfg.get("num_workers", 4))
    pin_memory = bool(training_cfg.get("pin_memory", True))

    track_diacritic = "diacritic_subset_accuracy" in (
        cfg.get("metrics", default=[]) or []
    )

    train_ds = GlyphGridDataset(
        train_ids,
        grids_dir,
        label_map,
        glyph_to_label,
        diacritic_flags if track_diacritic else None,
    )
    val_ds = GlyphGridDataset(
        val_ids,
        grids_dir,
        label_map,
        glyph_to_label,
        diacritic_flags if track_diacritic else None,
    )
    test_ds = GlyphGridDataset(
        test_ids,
        grids_dir,
        label_map,
        glyph_to_label,
        diacritic_flags if track_diacritic else None,
    )

    # Local collate removed; using top-level collate_grids for multiprocessing safety.

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_grids,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_grids,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_grids,
    )

    # Model
    centroids_array = None
    if centroid_path and centroid_path.exists():
        try:
            centroids_array = np.load(centroid_path)
        except Exception as e:
            print(f"[warn] Failed loading centroids: {e}", file=sys.stderr)
    # ------------------------------------------------------------------
    # Model (branch on architecture)
    # ------------------------------------------------------------------
    architecture = (cfg.get("model", "architecture") or "transformer").lower()
    if architecture == "cnn":
        if not _PHASE2_CNN_AVAILABLE:
            raise RuntimeError(
                "Requested model.architecture=cnn but phase2_cnn module not available."
            )
        model = build_phase2_cnn_model(
            cfg.raw, num_labels=num_labels, primitive_centroids=centroids_array
        )
        print("[INFO] Using Phase 2 CNN architecture", flush=True)
    else:
        model = build_phase2_model(
            cfg.raw, num_labels=num_labels, primitive_centroids=centroids_array
        )
        print("[INFO] Using Phase 2 Transformer architecture", flush=True)
    model.to(device)

    # Loss / Optim / Scheduler
    loss_cfg = cfg.get("loss", default={}) or {}
    loss_fn = build_loss(loss_cfg)
    optimizer = build_optimizer(model, cfg.get("optim", default={}) or {})
    epochs = int(training_cfg.get("epochs", 40))
    scheduler_cfg = cfg.get("scheduler", default={}) or {}
    scheduler = build_scheduler(optimizer, scheduler_cfg, total_epochs=epochs)

    # ------------------------------------------------------------------
    # Auto inverse-frequency class weighting (Bundle 2)
    # Trigger when loss.class_weights == 'auto_inverse_freq'
    # Computes weights: w_i = (N / (count_i + eps)) ** alpha, normalized to mean=1
    # ------------------------------------------------------------------
    class_weight_mode = (
        loss_cfg.get("class_weights") if isinstance(loss_cfg, dict) else None
    ) or None
    # Build frequency map from training glyph IDs (needed for diagnostics even if not weighting)
    class_counts = torch.zeros(num_labels, dtype=torch.float)
    for gid in train_ids:
        label_str = glyph_to_label.get(gid)
        if label_str is None:
            continue
        idx = label_map.get(label_str)
        if idx is not None:
            class_counts[idx] += 1
    if class_weight_mode == "auto_inverse_freq":
        eps = 1.0
        total = class_counts.sum().item()
        alpha = float(loss_cfg.get("class_weights_alpha", 0.4))
        inv = (total / (class_counts + eps)) ** alpha
        inv[class_counts == 0] = (
            0.0  # unseen classes remain 0 (will be ignored by loss smoothing)
        )
        # Normalize to mean 1 over seen classes
        seen = class_counts > 0
        if seen.any():
            inv_seen_mean = inv[seen].mean()
            if inv_seen_mean > 0:
                inv = inv / inv_seen_mean
        # Rebuild loss_fn with weights if using CrossEntropyLoss
        if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
            smoothing = float(loss_cfg.get("label_smoothing", 0.0) or 0.0)
            loss_fn = torch.nn.CrossEntropyLoss(
                weight=inv.to(device),
                label_smoothing=smoothing if smoothing > 0 else 0.0,
            )
            print(
                "[INFO] Applied auto inverse-frequency class weights "
                f"(alpha={alpha}, mean={inv[seen].mean().item():.3f})",
                flush=True,
            )

    early_cfg = cfg.get("early_stopping", default={}) or {}
    use_early = bool(early_cfg.get("enabled", True))
    monitor_metric = early_cfg.get("metric", "val/accuracy_top1")
    monitor_mode = early_cfg.get("mode", "max")
    patience = int(early_cfg.get("patience", 6))
    min_delta = float(early_cfg.get("min_delta", 0.0005))
    best_score = -float("inf") if monitor_mode == "max" else float("inf")
    no_improve_epochs = 0

    mixed_precision = training_cfg.get("mixed_precision", "amp") == "amp"
    grad_clip = float(training_cfg.get("grad_clip_norm", 0.0) or 0.0)
    gradient_accum = int(training_cfg.get("gradient_accumulation_steps", 1))
    # New logging / limiting parameters
    log_interval_steps = int(training_cfg.get("log_interval_steps", 100) or 100)
    limit_train_batches = training_cfg.get("limit_train_batches", None)
    # Token ID dropout probability (randomly replace primitive IDs with 0 / EMPTY during training)
    token_id_dropout = float(training_cfg.get("token_id_dropout", 0.0) or 0.0)
    # Automatic unweighted fine-tune epoch (switch off class weights & token dropout)
    unweighted_finetune_epoch = int(
        training_cfg.get("unweighted_finetune_after_epoch", -1) or -1
    )
    fine_tune_switched = False  # guarded inside loop
    # Resume support (path or env var PHASE2_RESUME)
    resume_from = training_cfg.get("resume_from") or os.environ.get("PHASE2_RESUME")
    if isinstance(limit_train_batches, float) and 0 < limit_train_batches <= 1:
        # Interpret as fraction of total steps
        # Will resolve after DataLoader is built (need len(train_loader))
        pass  # placeholder; resolved after train_loader creation
    elif isinstance(limit_train_batches, int):
        if limit_train_batches <= 0:
            limit_train_batches = None
    else:
        limit_train_batches = None

    # Logging / checkpoints
    ckpt_cfg = cfg.get("checkpoint", default={}) or {}
    ckpt_dir = Path(ckpt_cfg.get("dir", "checkpoints/phase2"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_top_k = int(ckpt_cfg.get("save_top_k", 3))
    save_last = bool(ckpt_cfg.get("save_last", True))
    filename_pattern = ckpt_cfg.get(
        "filename_pattern", "epoch{epoch:02d}-val{val_accuracy_top1:.4f}.pt"
    )

    tb_writer = None
    logging_cfg = cfg.get("logging", default={}) or {}
    backend = logging_cfg.get("backend", "tensorboard")
    tb_dir = logging_cfg.get("tensorboard_dir", "logs/phase2")
    if backend == "tensorboard" and _TB_AVAILABLE:
        tb_writer = SummaryWriter(log_dir=tb_dir)
    csv_log_dir = Path(logging_cfg.get("tensorboard_dir", "logs/phase2"))
    csv_log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_log_dir / "metrics.csv"
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "epoch",
                    "phase",
                    "loss",
                    "accuracy_top1",
                    "accuracy_top5",
                    "macro_f1_seen",
                    "macro_f1_all",
                    "diacritic_subset_accuracy",
                    "lr",
                    "time_sec",
                ]
            )

    top_checkpoints: List[Tuple[float, Path]] = []
    start_epoch = 0
    if resume_from:
        rp = Path(resume_from)
        if rp.exists():
            try:
                payload = torch.load(rp, map_location=device)
                model.load_state_dict(payload.get("model_state", {}))
                if "optimizer_state" in payload:
                    optimizer.load_state_dict(payload["optimizer_state"])
                if scheduler and "scheduler_state" in payload:
                    try:
                        scheduler.load_state_dict(payload["scheduler_state"])
                    except Exception:
                        pass
                start_epoch = int(payload.get("epoch", 0))
                metrics_payload = payload.get("metrics", {})
                if monitor_mode == "max":
                    best_score = metrics_payload.get("val_accuracy_top1", best_score)
                else:
                    best_score = metrics_payload.get("val_loss", best_score)
                print(
                    f"[RESUME] Loaded checkpoint '{resume_from}' (epoch {start_epoch})",
                    flush=True,
                )
                # Establish / refresh best.pt symlink
                best_link = ckpt_dir / "best.pt"
                try:
                    if best_link.exists() or best_link.is_symlink():
                        best_link.unlink()
                    if rp.parent == ckpt_dir:
                        best_link.symlink_to(rp.name)
                    else:
                        shutil.copy2(rp, best_link)
                except Exception:
                    pass
            except Exception as e:
                print(
                    f"[warn] Failed to resume from {resume_from}: {e}", file=sys.stderr
                )
        else:
            print(
                f"[warn] resume_from path does not exist: {resume_from}",
                file=sys.stderr,
            )

    print(
        f"[INFO] Phase 2 Training | device={device} | train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} | labels={num_labels}",
        flush=True,
    )
    # Reconstruct best.pt symlink/copy if missing (e.g., after cleanup)
    best_link = ckpt_dir / "best.pt"
    if not best_link.exists():
        import re

        epoch_ckpts = []
        for p in ckpt_dir.glob("epoch*-val*.pt"):
            m = re.match(r"epoch(\\d+)-val", p.name)
            if m:
                try:
                    epoch_ckpts.append((int(m.group(1)), p))
                except ValueError:
                    pass
        if epoch_ckpts:
            epoch_ckpts.sort(key=lambda x: x[0], reverse=True)
            _, best_candidate = epoch_ckpts[0]
            try:
                if best_link.exists() or best_link.is_symlink():
                    best_link.unlink()
                # Prefer symlink; fallback to copy
                try:
                    best_link.symlink_to(best_candidate.name)
                except Exception:
                    import shutil

                    shutil.copy2(best_candidate, best_link)
                print(
                    f"[INFO] Reconstructed best.pt -> {best_candidate.name}", flush=True
                )
            except Exception as e:
                print(f"[warn] Failed to reconstruct best.pt: {e}", file=sys.stderr)

    for epoch in range(start_epoch + 1, epochs + 1):
        start_time = time.time()

        # Training epoch
        model.train(True)
        # Gradient accumulation loop
        running_metrics = {
            "loss_sum": 0.0,
            "samples": 0,
            "correct1": 0.0,
            "correct5": 0.0,
            "subset_total": 0,
            "subset_correct": 0,
        }
        scaler = torch.cuda.amp.GradScaler(
            enabled=mixed_precision and device.type == "cuda"
        )

        for step, batch in enumerate(train_loader):
            grids, labels, glyph_ids, diacritic_flags = batch
            grids = grids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # Token ID dropout augmentation
            if token_id_dropout > 0.0 and model.training:
                # Replace a random subset of primitive IDs with 0 (EMPTY)
                rand_mask = torch.rand_like(grids.float()) < token_id_dropout
                if rand_mask.any():
                    grids = grids.masked_fill(rand_mask, 0)
            with torch.cuda.amp.autocast(
                enabled=mixed_precision and device.type == "cuda"
            ):
                logits = model(grids)
                loss = loss_fn(logits, labels) / gradient_accum
            if mixed_precision and scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (step + 1) % gradient_accum == 0:
                if grad_clip > 0:
                    if mixed_precision and scaler.is_enabled():
                        scaler.unscale_(optimizer)  # type: ignore
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if mixed_precision and scaler.is_enabled():
                    scaler.step(optimizer)  # type: ignore
                    scaler.update()
                else:
                    optimizer.step()  # type: ignore
                optimizer.zero_grad(set_to_none=True)  # type: ignore

            # Metrics accumulation
            with torch.no_grad():
                bsz = grids.size(0)
                running_metrics["samples"] += bsz
                running_metrics["loss_sum"] += loss.item() * bsz * gradient_accum
                acc1, acc5 = accuracy_topk(logits, labels, (1, 5))
                running_metrics["correct1"] += acc1 * bsz
                running_metrics["correct5"] += acc5 * bsz
                if track_diacritic:
                    mask = diacritic_flags.to(device=device)
                    subset_count = mask.sum().item()
                    if subset_count > 0:
                        preds = torch.argmax(logits, dim=1)
                        running_metrics["subset_total"] += subset_count
                        running_metrics["subset_correct"] += (
                            (preds[mask] == labels[mask]).sum().item()
                        )

            # Resolve fractional limit after knowing loader length
            if isinstance(limit_train_batches, float) and 0 < limit_train_batches <= 1:
                total_steps = len(train_loader)
                limit_train_batches = int(
                    max(1, round(limit_train_batches * total_steps))
                )

            global_step = step + 1
            if log_interval_steps > 0 and (
                global_step == 1 or global_step % log_interval_steps == 0
            ):
                avg_loss = running_metrics["loss_sum"] / max(
                    1, running_metrics["samples"]
                )
                avg_acc1 = running_metrics["correct1"] / max(
                    1, running_metrics["samples"]
                )
                avg_acc5 = running_metrics["correct5"] / max(
                    1, running_metrics["samples"]
                )
                if track_diacritic and running_metrics["subset_total"] > 0:
                    avg_dia = (
                        running_metrics["subset_correct"]
                        / running_metrics["subset_total"]
                    )
                    dia_str = f" dia_acc={avg_dia:.4f}"
                else:
                    dia_str = ""
                print(
                    f"[EPOCH {epoch:03d}][{global_step}/{len(train_loader)}] "
                    f"step_loss={(loss.item() * gradient_accum):.4f} avg_loss={avg_loss:.4f} "
                    f"acc@1={avg_acc1:.4f} acc@5={avg_acc5:.4f}{dia_str} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e}",
                    flush=True,
                )

            if (
                limit_train_batches
                and isinstance(limit_train_batches, int)
                and global_step >= limit_train_batches
            ):
                print(
                    f"[EPOCH {epoch:03d}] Reached limit_train_batches={limit_train_batches}; stopping early.",
                    flush=True,
                )
                break

        train_loss = running_metrics["loss_sum"] / max(1, running_metrics["samples"])
        train_acc1 = running_metrics["correct1"] / max(1, running_metrics["samples"])
        train_acc5 = running_metrics["correct5"] / max(1, running_metrics["samples"])
        train_subset_acc = (
            running_metrics["subset_correct"] / running_metrics["subset_total"]
            if running_metrics["subset_total"] > 0
            else None
        )

        # Validation
        val_stats = run_epoch(
            model,
            val_loader,
            loss_fn,
            optimizer=None,
            device=device,
            mixed_precision=False,
            is_train=False,
            compute_confusion=True,
            num_classes=num_labels,
            track_subset_diacritic=track_diacritic,
        )

        # Compute macro F1 variants if confusion matrix available
        macro_f1_seen = None
        macro_f1_all = None
        if "confusion" in val_stats:
            conf_float = val_stats["confusion"].float()
            training_seen_mask = (
                (class_counts > 0) if "class_counts" in locals() else None
            )
            macro_f1_all = compute_macro_f1(conf_float, None)
            macro_f1_seen = compute_macro_f1(conf_float, training_seen_mask)
            # ------------------------------------------------------------------
            # Diagnostics: label coverage & per-class accuracy histogram
            # ------------------------------------------------------------------
            try:
                conf = val_stats["confusion"]
                true_counts = conf.sum(dim=1)
                correct_diag = torch.diag(conf)
                nonzero = (true_counts > 0).sum().item()
                covered = (correct_diag > 0).sum().item()
                coverage_pct = 100.0 * covered / max(1, nonzero)
                # Per-class accuracy for seen classes
                acc_per_class = torch.zeros_like(true_counts, dtype=torch.float)
                seen_mask = true_counts > 0
                acc_per_class[seen_mask] = (
                    correct_diag[seen_mask].float() / true_counts[seen_mask].float()
                )
                # Histogram bins
                bins = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.000001]
                bin_counts = [0] * (len(bins) - 1)
                apc = acc_per_class[seen_mask].tolist()
                for v in apc:
                    for b in range(len(bins) - 1):
                        if bins[b] <= v < bins[b + 1]:
                            bin_counts[b] += 1
                            break
                print(
                    f"[VAL][diag] coverage={covered}/{nonzero} ({coverage_pct:.1f}%) "
                    f"acc_hist={bin_counts}",
                    flush=True,
                )
                # Head / Tail diagnostics (frequency deciles) - improved to exclude unseen classes
                try:
                    # Classes actually seen in training (frequency > 0)
                    seen_mask_counts = class_counts > 0
                    unseen_count = (~seen_mask_counts).sum().item()
                    seen_indices = torch.nonzero(seen_mask_counts, as_tuple=False).view(
                        -1
                    )
                    if seen_indices.numel() > 0:
                        seen_counts = class_counts[seen_indices]
                        # Sort only seen classes by frequency
                        seen_counts_sorted, seen_idx_sorted_local = torch.sort(
                            seen_counts
                        )
                        seen_sorted_global_idx = seen_indices[seen_idx_sorted_local]
                        decile = max(1, seen_sorted_global_idx.numel() // 10)
                        tail_seen = seen_sorted_global_idx[:decile]
                        head_seen = seen_sorted_global_idx[-decile:]
                        # Accuracy values
                        head_acc_vals = acc_per_class[head_seen]
                        tail_acc_vals = acc_per_class[tail_seen]
                        head_acc_mean = (
                            head_acc_vals.mean().item()
                            if head_acc_vals.numel()
                            else 0.0
                        )
                        tail_acc_mean = (
                            tail_acc_vals.mean().item()
                            if tail_acc_vals.numel()
                            else 0.0
                        )
                        tail_acc_nonzero = (
                            tail_acc_vals[tail_acc_vals > 0].mean().item()
                            if (tail_acc_vals > 0).any()
                            else 0.0
                        )
                        head_freq_mean = (
                            class_counts[head_seen].mean().item()
                            if head_seen.numel()
                            else 0.0
                        )
                        tail_freq_mean = (
                            class_counts[tail_seen].mean().item()
                            if tail_seen.numel()
                            else 0.0
                        )
                    else:
                        head_acc_mean = tail_acc_mean = tail_acc_nonzero = 0.0
                        head_freq_mean = tail_freq_mean = 0.0
                        decile = 0
                        unseen_count = int(class_counts.numel())
                    print(
                        f"[VAL][diag2] head_acc={head_acc_mean:.3f} tail_acc={tail_acc_mean:.3f} "
                        f"tail_acc_nonzero={tail_acc_nonzero:.3f} unseen_classes={unseen_count} "
                        f"head_freq_mean={head_freq_mean:.1f} tail_freq_mean={tail_freq_mean:.1f}",
                        flush=True,
                    )
                    # Frequency distribution logging every 2 epochs
                    if epoch % 2 == 0:
                        freq_bins = [0, 1, 2, 5, 10, 20, 50, 100, 1_000_000]
                        counts_per_bin = [0] * (len(freq_bins) - 1)
                        for c in class_counts.tolist():
                            for b in range(len(freq_bins) - 1):
                                if freq_bins[b] <= c < freq_bins[b + 1]:
                                    counts_per_bin[b] += 1
                                    break
                        print(
                            f"[VAL][freq] train_class_freq_bins={counts_per_bin} bins={freq_bins}",
                            flush=True,
                        )
                except Exception as e_head_tail:
                    print(
                        f"[warn] head/tail diagnostics failed: {e_head_tail}",
                        file=sys.stderr,
                    )
            except Exception as diag_e:
                print(f"[warn] diagnostics failed: {diag_e}", file=sys.stderr)

        # Scheduler step
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_stats["accuracy_top1"])
            else:
                scheduler.step()

        # Logging
        epoch_time = time.time() - start_time
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    epoch,
                    "train",
                    f"{train_loss:.6f}",
                    f"{train_acc1:.6f}",
                    f"{train_acc5:.6f}",
                    "",  # macro_f1_seen not computed for train
                    "",  # macro_f1_all not computed for train
                    f"{(train_subset_acc or 0):.6f}",
                    f"{current_lr:.6e}",
                    f"{epoch_time:.2f}",
                ]
            )
            w.writerow(
                [
                    epoch,
                    "val",
                    f"{val_stats['loss']:.6f}",
                    f"{val_stats['accuracy_top1']:.6f}",
                    f"{val_stats['accuracy_top5']:.6f}",
                    f"{(macro_f1_seen or 0):.6f}",
                    f"{(macro_f1_all or 0):.6f}",
                    f"{val_stats.get('diacritic_subset_accuracy', 0.0):.6f}",
                    f"{current_lr:.6e}",
                    f"{epoch_time:.2f}",
                ]
            )

        if tb_writer:
            tb_writer.add_scalar("train/loss", train_loss, epoch)
            tb_writer.add_scalar("train/accuracy_top1", train_acc1, epoch)
            tb_writer.add_scalar("val/loss", val_stats["loss"], epoch)
            tb_writer.add_scalar("val/accuracy_top1", val_stats["accuracy_top1"], epoch)
            tb_writer.add_scalar("val/accuracy_top5", val_stats["accuracy_top5"], epoch)
            if macro_f1_seen is not None:
                tb_writer.add_scalar("val/macro_f1_seen", macro_f1_seen, epoch)
            if macro_f1_all is not None:
                tb_writer.add_scalar("val/macro_f1_all", macro_f1_all, epoch)
            if track_diacritic and "diacritic_subset_accuracy" in val_stats:
                tb_writer.add_scalar(
                    "val/diacritic_subset_accuracy",
                    val_stats["diacritic_subset_accuracy"],
                    epoch,
                )

        print(
            f"[EPOCH {epoch:03d}] "
            f"Train: loss={train_loss:.4f} acc@1={train_acc1:.4f} "
            f"| Val: loss={val_stats['loss']:.4f} acc@1={val_stats['accuracy_top1']:.4f} acc@5={val_stats['accuracy_top5']:.4f} "
            f"macroF1_seen={(macro_f1_seen or 0):.4f} macroF1_all={(macro_f1_all or 0):.4f} "
            f"{'(dia=' + format(val_stats.get('diacritic_subset_accuracy', 0.0), '.4f') + ')' if track_diacritic else ''} "
            f"| lr={current_lr:.3e} time={epoch_time:.1f}s",
            flush=True,
        )

        # Early stopping / checkpointing
        monitored_value = None
        if monitor_metric == "val/accuracy_top1":
            monitored_value = val_stats["accuracy_top1"]
        elif monitor_metric == "val/loss":
            monitored_value = -val_stats["loss"]  # invert to treat as max
        else:
            monitored_value = val_stats.get("accuracy_top1", 0.0)

        improved = False
        if monitor_mode == "max":
            if monitored_value > best_score + min_delta:
                improved = True
        else:
            if monitored_value < best_score - min_delta:
                improved = True

        # Automatic unweighted fine-tune switch (executed once)
        if (
            not locals().get("fine_tune_switched", False)
            and unweighted_finetune_epoch >= 0
            and epoch == unweighted_finetune_epoch
        ):
            # Rebuild plain (unweighted) loss with same smoothing
            smoothing = float(loss_cfg.get("label_smoothing", 0.0) or 0.0)
            loss_fn = torch.nn.CrossEntropyLoss(
                label_smoothing=smoothing if smoothing > 0 else 0.0
            )
            token_id_dropout = 0.0
            fine_tune_switched = True
            print(
                f"[FINETUNE] Switched to unweighted loss & token_id_dropout=0.0 at epoch {epoch}",
                flush=True,
            )

        if improved:
            best_score = monitored_value
            no_improve_epochs = 0
            ckpt_name = filename_pattern.format(
                epoch=epoch, val_accuracy_top1=val_stats["accuracy_top1"]
            )
            ckpt_path = ckpt_dir / ckpt_name
            save_checkpoint(
                ckpt_path,
                model,
                optimizer,
                epoch,
                {
                    "val_accuracy_top1": val_stats["accuracy_top1"],
                    "val_loss": val_stats["loss"],
                    "macro_f1": macro_f1,
                },
                scheduler=scheduler,
            )
            # Maintain best.pt symlink (or copy fallback)
            best_link = ckpt_dir / "best.pt"
            try:
                if best_link.exists() or best_link.is_symlink():
                    best_link.unlink()
                best_link.symlink_to(ckpt_path.name)
            except Exception:
                try:
                    shutil.copy2(ckpt_path, best_link)
                except Exception:
                    pass
            top_checkpoints.append((val_stats["accuracy_top1"], ckpt_path))
            top_checkpoints.sort(key=lambda x: x[0], reverse=True)
            if len(top_checkpoints) > save_top_k:
                _, drop_path = top_checkpoints.pop(-1)
                if drop_path.exists():
                    drop_path.unlink(missing_ok=True)
        else:
            no_improve_epochs += 1

        if use_early and patience > 0 and no_improve_epochs >= patience:
            print(
                f"[EARLY STOP] No improvement for {patience} epochs (best={best_score:.5f}).",
                flush=True,
            )
            break

    # Final test evaluation (optional)
    print("[INFO] Evaluating best model (last state) on test split...", flush=True)
    test_stats = run_epoch(
        model,
        test_loader,
        loss_fn,
        optimizer=None,
        device=device,
        mixed_precision=False,
        is_train=False,
        compute_confusion=True,
        num_classes=num_labels,
        track_subset_diacritic=track_diacritic,
    )
    macro_f1_test_seen = None
    macro_f1_test_all = None
    if "confusion" in test_stats:
        conf_test = test_stats["confusion"].float()
        training_seen_mask = (class_counts > 0) if "class_counts" in locals() else None
        macro_f1_test_all = compute_macro_f1(conf_test, None)
        macro_f1_test_seen = compute_macro_f1(conf_test, training_seen_mask)
    print(
        f"[TEST] loss={test_stats['loss']:.4f} acc@1={test_stats['accuracy_top1']:.4f} "
        f"acc@5={test_stats['accuracy_top5']:.4f} macroF1_seen={(macro_f1_test_seen or 0):.4f} "
        f"macroF1_all={(macro_f1_test_all or 0):.4f} "
        f"{'(dia=' + format(test_stats.get('diacritic_subset_accuracy', 0.0), '.4f') + ')' if track_diacritic else ''}",
        flush=True,
    )

    # Save last checkpoint if requested
    if save_last:
        last_path = ckpt_dir / "last.pt"
        save_checkpoint(
            last_path,
            model,
            optimizer,
            locals().get("epoch", 0),
            {
                "val_accuracy_top1": (
                    val_stats["accuracy_top1"]
                    if "val_stats" in locals()
                    else float("nan")
                ),
                "val_loss": (
                    val_stats["loss"] if "val_stats" in locals() else float("nan")
                ),
                "test_accuracy_top1": test_stats.get("accuracy_top1", 0.0),
            },
            scheduler=scheduler,
        )
        print(f"[CHECKPOINT] Saved last checkpoint to {last_path}", flush=True)

    if tb_writer:
        tb_writer.close()

    # Summarize kept top-K
    if top_checkpoints:
        kept_summary = ", ".join(
            f"{p.name}:{acc:.4f}"
            for acc, p in sorted(top_checkpoints, key=lambda x: x[0], reverse=True)
        )
        print(
            f"[CHECKPOINT] Kept top-{min(len(top_checkpoints), save_top_k)}: {kept_summary}",
            flush=True,
        )

    print("[DONE] Phase 2 training complete.", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Phase 2 Transformer Training")
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(os.environ.get("PHASE2_CFG", "configs/phase2.yaml")),
        help="Path to phase2 YAML config.",
    )
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.config.exists():
        print(f"[ERROR] Config not found: {args.config}", file=sys.stderr)
        return 2
    try:
        cfg = Phase2Config.from_yaml(args.config)
        train_phase2(cfg)
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
