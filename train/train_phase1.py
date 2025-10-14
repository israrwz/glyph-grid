#!/usr/bin/env python3
"""
Phase 1 Training Script (Primitive Cell Classification)

Scaffold implementing the core training loop for the Phase 1 CNN described in:
  - NEW_PLAN.md (§6.2, §6.3, §6.4)
  - configs/phase1.yaml

Goals:
  1. Load configuration (YAML)
  2. Load primitive cell dataset:
       - Cells stored under data/processed/cells (e.g., cells.npy or shard_*.npy)
       - Assignments file maps cell_id -> primitive_id (Parquet or JSONL)
       - Optional split files listing cell_ids for train / val / test
  3. Build DataLoader(s)
  4. Instantiate baseline CNN (models/phase1_cnn.py)
  5. Train with:
       - CrossEntropy loss (+ optional label smoothing)
       - AdamW optimizer
       - Cosine LR schedule (warmup) or OneCycle fallback
       - Mixed precision (optional)
       - Early stopping (optional)
  6. Track metrics: accuracy@1, accuracy@5 (minimal scaffold)
  7. Checkpoint best model(s)
  8. (Optional) Export ONNX / TorchScript if enabled

Design Principles:
  - Keep external dependencies minimal (torch, pyyaml, pandas optional).
  - Fail loud with actionable messages when expected artifacts absent.
  - Clear separation between configuration parsing, data, model, and training loop.
  - Extensible hooks (TODO markers) for future metrics (confusion matrix, bucket accuracy).

NOTE:
  This is a scaffold: certain advanced plan features (frequency bucket accuracy,
  confusion matrix sparse logging, TensorBoard / W&B integration) are sketched or
  stubbed with TODOs for incremental refinement.

Usage:
  python -m train.train_phase1 --config configs/phase1.yaml
  (or) python train/train_phase1.py --config configs/phase1.yaml

Environment Expectations:
  - PyTorch installed
  - YAML config present
  - data/processed/cells/{cells.npy|shard_*.npy} exists
  - assignments file produced by primitives pipeline

License: Follow project’s root license.

"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Third-party imports with guarded fallbacks
# ---------------------------------------------------------------------------

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise RuntimeError("Missing dependency: pyyaml (pip install pyyaml)") from e

try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
except ImportError as e:  # pragma: no cover
    raise RuntimeError("PyTorch is required (pip install torch).") from e

try:
    import pandas as pd  # Optional (Parquet / CSV)
except ImportError:
    pd = None  # type: ignore

# Local imports
try:
    from models.phase1_cnn import build_phase1_model
    from data.primitives import CellSource
except ImportError as e:
    raise RuntimeError(
        "Local imports failed. Ensure PYTHONPATH includes project root."
    ) from e


# ---------------------------------------------------------------------------
# Configuration Parsing Helpers
# ---------------------------------------------------------------------------


@dataclass
class EarlyStopConfig:
    patience: int
    min_delta: float
    monitor: str
    mode: str


@dataclass
class TrainConfig:
    experiment_name: str
    seed: int
    deterministic: bool
    data: Dict[str, Any]
    vocabulary: Dict[str, Any]
    model: Dict[str, Any]
    optim: Dict[str, Any]
    scheduler: Dict[str, Any]
    training: Dict[str, Any]
    loss: Dict[str, Any]
    metrics: Dict[str, Any]
    checkpoint: Dict[str, Any]
    logging: Dict[str, Any]
    eval_cfg: Dict[str, Any]
    export: Dict[str, Any]
    debug: Dict[str, Any]
    sanity: Dict[str, Any]
    misclassified_render: Dict[str, Any]

    @staticmethod
    def from_yaml(path: str | Path) -> "TrainConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        # Provide fallbacks for sections (some YAML keys might be absent if user trims config)
        def section(key: str, default: Dict[str, Any] | None = None):
            return raw.get(key, default or {})

        return TrainConfig(
            experiment_name=raw.get("experiment_name", "phase1_experiment"),
            seed=raw.get("seed", 42),
            deterministic=raw.get("deterministic", True),
            data=section("data"),
            vocabulary=section("vocabulary"),
            model=section("model"),
            optim=section("optim"),
            scheduler=section("scheduler"),
            training=section("training"),
            loss=section("loss"),
            metrics=section("metrics"),
            checkpoint=section("checkpoint"),
            logging=section("logging"),
            eval_cfg=section("eval"),
            export=section("export"),
            debug=section("debug"),
            sanity=section("sanity"),
            misclassified_render=section("misclassified_render"),
        )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PrimitiveCellDataset(Dataset):
    """
    Dataset returning (cell_tensor, primitive_id).

    Data Sources:
      - cells_dir: directory with cells.npy or shard_*.npy (identical format to CellSource)
      - assignments_file: Parquet (.parquet) OR JSONL (.jsonl) OR CSV (.csv)
         Must provide at least: cell_id, primitive_id
      - Optional split_file: text file listing selected cell_ids (one per line)

    Cell Loading:
      - Uses CellSource to stream cells on demand (kept open as memory map).
      - Internal index maps dataset index -> cell_id -> primitive_id
    """

    def __init__(
        self,
        cells_dir: Path,
        assignments_file: Path,
        split_file: Optional[Path] = None,
        transform=None,
        empty_class_id: int = 0,
        empty_sampling_ratio: float = 0.1,
        return_cell_id: bool = False,
    ):
        super().__init__()
        self.cells_dir = cells_dir
        self.assignments_file = assignments_file
        self.transform = transform
        self.empty_class_id = empty_class_id
        self.empty_sampling_ratio = float(empty_sampling_ratio)
        self.return_cell_id = bool(return_cell_id)
        if not (0.0 < self.empty_sampling_ratio <= 1.0):
            raise ValueError("empty_sampling_ratio must be in (0,1].")

        if not cells_dir.exists():
            raise FileNotFoundError(f"cells_dir missing: {cells_dir}")

        self._cell_source = CellSource(cells_dir)
        # Shard / array indexing structures for fast random access:
        #  - consolidated_npy: keep a single mmap reference
        #  - sharded_npy: build a list of (start, end, path) and cache mmap objects on demand
        self._consolidated_array = None
        self._shard_index = None  # list[dict]: {"start":int,"end":int,"path":Path,"array":np.memmap or ndarray}

        # Load assignments
        self._assign_map: Dict[int, int] = self._load_assignments(assignments_file)

        # Restrict to split
        if split_file and split_file.exists():
            chosen_ids = self._load_split_ids(split_file)
            # Filter to intersection
            filtered = {
                cid: self._assign_map[cid]
                for cid in chosen_ids
                if cid in self._assign_map
            }
            self._assign_map = filtered

        # Build deterministic ordered index (apply empty downsampling if enabled)
        base_ids: List[int] = sorted(self._assign_map.keys())
        if self.empty_sampling_ratio < 1.0:
            import random as _random

            kept: List[int] = []
            for cid in base_ids:
                pid = self._assign_map[cid]
                if pid == self.empty_class_id:
                    if _random.random() < self.empty_sampling_ratio:
                        kept.append(cid)
                else:
                    kept.append(cid)
            self._cell_ids = kept
        else:
            self._cell_ids = base_ids

        if len(self._cell_ids) == 0:
            raise RuntimeError(
                "PrimitiveCellDataset empty after loading assignments/split/downsampling."
            )

        # Quick sanity check
        first_id = self._cell_ids[0]
        if first_id not in self._assign_map:
            raise RuntimeError("Internal indexing error (missing first cell id).")

    # ---------------------------------------------
    @staticmethod
    def _load_split_ids(path: Path) -> List[int]:
        ids: List[int] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        ids.append(int(line))
                    except ValueError:
                        continue
        return ids

    # ---------------------------------------------
    @staticmethod
    def _load_assignments(path: Path) -> Dict[int, int]:
        if not path.exists():
            raise FileNotFoundError(f"Assignments file not found: {path}")

        ext = path.suffix.lower()
        mapping: Dict[int, int] = {}
        if ext == ".parquet":
            if pd is None:
                raise RuntimeError("pandas required to read parquet assignments.")
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
                        rec = yaml.safe_load(line)  # also works for JSON line
                        mapping[int(rec["cell_id"])] = int(rec["primitive_id"])
                    except Exception:
                        continue
        elif ext in (".csv",):
            with open(path, "r", newline="", encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                for rec in rdr:
                    try:
                        mapping[int(rec["cell_id"])] = int(rec["primitive_id"])
                    except Exception:
                        continue
        else:
            raise ValueError(
                f"Unsupported assignments file extension: {ext} (expected parquet, jsonl, csv)"
            )
        if not mapping:
            raise RuntimeError("No cell assignments loaded (empty mapping).")
        return mapping

    # ---------------------------------------------
    def __len__(self) -> int:
        return len(self._cell_ids)

    # ---------------------------------------------
    def __getitem__(self, index: int):
        cid = self._cell_ids[index]
        label = self._assign_map[cid]

        # Retrieve cell bitmap from cell source
        # We iterate sources in order; direct random access requires small index->array mapping.
        # For efficiency, we build a simple cache dictionary (optional future optimization).
        cell = self._load_cell_by_id(cid)

        # Convert to torch tensor (1,8,8) float32
        # cell is NP array (8,8) uint8/bool 0 or 255; unify to float 0..1
        import numpy as np

        if cell.dtype != np.uint8 and cell.dtype != np.bool_:
            cell = cell.astype(np.uint8)
        if cell.ndim != 2 or cell.shape != (8, 8):
            raise ValueError(f"Unexpected cell shape for id {cid}: {cell.shape}")

        tensor = torch.from_numpy((cell > 0).astype("float32")).unsqueeze(0)  # (1,8,8)

        if self.transform:
            tensor = self.transform(tensor)

        if self.return_cell_id:
            return tensor, label, cid
        return tensor, label

    # ---------------------------------------------
    def _build_shard_index_if_needed(self):
        """
        Build in-memory index for fast O(log S) shard resolution.

        For 'consolidated_npy':
            - Load mmap once, store in self._consolidated_array (no per-call np.load).
        For 'sharded_npy':
            - Build list of dicts:
                {
                  "start": global_start_cell_id,
                  "end": exclusive_end_cell_id,
                  "path": Path,
                  "array": None or mmap (lazy)
                }
        """
        if self._shard_index is not None or self._consolidated_array is not None:
            return
        fmt = self._cell_source._format
        root = self._cell_source.root
        import numpy as np

        if fmt == "consolidated_npy":
            self._consolidated_array = np.load(root / "cells.npy", mmap_mode="r")
        elif fmt == "sharded_npy":
            self._shard_index = []
            cumulative = 0
            for sp in sorted(root.glob("shard_*.npy")):
                arr = np.load(sp, mmap_mode="r")
                length = len(arr)
                self._shard_index.append(
                    {
                        "start": cumulative,
                        "end": cumulative + length,
                        "path": sp,
                        "array": arr,  # keep mmap reference
                    }
                )
                cumulative += length
        else:
            raise NotImplementedError("LMDB / other formats not implemented here.")

    def _load_cell_by_id(self, cid: int):
        """
        Fast random access using pre-built shard index or consolidated mmap.
        """
        self._build_shard_index_if_needed()
        fmt = self._cell_source._format
        import numpy as np

        if fmt == "consolidated_npy":
            arr = self._consolidated_array
            if cid >= len(arr):
                raise KeyError(f"Cell id {cid} out of range (N={len(arr)})")
            cell = arr[cid]
            return cell if cell.ndim == 2 else cell[0]

        if fmt == "sharded_npy":
            # Binary search over shard index (list is small; linear acceptable but we do bisect)
            index = self._shard_index
            lo, hi = 0, len(index) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                entry = index[mid]
                if cid < entry["start"]:
                    hi = mid - 1
                elif cid >= entry["end"]:
                    lo = mid + 1
                else:
                    local_offset = cid - entry["start"]
                    arr = entry["array"]
                    cell = arr[local_offset]
                    return cell if cell.ndim == 2 else cell[0]
            raise KeyError(f"Cell id {cid} beyond shard ranges.")

        raise NotImplementedError("LMDB / other formats not implemented here.")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@torch.no_grad()
def accuracy_topk(
    logits: torch.Tensor, targets: torch.Tensor, topk=(1,)
) -> List[float]:
    maxk = max(topk)
    batch_size = targets.size(0)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()  # shape (maxk, B)
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    out: List[float] = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum().item()
        out.append(correct_k / batch_size)
    return out


# ---------------------------------------------------------------------------
# Loss (with optional label smoothing)
# ---------------------------------------------------------------------------


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps: float = 0.0):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        if self.eps == 0.0:
            return nn.functional.cross_entropy(logits, target)
        n_classes = logits.size(-1)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.eps / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.eps)
        return (-true_dist * log_probs).sum(dim=1).mean()


def build_loss(cfg: Dict[str, Any]) -> nn.Module:
    smoothing = float(cfg.get("label_smoothing", 0.0) or 0.0)
    return LabelSmoothingCrossEntropy(eps=smoothing)


# ---------------------------------------------------------------------------
# Optimizer & Scheduler
# ---------------------------------------------------------------------------


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    name = cfg.get("name", "adamw").lower()
    lr = float(cfg.get("lr", 1e-3))
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
    strat = sched_cfg.get("strategy", "cosine").lower()
    if strat == "cosine":
        warmup_epochs = int(sched_cfg.get("warmup_epochs", 0))
        min_lr_scale = float(sched_cfg.get("min_lr_scale", 0.1))
        base_lrs = [group["lr"] for group in optimizer.param_groups]

        def lr_lambda(epoch: int):
            if epoch < warmup_epochs and warmup_epochs > 0:
                return (epoch + 1) / warmup_epochs
            # Cosine over remaining
            progress = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr_scale + (1 - min_lr_scale) * cosine_factor

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    elif strat == "onecycle":
        # OneCycle typically needs steps_per_epoch; we adapt with total epochs only (simplified)
        max_lr = float(sched_cfg.get("max_lr", optimizer.param_groups[0]["lr"]))
        div_factor = float(sched_cfg.get("div_factor", 25))
        pct_start = float(sched_cfg.get("pct_start", 0.3))
        # For simplicity, we will create per-epoch lambda schedule (not per-step)
        initial_lr = max_lr / div_factor
        for pg in optimizer.param_groups:
            pg["lr"] = initial_lr

        def lr_lambda(epoch: int):
            pct = epoch / max(1, total_epochs - 1)
            if pct < pct_start:
                return 1 + (max_lr / initial_lr - 1) * (pct / pct_start)
            else:
                dec_pct = (pct - pct_start) / max(1e-8, (1 - pct_start))
                return (max_lr / initial_lr) * (1 - dec_pct)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    else:
        return None


# ---------------------------------------------------------------------------
# Checkpoint Utils
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(state, path)


# ---------------------------------------------------------------------------
# Training / Validation Loop
# ---------------------------------------------------------------------------


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    mixed_precision: bool = False,
    grad_clip: Optional[float] = None,
    is_train: bool = True,
    *,
    empty_class_id: int = 0,
    freq_bucket_boundaries: Sequence[int] | None = None,
) -> Dict[str, float]:
    """
    Extended epoch runner:
      - Tracks top1 / top5
      - Tracks non-empty top1 (targets != empty_class_id)
      - Tracks per-frequency-bucket accuracy (buckets defined by cumulative support thresholds
        in freq_bucket_boundaries, e.g. [10,100,1000,10000])
      - Collects a small sample of misclassified examples (returned in metrics under 'misclassified')
        Format: list of dicts { 'cell_id': optional, 'pred': int, 'target': int }
        (cell_id omitted here because DataLoader only yields tensors; could be added by
         wrapping dataset to return ids.)
    """
    model.train(is_train)
    scaler = torch.cuda.amp.GradScaler(
        enabled=mixed_precision and is_train and device.type == "cuda"
    )

    total_loss = 0.0
    total_samples = 0
    total_correct1 = 0
    total_correct5 = 0
    total_nonempty = 0
    total_nonempty_correct1 = 0

    # Frequency bucket accumulators: list of dicts with 'correct'/'total'
    bucket_defs: List[Tuple[str, int, int]] = []
    if freq_bucket_boundaries:
        # Build bucket (lo inclusive, hi exclusive) with labels
        prev = 0
        for b in freq_bucket_boundaries:
            bucket_defs.append((f"[{prev},{b})", prev, b))
            prev = b
        bucket_defs.append((f"[{prev},inf)", prev, 10**12))
        bucket_stats = {
            label: {"correct": 0, "total": 0} for (label, _, _) in bucket_defs
        }
    else:
        bucket_stats = {}

    # Misclassified sampler (first N unique)
    max_misclassified_keep = 50
    # Store dicts with 'pred','target','cell' (the 8x8 tensor CPU numpy) for rendering
    misclassified: List[Dict[str, Any]] = []

    for batch_idx, batch in enumerate(loader):
        # Support optional (x,y,cell_ids) when dataset was created with return_cell_id=True
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x, y, cell_ids = batch
        else:
            x, y = batch  # type: ignore
            cell_ids = None
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y, device=device, dtype=torch.long)
        if cell_ids is not None and hasattr(cell_ids, "to"):
            cell_ids = cell_ids.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=mixed_precision and device.type == "cuda"):
            logits = model(x)
            loss = loss_fn(logits, y)

        if is_train:
            optimizer.zero_grad(set_to_none=True)  # type: ignore
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip:
                    scaler.unscale_(optimizer)  # type: ignore
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)  # type: ignore
                scaler.update()
            else:
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()  # type: ignore

        with torch.no_grad():
            bsz = x.size(0)
            total_loss += loss.item() * bsz
            total_samples += bsz

            acc1, acc5 = accuracy_topk(logits, y, topk=(1, 5))
            total_correct1 += acc1 * bsz
            total_correct5 += acc5 * bsz

            # Non-empty accuracy
            mask_nonempty = y != empty_class_id
            nonempty_count = mask_nonempty.sum().item()
            if nonempty_count > 0:
                _, preds = torch.max(logits, dim=1)
                nonempty_correct = (
                    (preds[mask_nonempty] == y[mask_nonempty]).sum().item()
                )
                total_nonempty += nonempty_count
                total_nonempty_correct1 += nonempty_correct

            # Misclassified sampling (capture cell patch)
            if len(misclassified) < max_misclassified_keep:
                _, preds_full = torch.max(logits, dim=1)
                mism = (preds_full != y).nonzero(as_tuple=False).view(-1)
                for midx in mism:
                    if len(misclassified) >= max_misclassified_keep:
                        break
                    cell_patch = x[midx].detach().cpu().numpy()  # (1,8,8) float
                    rec: Dict[str, Any] = {
                        "pred": int(preds_full[midx].item()),
                        "target": int(y[midx].item()),
                        "cell": cell_patch.squeeze(0),  # (8,8)
                    }
                    if cell_ids is not None:
                        try:
                            rec["cell_id"] = int(cell_ids[midx].item())
                        except Exception:
                            pass
                    misclassified.append(rec)

            # Frequency bucket stats (requires dataset frequency information embedded via loader.dataset)
            if bucket_stats:
                # Expect dataset to have attribute primitive_frequency (dict: pid -> count)
                freq_map = getattr(loader.dataset, "primitive_frequency", None)
                if freq_map is not None:
                    _, preds_full = torch.max(logits, dim=1)
                    for idx in range(bsz):
                        tgt = int(y[idx].item())
                        freq = int(freq_map.get(tgt, 0))
                        # Determine bucket
                        for label, lo, hi in bucket_defs:
                            if lo <= freq < hi:
                                bucket_stats[label]["total"] += 1
                                if preds_full[idx].item() == tgt:
                                    bucket_stats[label]["correct"] += 1
                                break

    metrics = {
        "loss": total_loss / max(1, total_samples),
        "accuracy_top1": total_correct1 / max(1, total_samples),
        "accuracy_top5": total_correct5 / max(1, total_samples),
    }
    if total_nonempty > 0:
        metrics["nonempty_accuracy_top1"] = total_nonempty_correct1 / total_nonempty
    if bucket_stats:
        for label, stat in bucket_stats.items():
            if stat["total"] > 0:
                metrics[f"bucket_acc{label}"] = stat["correct"] / stat["total"]
    metrics["misclassified"] = misclassified
    return metrics


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def train_phase1(cfg: TrainConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(cfg.seed, cfg.deterministic)

    # Dataset paths & validation
    data_cfg = cfg.data
    cells_dir = Path(data_cfg["cells_dir"])
    assignments_file = Path(data_cfg["assignments_file"])
    empty_class_id = int(data_cfg.get("empty_class_id", 0))

    train_split = (
        Path(data_cfg.get("train_split", "")) if data_cfg.get("train_split") else None
    )
    val_split = (
        Path(data_cfg.get("val_split", "")) if data_cfg.get("val_split") else None
    )

    # Resolve optional test split (may be absent)
    test_split = (
        Path(data_cfg.get("test_split", "")) if data_cfg.get("test_split") else None
    )

    # Training dataset (apply empty cell downsampling)
    train_ds = PrimitiveCellDataset(
        cells_dir=cells_dir,
        assignments_file=assignments_file,
        split_file=train_split,
        empty_class_id=empty_class_id,
        empty_sampling_ratio=float(cfg.data.get("empty_sampling_ratio", 0.1)),
        return_cell_id=True,
    )

    # Validation dataset (keep all empties for unbiased metrics)
    val_ds = PrimitiveCellDataset(
        cells_dir=cells_dir,
        assignments_file=assignments_file,
        split_file=val_split,
        empty_class_id=empty_class_id,
        empty_sampling_ratio=1.0,
        return_cell_id=True,
    )

    # Test dataset (optional). If no test split provided, reuse val_ds reference.
    test_ds = (
        PrimitiveCellDataset(
            cells_dir=cells_dir,
            assignments_file=assignments_file,
            split_file=test_split,
            empty_class_id=empty_class_id,
            empty_sampling_ratio=1.0,
            return_cell_id=True,
        )
        if test_split
        else val_ds
    )

    print(
        f"[DATA] empty_sampling_ratio(train)={cfg.data.get('empty_sampling_ratio', 0.1)} | "
        f"train_cells={len(train_ds)} val_cells={len(val_ds)} "
        f"test_cells={len(test_ds)}",
        flush=True,
    )

    loader_cfg = cfg.data  # reuse fields
    batch_size = (
        int(cfg.loader_get("batch_size", 1024))
        if hasattr(cfg, "loader_get")
        else int(cfg.training.get("batch_size", cfg.model.get("batch_size", 1024)))
    )
    # Prefer loader section if present
    loader_section = (
        getattr(cfg, "loader", None)
        or cfg.__dict__.get("loader", None)
        or cfg.__dict__.get("data", {})
    )
    batch_size = int(loader_section.get("batch_size", 1024))
    num_workers = int(loader_section.get("num_workers", 4))
    pin_memory = bool(loader_section.get("pin_memory", True))
    shuffle = bool(loader_section.get("shuffle", True))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # Model
    model = build_phase1_model(cfg.model)
    model.to(device)

    # Loss
    loss_fn = build_loss(cfg.loss)

    # Optim + Scheduler
    optimizer = build_optimizer(model, cfg.optim)
    epochs = int(cfg.training.get("epochs", 60))
    scheduler = build_scheduler(optimizer, cfg.scheduler, total_epochs=epochs)

    # Early stopping
    es_cfg = cfg.training.get("early_stop", {})
    patience = int(es_cfg.get("patience", 8))
    min_delta = float(es_cfg.get("min_delta", 0.0))
    monitor = es_cfg.get("monitor", "val/accuracy_top1")
    mode = es_cfg.get("mode", "max").lower()
    best_metric = -float("inf") if mode == "max" else float("inf")
    epochs_no_improve = 0

    # Checkpoint cfg
    ckpt_dir = Path(cfg.checkpoint.get("dir", "checkpoints/phase1"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_top_k = int(cfg.checkpoint.get("save_top_k", 3))
    top_checkpoints: List[Tuple[float, Path]] = []

    # Logging
    log_interval = int(cfg.training.get("log_interval_batches", 50))
    metrics_csv = Path(cfg.logging.get("csv_log_file", "logs/phase1_metrics.csv"))
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    if not metrics_csv.exists():
        with metrics_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "phase",
                    "loss",
                    "accuracy_top1",
                    "accuracy_top5",
                    "nonempty_accuracy_top1",
                    "lr",
                    "time_sec",
                ]
            )

    mixed_precision = bool(cfg.training.get("mixed_precision", True))
    grad_clip = cfg.training.get("gradient_clip_norm", None)
    if grad_clip is not None:
        grad_clip = float(grad_clip)

    print(
        f"[INFO] Device={device} | Train size={len(train_ds)} | Val size={len(val_ds)} | "
        f"Params={sum(p.numel() for p in model.parameters()) / 1e3:.1f}K",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        start_epoch_time = time.time()
        train_stats = run_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            mixed_precision=mixed_precision,
            grad_clip=grad_clip,
            is_train=True,
        )
        if scheduler:
            scheduler.step()

        val_stats = run_epoch(
            model,
            val_loader,
            loss_fn,
            optimizer=None,
            device=device,
            mixed_precision=False,
            grad_clip=None,
            is_train=False,
        )
        # Render misclassified samples (validation) if enabled and available
        render_cfg = getattr(cfg, "misclassified_render", {}) or {}
        if render_cfg.get("enabled", False):
            try:
                from PIL import Image, ImageDraw, ImageFont

                out_dir = (
                    Path(render_cfg.get("output_dir", "logs/phase1_misclassified"))
                    / f"epoch_{epoch:03d}"
                )
                out_dir.mkdir(parents=True, exist_ok=True)
                layout = render_cfg.get("layout", "horizontal")
                scale = int(render_cfg.get("scale", 16))
                gap = int(render_cfg.get("gap_px", 8))
                bg_val = int(render_cfg.get("bg_value", 0))
                annotate = bool(render_cfg.get("annotate", True))
                fg_val = int(render_cfg.get("fg_value", 255))
                font_size = int(render_cfg.get("font_size", 14))
                # Attempt font
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None
                max_samples = int(render_cfg.get("max_samples_per_epoch", 50))
                # Load centroids once (expects with-empty at index 0). Fallback to None if missing.
                import json
                import numpy as np

                centroids = None
                centroid_path = (
                    Path(getattr(cfg, "vocabulary", {}).get("centroid_file", ""))
                    if hasattr(cfg, "vocabulary")
                    else None
                )
                if centroid_path and centroid_path.exists():
                    try:
                        centroids = np.load(centroid_path)
                    except Exception as _e:
                        centroids = None
                samples = val_stats.get("misclassified", [])[:max_samples]
                mis_index = []
                for idx, rec in enumerate(samples):
                    cell = rec.get("cell")
                    if cell is None:
                        continue
                    pred = rec["pred"]
                    target = rec["target"]
                    cell_id = rec.get("cell_id", None)
                    # Prepare cell array (binary 0/255)
                    arr_cell = (
                        (cell * 255.0).astype("uint8")
                        if cell.max() <= 1.0
                        else cell.astype("uint8")
                    )

                    # Derive centroid panels (reshape 8x8). Handle empty or missing gracefully.
                    def get_centroid(pid: int):
                        if centroids is None:
                            return np.zeros_like(arr_cell)
                        if pid < 0 or pid >= len(centroids):
                            return np.zeros_like(arr_cell)
                        vec = centroids[pid]
                        if vec.shape[0] != 64:
                            return np.zeros_like(arr_cell)
                        return (vec.reshape(8, 8) * 255.0).clip(0, 255).astype("uint8")

                    arr_pred_centroid = get_centroid(pred)
                    arr_target_centroid = get_centroid(target)
                    # Binarize centroids for XOR visualization (threshold 128)
                    pred_bin = (arr_pred_centroid >= 128).astype("uint8")
                    target_bin = (arr_target_centroid >= 128).astype("uint8")
                    cell_bin = (arr_cell >= 128).astype("uint8")
                    xor_pred = ((cell_bin ^ pred_bin) * 255).astype("uint8")
                    xor_target = ((cell_bin ^ target_bin) * 255).astype("uint8")
                    # Distances (L2) between cell (0/1) vector and centroid (normalized 0..1)
                    if centroids is not None and 0 <= pred < len(centroids):
                        cell_vec = cell_bin.reshape(-1).astype("float32")
                        pred_vec = (centroids[pred].reshape(-1)).astype("float32")
                        target_vec = (
                            centroids[target].reshape(-1).astype("float32")
                            if 0 <= target < len(centroids)
                            else np.zeros_like(cell_vec)
                        )
                        dist_pred = float(np.linalg.norm(cell_vec - pred_vec))
                        dist_target = float(np.linalg.norm(cell_vec - target_vec))
                    else:
                        dist_pred = float("nan")
                        dist_target = float("nan")
                    # Compose 5 panels horizontally: cell | pred_centroid | target_centroid | XOR(cell,pred) | XOR(cell,target)
                    panels = [
                        ("cell", arr_cell),
                        ("pred", arr_pred_centroid),
                        ("target", arr_target_centroid),
                        ("xor_pred", xor_pred),
                        ("xor_target", xor_target),
                    ]
                    h, w = arr_cell.shape
                    panel_w = w * scale
                    panel_h = h * scale
                    gap_count = len(panels) - 1
                    W = panel_w * len(panels) + gap * gap_count
                    H = panel_h
                    canvas = Image.new("RGBA", (W, H), color=(0, 0, 0, 0))
                    # Paste panels
                    for p_i, (_label, arr) in enumerate(panels):
                        img_mode = "L"
                        # XOR panels: optionally colorize (xor_pred -> red, xor_target -> blue)
                        if _label == "xor_pred":
                            # map grayscale to RGBA red
                            rgba = np.zeros((h, w, 4), dtype="uint8")
                            rgba[..., 0] = arr  # Red
                            rgba[..., 3] = (arr > 0) * 127  # alpha 50%
                            panel_img = Image.fromarray(rgba, mode="RGBA")
                        elif _label == "xor_target":
                            rgba = np.zeros((h, w, 4), dtype="uint8")
                            rgba[..., 2] = arr  # Blue
                            rgba[..., 3] = (arr > 0) * 127
                            panel_img = Image.fromarray(rgba, mode="RGBA")
                        else:
                            panel_img = Image.fromarray(arr, mode=img_mode).convert(
                                "RGBA"
                            )
                        panel_img = panel_img.resize((panel_w, panel_h), Image.NEAREST)
                        x_off = p_i * (panel_w + gap)
                        canvas.paste(panel_img, (x_off, 0), panel_img)
                    if annotate:
                        draw = ImageDraw.Draw(canvas)
                        text = f"pred={pred} target={target} dp={dist_pred:.2f} dt={dist_target:.2f}"
                        draw.text((4, 4), text, fill=(255, 255, 255, 255), font=font)
                        if cell_id is not None:
                            draw.text(
                                (4, 4 + font_size + 2),
                                f"id={cell_id}",
                                fill=(200, 200, 200, 255),
                                font=font,
                            )
                    out_path = (
                        out_dir / f"mis_{idx:02d}_cid{cell_id}_p{pred}_t{target}.png"
                    )
                    canvas.save(out_path)
                    mis_index.append(
                        {
                            "cell_id": int(cell_id) if cell_id is not None else None,
                            "pred": pred,
                            "target": target,
                            "dist_pred": dist_pred,
                            "dist_target": dist_target,
                            "image": out_path.as_posix(),
                        }
                    )
                # Write per-epoch JSON index
                try:
                    with (out_dir / "misclassified_index.json").open(
                        "w", encoding="utf-8"
                    ) as jf:
                        json.dump(mis_index, jf, ensure_ascii=False, indent=2)
                except Exception as _e:
                    print(
                        f"[warn] Failed writing misclassified_index.json: {_e}",
                        file=sys.stderr,
                    )
            except Exception as e:
                print(
                    f"[warn] Failed rendering misclassified samples: {e}",
                    file=sys.stderr,
                )

        epoch_time = time.time() - start_epoch_time
        current_lr = optimizer.param_groups[0]["lr"]

        # Write metrics
        with metrics_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    "train",
                    f"{train_stats['loss']:.6f}",
                    f"{train_stats['accuracy_top1']:.6f}",
                    f"{train_stats['accuracy_top5']:.6f}",
                    f"{train_stats.get('nonempty_accuracy_top1', 0.0):.6f}",
                    f"{current_lr:.6e}",
                    f"{epoch_time:.2f}",
                ]
            )
            writer.writerow(
                [
                    epoch,
                    "val",
                    f"{val_stats['loss']:.6f}",
                    f"{val_stats['accuracy_top1']:.6f}",
                    f"{val_stats['accuracy_top5']:.6f}",
                    f"{val_stats.get('nonempty_accuracy_top1', 0.0):.6f}",
                    f"{current_lr:.6e}",
                    f"{epoch_time:.2f}",
                ]
            )

        if epoch % 1 == 0:
            print(
                f"[EPOCH {epoch:03d}] "
                f"Train: loss={train_stats['loss']:.4f} acc@1={train_stats['accuracy_top1']:.4f} "
                f"(nonempty={train_stats.get('nonempty_accuracy_top1', 0.0):.4f}) "
                f"| Val: loss={val_stats['loss']:.4f} acc@1={val_stats['accuracy_top1']:.4f} "
                f"(nonempty={val_stats.get('nonempty_accuracy_top1', 0.0):.4f}) "
                f"acc@5={val_stats['accuracy_top5']:.4f} | lr={current_lr:.3e} "
                f"time={epoch_time:.1f}s",
                flush=True,
            )

        # Monitor metric
        monitored_value = None
        if monitor == "val/accuracy_top1":
            monitored_value = val_stats["accuracy_top1"]
        elif monitor == "val/loss":
            monitored_value = -val_stats["loss"]  # invert for max logic
        else:
            # Default fallback: val acc@1
            monitored_value = val_stats["accuracy_top1"]

        improved = False
        if mode == "max":
            if monitored_value > best_metric + min_delta:
                improved = True
        else:  # mode == "min"
            if monitored_value < best_metric - min_delta:
                improved = True

        if improved:
            best_metric = monitored_value
            epochs_no_improve = 0
            # Save checkpoint
            metric_tag = f"{val_stats['accuracy_top1']:.4f}"
            ckpt_name = cfg.checkpoint.get(
                "filename_pattern", "epoch{epoch:02d}-val{val_accuracy_top1:.4f}.pt"
            ).format(epoch=epoch, val_accuracy_top1=val_stats["accuracy_top1"])
            ckpt_path = ckpt_dir / ckpt_name
            save_checkpoint(
                ckpt_path,
                model,
                optimizer,
                epoch,
                {
                    "val_accuracy_top1": val_stats["accuracy_top1"],
                    "val_loss": val_stats["loss"],
                },
            )
            top_checkpoints.append((val_stats["accuracy_top1"], ckpt_path))
            # Keep only top_k
            top_checkpoints.sort(key=lambda x: x[0], reverse=True)
            if len(top_checkpoints) > save_top_k:
                _, drop_path = top_checkpoints.pop(-1)
                if drop_path.exists():
                    drop_path.unlink(missing_ok=True)
        else:
            epochs_no_improve += 1

        if patience > 0 and epochs_no_improve >= patience:
            print(
                f"[EARLY STOP] No improvement in {patience} epochs (best metric={best_metric:.5f}).",
                flush=True,
            )
            break

    # Final export (optional)
    export_cfg = cfg.export
    if export_cfg.get("onnx", {}).get("enabled", False):
        try:
            onnx_path = Path(export_cfg["onnx"]["path"])
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            dummy = torch.zeros(1, cfg.model.get("in_channels", 1), 8, 8, device=device)
            torch.onnx.export(
                model,
                dummy,
                onnx_path.as_posix(),
                input_names=["cell"],
                output_names=["logits"],
                opset_version=export_cfg["onnx"].get("opset", 18),
            )
            print(f"[EXPORT] ONNX saved to {onnx_path}")
        except Exception as e:
            print(f"[EXPORT][WARN] ONNX export failed: {e}")

    if export_cfg.get("jit", {}).get("enabled", False):
        try:
            jit_path = Path(export_cfg["jit"]["path"])
            jit_path.parent.mkdir(parents=True, exist_ok=True)
            scripted = torch.jit.script(model.cpu())
            scripted.save(jit_path.as_posix())
            print(f"[EXPORT] TorchScript saved to {jit_path}")
            model.to(device)
        except Exception as e:
            print(f"[EXPORT][WARN] TorchScript export failed: {e}")

    print("[DONE] Phase 1 training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 Primitive Training")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase1.yaml"),
        help="Path to phase1 YAML config.",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.config.exists():
        print(f"[ERROR] Config file not found: {args.config}", file=sys.stderr)
        return 2
    cfg = TrainConfig.from_yaml(args.config)
    try:
        train_phase1(cfg)
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
