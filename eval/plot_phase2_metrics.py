#!/usr/bin/env python3
"""
plot_phase2_metrics.py
======================

Utility script to visualize Phase 2 training metrics, including the dual macro F1
variants (macro_f1_seen vs macro_f1_all) recently added to the training loop.

It parses the metrics CSV emitted by `train/train_phase2.py` and produces
multi-panel plots for:
  1. Loss (train vs val)
  2. Accuracy (top1 train/val, top5 val)
  3. Macro F1 (seen vs all)
  4. Diacritic subset accuracy (train vs val, if available)
  5. (Optional) Accuracy gap (train_top1 - val_top1)

The script is dependency-light (only standard library + optional matplotlib).
If matplotlib is not available, it will exit with an informative message.

CSV Expectations (columns):
  epoch, phase (train|val), loss, accuracy_top1, accuracy_top5,
  macro_f1_seen, macro_f1_all, diacritic_subset_accuracy, lr, time_sec

Earlier runs might not have the dual macro F1 columns; the script handles blanks gracefully.

Usage:
  python -m eval.plot_phase2_metrics \
      --metrics-csv logs/phase2/metrics.csv \
      --out-dir plots/phase2 \
      --format png \
      --rolling 2

Options:
  --rolling N        Apply a simple centered moving average (window N) to smooth curves
  --show             Display plots interactively (in addition to saving)
  --separate         Save individual PNG/SVG files per panel instead of a single grid
  --no-gap           Disable accuracy gap panel

Exit codes:
  0 success
  2 metrics file not found
  3 matplotlib missing
  4 generic failure

License: Follows project root license.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    train_acc1: Optional[float] = None
    val_acc1: Optional[float] = None
    train_acc5: Optional[float] = None  # seldom used
    val_acc5: Optional[float] = None
    macro_f1_seen: Optional[float] = None
    macro_f1_all: Optional[float] = None
    train_dia_acc: Optional[float] = None
    val_dia_acc: Optional[float] = None
    lr: Optional[float] = None
    time_sec: Optional[float] = (
        None  # epoch wall time (duplicated train/val rows; we keep last)
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Plot Phase 2 training metrics with dual macro F1."
    )
    ap.add_argument(
        "--metrics-csv",
        required=True,
        type=Path,
        help="Path to metrics.csv produced during training.",
    )
    ap.add_argument(
        "--out-dir", required=True, type=Path, help="Output directory for plots."
    )
    ap.add_argument(
        "--format",
        default="png",
        choices=["png", "svg", "pdf"],
        help="Image file format.",
    )
    ap.add_argument(
        "--rolling",
        type=int,
        default=0,
        help="Optional moving average window (>=2) for smoothing curves.",
    )
    ap.add_argument("--show", action="store_true", help="Show plots interactively.")
    ap.add_argument(
        "--separate",
        action="store_true",
        help="Save separate files for each panel instead of a grid.",
    )
    ap.add_argument("--no-gap", action="store_true", help="Disable accuracy gap panel.")
    return ap


def safe_float(x: str) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def read_metrics(csv_path: Path) -> Dict[int, EpochMetrics]:
    rows: Dict[int, EpochMetrics] = {}
    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Column names may include macro_f1_seen / macro_f1_all or only macro_f1 (legacy).
        has_macro_seen = "macro_f1_seen" in reader.fieldnames
        has_macro_all = "macro_f1_all" in reader.fieldnames
        # Legacy fallback
        has_macro_legacy = "macro_f1" in reader.fieldnames

        for row in reader:
            try:
                epoch = int(row.get("epoch", "0"))
            except ValueError:
                continue
            phase = (row.get("phase") or "").strip().lower()
            if epoch not in rows:
                rows[epoch] = EpochMetrics(epoch=epoch)

            em = rows[epoch]

            # Common metrics
            loss = safe_float(row.get("loss"))
            acc1 = safe_float(row.get("accuracy_top1"))
            acc5 = safe_float(row.get("accuracy_top5"))
            dia_acc = safe_float(row.get("diacritic_subset_accuracy"))
            lr = safe_float(row.get("lr"))
            time_sec = safe_float(row.get("time_sec"))

            if phase == "train":
                em.train_loss = loss
                em.train_acc1 = acc1
                em.train_acc5 = acc5
                em.train_dia_acc = dia_acc
            elif phase == "val":
                em.val_loss = loss
                em.val_acc1 = acc1
                em.val_acc5 = acc5
                em.val_dia_acc = dia_acc
                if has_macro_seen:
                    em.macro_f1_seen = safe_float(row.get("macro_f1_seen"))
                if has_macro_all:
                    em.macro_f1_all = safe_float(row.get("macro_f1_all"))
                # Legacy macro_f1 fallback
                if not has_macro_seen and has_macro_legacy and em.macro_f1_seen is None:
                    m = safe_float(row.get("macro_f1"))
                    em.macro_f1_seen = m
                    em.macro_f1_all = m
            # Record lr/time from either row; val row will overwrite train row (acceptable)
            if lr is not None:
                em.lr = lr
            if time_sec is not None:
                em.time_sec = time_sec

    return rows


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------


def moving_average(values: List[Optional[float]], window: int) -> List[Optional[float]]:
    if window <= 1:
        return values
    out: List[Optional[float]] = [None] * len(values)
    half = window // 2
    for i in range(len(values)):
        start = max(0, i - half)
        end = min(len(values), i + half + (0 if window % 2 == 0 else 1))
        slice_vals = [v for v in values[start:end] if v is not None]
        if len(slice_vals) == 0:
            out[i] = values[i]
        else:
            out[i] = sum(slice_vals) / len(slice_vals)
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def ensure_matplotlib():
    try:
        import matplotlib  # noqa: F401
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as e:
        print(
            f"[ERROR] matplotlib is required for plotting but not available: {e}\n"
            "Install with: pip install matplotlib",
            file=sys.stderr,
        )
        sys.exit(3)


def plot_metrics(
    metrics: Dict[int, EpochMetrics],
    out_dir: Path,
    img_format: str,
    rolling: int,
    show: bool,
    separate: bool,
    include_gap: bool,
):
    import matplotlib.pyplot as plt

    epochs_sorted = sorted(metrics.keys())
    # Build lists
    train_loss = [metrics[e].train_loss for e in epochs_sorted]
    val_loss = [metrics[e].val_loss for e in epochs_sorted]
    train_acc1 = [metrics[e].train_acc1 for e in epochs_sorted]
    val_acc1 = [metrics[e].val_acc1 for e in epochs_sorted]
    val_acc5 = [metrics[e].val_acc5 for e in epochs_sorted]
    macro_f1_seen = [metrics[e].macro_f1_seen for e in epochs_sorted]
    macro_f1_all = [metrics[e].macro_f1_all for e in epochs_sorted]
    train_dia = [metrics[e].train_dia_acc for e in epochs_sorted]
    val_dia = [metrics[e].val_dia_acc for e in epochs_sorted]

    if rolling > 1:
        train_loss = moving_average(train_loss, rolling)
        val_loss = moving_average(val_loss, rolling)
        train_acc1 = moving_average(train_acc1, rolling)
        val_acc1 = moving_average(val_acc1, rolling)
        val_acc5 = moving_average(val_acc5, rolling)
        macro_f1_seen = moving_average(macro_f1_seen, rolling)
        macro_f1_all = moving_average(macro_f1_all, rolling)
        train_dia = moving_average(train_dia, rolling)
        val_dia = moving_average(val_dia, rolling)

    if separate:
        out_dir.mkdir(parents=True, exist_ok=True)

    def save_or_show(fig: Any, name: str):
        path = out_dir / f"{name}.{img_format}"
        fig.tight_layout()
        fig.savefig(path)
        print(f"[INFO] Saved {path}")
        if not separate:
            plt.close(fig)

    # Panel 1: Loss
    def panel_loss():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs_sorted, train_loss, label="train_loss", marker="o")
        ax.plot(epochs_sorted, val_loss, label="val_loss", marker="o")
        ax.set_title("Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        return fig

    # Panel 2: Accuracy
    def panel_accuracy():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs_sorted, train_acc1, label="train_top1", marker="o")
        ax.plot(epochs_sorted, val_acc1, label="val_top1", marker="o")
        ax.plot(epochs_sorted, val_acc5, label="val_top5", linestyle="--", marker="x")
        ax.set_title("Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend()
        return fig

    # Panel 3: Macro F1
    def panel_macro_f1():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs_sorted, macro_f1_seen, label="macro_f1_seen", marker="o")
        ax.plot(
            epochs_sorted,
            macro_f1_all,
            label="macro_f1_all",
            marker="x",
            linestyle="--",
        )
        ax.set_title("Macro F1 (Seen vs All)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Macro F1")
        ax.grid(True, alpha=0.3)
        ax.legend()
        return fig

    # Panel 4: Diacritic subset accuracy
    def panel_diacritic():
        fig, ax = plt.subplots(figsize=(6, 4))
        if any(v is not None for v in train_dia):
            ax.plot(epochs_sorted, train_dia, label="train_diacritic_acc", marker="o")
        if any(v is not None for v in val_dia):
            ax.plot(epochs_sorted, val_dia, label="val_diacritic_acc", marker="x")
        ax.set_title("Diacritic Subset Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend()
        return fig

    # Panel 5: Accuracy gap (optional)
    def panel_gap():
        gap_vals = []
        for t, v in zip(train_acc1, val_acc1):
            if t is not None and v is not None:
                gap_vals.append(t - v)
            else:
                gap_vals.append(None)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs_sorted, gap_vals, label="train_top1 - val_top1", marker="o")
        ax.set_title("Accuracy Gap (Overfitting Signal)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gap")
        ax.grid(True, alpha=0.3)
        ax.legend()
        return fig

    if separate:
        if any(x is not None for x in train_loss + val_loss):
            save_or_show(panel_loss(), "loss")
        if any(x is not None for x in train_acc1 + val_acc1):
            save_or_show(panel_accuracy(), "accuracy")
        if any(x is not None for x in macro_f1_seen + macro_f1_all):
            save_or_show(panel_macro_f1(), "macro_f1")
        if any(x is not None for x in train_dia + val_dia):
            save_or_show(panel_diacritic(), "diacritic_accuracy")
        if include_gap and any(x is not None for x in train_acc1 + val_acc1):
            save_or_show(panel_gap(), "accuracy_gap")
    else:
        # Combined grid
        panels = [
            panel_loss(),
            panel_accuracy(),
            panel_macro_f1(),
            panel_diacritic(),
        ]
        if include_gap:
            panels.append(panel_gap())

        import math as _math

        n = len(panels)
        cols = 2
        rows = _math.ceil(n / cols)
        # Re-draw into a single figure
        import matplotlib.pyplot as plt2

        fig, axes = plt2.subplots(rows, cols, figsize=(cols * 6, rows * 4))
        axes = axes.flatten() if isinstance(axes, (list, tuple)) else axes.ravel()

        for ax, subfig in zip(axes, panels):
            # Transfer artists
            for artist in subfig.axes[0].get_children():
                # High-level clone unsafe; easier: replot underlying data forcibly
                # Instead, we reconstruct by grabbing lines
                pass
            # Instead of copying, we regenerate the content by calling the functions differently.
            plt2.close(subfig)

        # Re-generate directly on axes to avoid complex artist copying.
        # (Re-call logic for each panel with a target axis)
        def redraw_loss(ax):
            ax.plot(epochs_sorted, train_loss, label="train_loss", marker="o")
            ax.plot(epochs_sorted, val_loss, label="val_loss", marker="o")
            ax.set_title("Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
            ax.legend()

        def redraw_acc(ax):
            ax.plot(epochs_sorted, train_acc1, label="train_top1", marker="o")
            ax.plot(epochs_sorted, val_acc1, label="val_top1", marker="o")
            ax.plot(
                epochs_sorted, val_acc5, label="val_top5", linestyle="--", marker="x"
            )
            ax.set_title("Accuracy")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.grid(True, alpha=0.3)
            ax.legend()

        def redraw_macro(ax):
            ax.plot(epochs_sorted, macro_f1_seen, label="macro_f1_seen", marker="o")
            ax.plot(
                epochs_sorted,
                macro_f1_all,
                label="macro_f1_all",
                marker="x",
                linestyle="--",
            )
            ax.set_title("Macro F1 (Seen vs All)")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Macro F1")
            ax.grid(True, alpha=0.3)
            ax.legend()

        def redraw_diacritic(ax):
            if any(v is not None for v in train_dia):
                ax.plot(epochs_sorted, train_dia, label="train_diacritic", marker="o")
            if any(v is not None for v in val_dia):
                ax.plot(epochs_sorted, val_dia, label="val_diacritic", marker="x")
            ax.set_title("Diacritic Accuracy")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.grid(True, alpha=0.3)
            ax.legend()

        def redraw_gap(ax):
            gap_vals = []
            for t, v in zip(train_acc1, val_acc1):
                if t is not None and v is not None:
                    gap_vals.append(t - v)
                else:
                    gap_vals.append(None)
            ax.plot(epochs_sorted, gap_vals, label="acc_gap", marker="o")
            ax.set_title("Accuracy Gap")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Train - Val")
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Redraw according to desired ordering
        redraw_funcs = [redraw_loss, redraw_acc, redraw_macro, redraw_diacritic]
        if include_gap:
            redraw_funcs.append(redraw_gap)

        for func, ax in zip(redraw_funcs, axes):
            func(ax)

        # Hide unused axes
        for ax in axes[len(redraw_funcs) :]:
            ax.axis("off")

        out_dir.mkdir(parents=True, exist_ok=True)
        combined_path = out_dir / f"phase2_metrics.{img_format}"
        fig.tight_layout()
        fig.savefig(combined_path)
        print(f"[INFO] Saved {combined_path}")
        if not show:
            import matplotlib.pyplot as plt3

            plt3.close(fig)

    if show:
        import matplotlib.pyplot as plt_show

        plt_show.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = parse_args(argv)
    args = parser.parse_args(argv)

    try:
        ensure_matplotlib()
        metrics = read_metrics(args.metrics_csv)
        if not metrics:
            print("[ERROR] No metrics parsed (empty CSV?)", file=sys.stderr)
            return 2
        plot_metrics(
            metrics=metrics,
            out_dir=args.out_dir,
            img_format=args.format,
            rolling=args.rolling,
            show=args.show,
            separate=args.separate,
            include_gap=not args.no_gap,
        )
        return 0
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2
    except SystemExit:
        raise
    except Exception as e:
        print(f"[ERROR] Unhandled exception: {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
