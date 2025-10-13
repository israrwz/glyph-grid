#!/usr/bin/env python3
"""
Phase 1 Baseline CNN (Primitive Cell Classifier)

Implements the reference architecture from NEW_PLAN.md §6.2 and configs/phase1.yaml:

Architecture:
  Input: (B, 1, 8, 8) binary cells (0/1 or 0/255)
  Block 1: Conv(32,3,pad=1) + BatchNorm + ReLU -> MaxPool(2)  -> (B,32,4,4)
  Block 2: Conv(64,3,pad=1) + BatchNorm + ReLU -> MaxPool(2)  -> (B,64,2,2)
  Flatten: 64 * 2 * 2 = 256
  FC: 256 -> 128 + ReLU + Dropout(0.2)
  FC: 128 -> 1024 logits (primitive classes; 0 = EMPTY)

Features:
  - Optional weight initialization (Kaiming Normal for conv / linear)
  - Explicit parameter counting helper
  - Mixed precision friendly (no hard-coded dtype conversions)
  - Accepts either float or uint8 inputs; internally normalizes to float in [0,1]
  - Minimal dependencies (torch & optional torchinfo for summary)

Usage:
  from models.phase1_cnn import build_phase1_model
  model = build_phase1_model({
      "in_channels": 1,
      "conv_blocks": [
         {"out_channels": 32, "kernel": 3, "stride": 1, "padding": 1, "batchnorm": True, "pool": 2},
         {"out_channels": 64, "kernel": 3, "stride": 1, "padding": 1, "batchnorm": True, "pool": 2},
      ],
      "flatten_dim": 256,
      "fc_hidden": 128,
      "fc_dropout": 0.2,
      "num_classes": 1024,
      "weight_init": "kaiming_normal",
  })

Limitations / Notes:
  - This module intentionally avoids any training loop logic; see train/train_phase1.py (to be implemented) for end‑to‑end usage.
  - Assumes the upstream pipeline guarantees correct 8x8 sizing; adds assert for safety in debug mode.
  - If you later switch to learned embeddings instead of hard IDs, this model may be repurposed as a feature extractor by removing final classifier layer.

Copyright:
  Provided under the project license. No external code copied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import torch
    from torch import nn
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "PyTorch is required to use Phase 1 CNN. Please install torch."
    ) from e


# ---------------------------------------------------------------------------
# Configuration Dataclass (Optional Helper)
# ---------------------------------------------------------------------------


@dataclass
class Phase1CNNConfig:
    in_channels: int = 1
    conv_blocks: List[Dict[str, Any]] = None  # Filled in build helper
    flatten_dim: int = 256
    fc_hidden: int = 128
    fc_dropout: float = 0.2
    num_classes: int = 1024
    weight_init: str = "kaiming_normal"

    def validate(self):
        assert self.in_channels > 0, "in_channels must be positive"
        assert isinstance(self.conv_blocks, list) and len(self.conv_blocks) == 2, (
            "Baseline expects exactly 2 conv blocks (see plan)."
        )
        assert self.flatten_dim == 256, (
            f"Expected flatten_dim=256 for (64,2,2) layout; got {self.flatten_dim}."
        )
        assert self.fc_hidden > 0
        assert self.num_classes >= 2
        assert 0.0 <= self.fc_dropout < 1.0
        assert self.weight_init in {"kaiming_normal", "none"}


# ---------------------------------------------------------------------------
# Model Definition
# ---------------------------------------------------------------------------


class BaselinePhase1CNN(nn.Module):
    """
    Baseline primitive classifier for 8x8 cell bitmaps.

    Args:
        in_channels: Input channels (1 for binary cells).
        conv_blocks: List of 2 dicts, each with:
            - out_channels (int)
            - kernel (int)
            - stride (int)
            - padding (int)
            - batchnorm (bool)
            - pool (int or None)  # pooling kernel (assumed square)
        flatten_dim: Expected flattened dimension after conv blocks.
        fc_hidden: Hidden units in intermediate FC layer.
        fc_dropout: Dropout probability after hidden FC.
        num_classes: Output classes (1024 by default).
    """

    def __init__(
        self,
        in_channels: int,
        conv_blocks: List[Dict[str, Any]],
        flatten_dim: int,
        fc_hidden: int,
        fc_dropout: float,
        num_classes: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.flatten_dim = flatten_dim
        self.fc_hidden = fc_hidden
        self.num_classes = num_classes
        self.fc_dropout_p = fc_dropout

        # Build convolutional feature extractor
        layers: List[nn.Module] = []
        current_in = in_channels
        for i, spec in enumerate(conv_blocks):
            oc = spec["out_channels"]
            k = spec.get("kernel", 3)
            s = spec.get("stride", 1)
            p = spec.get("padding", 0)
            use_bn = bool(spec.get("batchnorm", True))
            pool = spec.get("pool", 2)
            conv = nn.Conv2d(
                current_in, oc, kernel_size=k, stride=s, padding=p, bias=not use_bn
            )
            layers.append(conv)
            if use_bn:
                layers.append(nn.BatchNorm2d(oc))
            layers.append(nn.ReLU(inplace=True))
            if pool:
                layers.append(nn.MaxPool2d(kernel_size=pool))
            current_in = oc
        self.features = nn.Sequential(*layers)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=fc_dropout),
            nn.Linear(fc_hidden, num_classes),
        )

        # Track conv specs for introspection
        self._conv_specs = conv_blocks

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Input:
            x: Tensor (B,1,8,8) or (B,in_channels,8,8), dtype float/uint8/bool
        Returns:
            logits: (B, num_classes)
        """
        # Normalize to float32 in [0,1] if given uint8/bool
        if x.dtype == torch.uint8 or x.dtype == torch.bool:
            x = x.to(torch.float32)
        if x.max() > 1.0:  # e.g., 0/255
            x = x / 255.0

        if self.training and __debug__:
            assert x.shape[-2:] == (8, 8), (
                f"Expected input spatial size (8,8); got {x.shape[-2:]}"
            )

        feats = self.features(x)
        logits = self.classifier(feats)
        return logits

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, flatten_dim={self.flatten_dim}, "
            f"fc_hidden={self.fc_hidden}, num_classes={self.num_classes}, "
            f"dropout={self.fc_dropout_p}"
        )


# ---------------------------------------------------------------------------
# Weight Initialization
# ---------------------------------------------------------------------------


def apply_weight_init(module: nn.Module, strategy: str = "kaiming_normal") -> None:
    """
    Apply weight initialization recursively.
    """
    if strategy == "none":
        return

    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if strategy == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            else:  # Future strategies can be added here
                raise ValueError(f"Unknown weight init strategy: {strategy}")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_phase1_model(cfg_dict: Dict[str, Any]) -> BaselinePhase1CNN:
    """
    Construct a BaselinePhase1CNN from a simple config dictionary
    (mirrors the 'model' section of configs/phase1.yaml).

    Expected keys in cfg_dict:
      in_channels, conv_blocks, flatten_dim, fc_hidden,
      fc_dropout, num_classes, weight_init

    Returns:
        BaselinePhase1CNN (initialized, weights optionally seeded)
    """
    # Provide defaults if keys missing
    defaults = {
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
    merged = {**defaults, **cfg_dict}

    cfg = Phase1CNNConfig(
        in_channels=merged["in_channels"],
        conv_blocks=merged["conv_blocks"],
        flatten_dim=merged["flatten_dim"],
        fc_hidden=merged["fc_hidden"],
        fc_dropout=merged["fc_dropout"],
        num_classes=merged["num_classes"],
        weight_init=merged["weight_init"],
    )
    cfg.validate()

    model = BaselinePhase1CNN(
        in_channels=cfg.in_channels,
        conv_blocks=cfg.conv_blocks,
        flatten_dim=cfg.flatten_dim,
        fc_hidden=cfg.fc_hidden,
        fc_dropout=cfg.fc_dropout,
        num_classes=cfg.num_classes,
    )

    apply_weight_init(model, cfg.weight_init)
    return model


# ---------------------------------------------------------------------------
# Optional: Simple Self-Test
# ---------------------------------------------------------------------------


def _self_test() -> None:  # pragma: no cover
    cfg = {
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
    model = build_phase1_model(cfg)
    x = torch.randint(0, 2, (8, 1, 8, 8), dtype=torch.uint8)
    logits = model(x)
    assert logits.shape == (8, 1024), f"Unexpected output shape: {logits.shape}"
    print("Self-test passed. Param count:", model.count_parameters())


if __name__ == "__main__":  # pragma: no cover
    _self_test()


__all__ = [
    "BaselinePhase1CNN",
    "Phase1CNNConfig",
    "build_phase1_model",
    "apply_weight_init",
]
