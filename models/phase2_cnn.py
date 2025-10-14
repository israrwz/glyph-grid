#!/usr/bin/env python3
"""
Phase 2 CNN (Glyph Classification from Primitive ID Grid)
========================================================

This module provides a convolutional alternative to the transformer-based
Phase 2 glyph classifier. It consumes a (B,16,16) tensor of primitive
IDs (uint / int64) and outputs logits over glyph labels.

Motivation
----------
A compact CNN can:
  * Serve as a faster CPU / edge baseline.
  * Provide architectural diversity for ensembling (CNN + Transformer).
  * Offer stronger inductive spatial bias on small 16x16 grids.
  * Enable quicker hyperparameter iteration cycles.

High-Level Architecture
-----------------------
1. Primitive ID Embedding: nn.Embedding(vocab_size, embedding_dim)
   (Optional centroid initialization; projection if centroid_dim != embedding_dim)
2. Feature Tensor: (B, embedding_dim, 16, 16)
3. Convolutional "stages":
   - Stem: Conv(embedding_dim -> first_stage_channels)
   - Multiple residual blocks per stage
   - Spatial downsampling between stages via stride=2 conv (configurable)
4. Global pooling (mean)
5. Classification head (optional hidden MLP or single linear layer)

Configuration Mapping (phase2.yaml style)
-----------------------------------------
input:
  primitive_vocab_size
  embedding_dim
  normalize_embeddings (applied after embedding lookup; L2 norm)
model:
  architecture: "cnn"
  cnn:
    stages:               # Optional explicit per-stage channel list (overrides width_base & width_factor)
      - 64
      - 128
      - 192
    width_base: 64        # Used if stages not provided
    width_factor: 2.0     # Used if stages not provided (geometric progression)
    num_stages: 3
    blocks_per_stage: [2,2,2]  # Residual blocks per stage
    kernel_size: 3
    stem_kernel_size: 3
    stem_stride: 1
    downsample: conv      # {conv, none}
    activation: gelu      # {gelu, relu}
    dropout: 0.1
    bn_eps: 1e-5
    classifier_hidden_dim: 0   # 0 or None => single linear
    classifier_dropout: 0.1
  init:
    embedding_from_centroids: true
    centroid_requires_grad: true
    weight_init: xavier_uniform  # {xavier_uniform, kaiming_normal, default}

Factory
-------
build_phase2_cnn_model(cfg_dict, num_labels, primitive_centroids=None)

Usage
-----
from models.phase2_cnn import build_phase2_cnn_model
model = build_phase2_cnn_model(cfg.raw, num_labels=NUM_LABELS, primitive_centroids=centroids_array)
logits = model(grid_ids)  # grid_ids shape (B,16,16)

Note
----
This file is *additive* and does not modify the transformer implementation.
Training script (train/train_phase2.py) should branch on:
  if cfg.get("model","architecture") == "cnn": build_phase2_cnn_model(...)
else fall back to existing transformer builder.

License
-------
Follows the project root license.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Utility / Activation
# ---------------------------------------------------------------------------


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"Unsupported activation '{name}' (expected gelu|relu)")


# ---------------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------------


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        activation: str = "gelu",
        bn_eps: float = 1e-5,
        dropout: float = 0.0,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch, eps=bn_eps)
        self.act = _make_activation(activation)
        self.do = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.do(self.act(self.bn(self.conv(x))))


class ResidualBlock(nn.Module):
    """
    Standard pre-activation style residual block (Conv-BN-Act ...).
    Downsampling done outside block via stage transition.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        activation: str = "gelu",
        bn_eps: float = 1e-5,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv1 = ConvBNAct(
            channels,
            channels,
            kernel_size=kernel_size,
            activation=activation,
            bn_eps=bn_eps,
            dropout=dropout,
        )
        self.conv2 = ConvBNAct(
            channels,
            channels,
            kernel_size=kernel_size,
            activation=activation,
            bn_eps=bn_eps,
            dropout=0.0,  # avoid double dropout inside block
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.dropout(out)
        return out + x


# ---------------------------------------------------------------------------
# Configuration Dataclass
# ---------------------------------------------------------------------------


@dataclass
class Phase2CNNConfig:
    vocab_size: int
    embedding_dim: int
    normalize_embeddings: bool
    # Architecture
    stages: List[int]
    blocks_per_stage: List[int]
    kernel_size: int
    stem_kernel_size: int
    stem_stride: int
    downsample: str
    activation: str
    dropout: float
    bn_eps: float
    classifier_hidden_dim: int
    classifier_dropout: float
    # Init
    embedding_from_centroids: bool
    centroid_requires_grad: bool
    weight_init: str


# ---------------------------------------------------------------------------
# Main CNN Model
# ---------------------------------------------------------------------------


class Phase2CNGlyphClassifier(nn.Module):
    """
    CNN-based glyph classifier for primitive ID grids.

    Forward Input:
      grid_ids: (B,16,16) int64 primitive IDs

    Output:
      logits: (B, num_labels)
    """

    def __init__(
        self,
        cfg: Phase2CNNConfig,
        num_labels: int,
        primitive_centroids: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.num_labels = num_labels

        # Embedding
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)

        # Optional centroid initialization
        if (
            cfg.embedding_from_centroids
            and primitive_centroids is not None
            and primitive_centroids.shape[0] == cfg.vocab_size
        ):
            with torch.no_grad():
                centroids = primitive_centroids.float()
                if centroids.shape[1] != cfg.embedding_dim:
                    # Project to embedding dim
                    proj = nn.Linear(centroids.shape[1], cfg.embedding_dim, bias=False)
                    nn.init.xavier_uniform_(proj.weight)
                    centroids = proj(centroids)
                self.embedding.weight.data.copy_(centroids)
            if not cfg.centroid_requires_grad:
                self.embedding.weight.requires_grad_(False)

        # Stem
        stem_out = cfg.stages[0]
        self.stem = ConvBNAct(
            cfg.embedding_dim,
            stem_out,
            kernel_size=cfg.stem_kernel_size,
            stride=cfg.stem_stride,
            activation=cfg.activation,
            bn_eps=cfg.bn_eps,
            dropout=cfg.dropout,
        )

        # Stages
        blocks = []
        in_channels = stem_out
        for stage_idx, stage_channels in enumerate(cfg.stages):
            if stage_idx > 0:
                # Downsample between stages if configured
                if cfg.downsample == "conv":
                    blocks.append(
                        ConvBNAct(
                            in_channels,
                            stage_channels,
                            kernel_size=3,
                            stride=2,
                            activation=cfg.activation,
                            bn_eps=cfg.bn_eps,
                            dropout=cfg.dropout,
                        )
                    )
                elif cfg.downsample == "none":
                    # Direct channel adjust if mismatch
                    if in_channels != stage_channels:
                        blocks.append(
                            ConvBNAct(
                                in_channels,
                                stage_channels,
                                kernel_size=1,
                                stride=1,
                                activation=cfg.activation,
                                bn_eps=cfg.bn_eps,
                                dropout=0.0,
                            )
                        )
                else:
                    raise ValueError(f"Unsupported downsample mode {cfg.downsample}")
                in_channels = stage_channels

            num_blocks = (
                cfg.blocks_per_stage[stage_idx]
                if stage_idx < len(cfg.blocks_per_stage)
                else cfg.blocks_per_stage[-1]
            )
            for _ in range(num_blocks):
                blocks.append(
                    ResidualBlock(
                        channels=in_channels,
                        kernel_size=cfg.kernel_size,
                        activation=cfg.activation,
                        bn_eps=cfg.bn_eps,
                        dropout=cfg.dropout,
                    )
                )

        self.features = nn.Sequential(*blocks)

        # Head
        feat_dim = cfg.stages[-1]
        if cfg.classifier_hidden_dim and cfg.classifier_hidden_dim > 0:
            hidden = cfg.classifier_hidden_dim
            self.classifier = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, hidden),
                _make_activation(cfg.activation),
                nn.Dropout(cfg.classifier_dropout),
                nn.Linear(hidden, num_labels),
            )
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Dropout(cfg.classifier_dropout),
                nn.Linear(feat_dim, num_labels),
            )

        self.normalize_embeddings = cfg.normalize_embeddings

        self._apply_weight_init(cfg.weight_init)

    # ------------------------------------------------------------------
    # Weight Initialization
    # ------------------------------------------------------------------
    def _apply_weight_init(self, strategy: str):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if strategy == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight)
                elif strategy == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                if strategy == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight)
                elif strategy == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, grid_ids: torch.Tensor) -> torch.Tensor:
        """
        grid_ids: (B,16,16) int64
        """
        if grid_ids.dim() != 3 or grid_ids.shape[1:] != (16, 16):
            raise ValueError(
                f"Expected grid_ids shape (B,16,16); got {tuple(grid_ids.shape)}"
            )
        emb = self.embedding(grid_ids)  # (B,16,16,E)
        if self.normalize_embeddings:
            emb = F.normalize(emb, p=2, dim=-1)
        # Rearrange to NCHW
        x = emb.permute(0, 3, 1, 2)  # (B,E,16,16)
        x = self.stem(x)
        x = self.features(x)  # (B,C,H,W)
        # Global average pool
        x = x.mean(dim=(2, 3))  # (B,C)
        logits = self.classifier(x)
        return logits


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _extract_cnn_cfg(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    model_root = cfg_dict.get("model", {})
    cnn_root = model_root.get("cnn", {})
    init_root = model_root.get("init", {})
    input_root = cfg_dict.get("input", {})

    vocab_size = int(input_root.get("primitive_vocab_size", 1024))
    embedding_dim = int(input_root.get("embedding_dim", 64))
    normalize_embeddings = bool(input_root.get("normalize_embeddings", False))

    # Stage channels
    stages: Optional[Sequence[int]] = cnn_root.get("stages")
    if stages:
        stage_channels = [int(c) for c in stages]
    else:
        width_base = int(cnn_root.get("width_base", 64))
        width_factor = float(cnn_root.get("width_factor", 2.0))
        num_stages = int(cnn_root.get("num_stages", 3))
        stage_channels = [
            int(round(width_base * (width_factor**i))) for i in range(num_stages)
        ]

    blocks_per_stage = cnn_root.get("blocks_per_stage", [2] * len(stage_channels))
    if len(blocks_per_stage) < len(stage_channels):
        # Pad with last value
        last = blocks_per_stage[-1]
        blocks_per_stage = list(blocks_per_stage) + [last] * (
            len(stage_channels) - len(blocks_per_stage)
        )

    cfg = {
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "normalize_embeddings": normalize_embeddings,
        "stages": stage_channels,
        "blocks_per_stage": [int(b) for b in blocks_per_stage],
        "kernel_size": int(cnn_root.get("kernel_size", 3)),
        "stem_kernel_size": int(cnn_root.get("stem_kernel_size", 3)),
        "stem_stride": int(cnn_root.get("stem_stride", 1)),
        "downsample": cnn_root.get("downsample", "conv"),
        "activation": cnn_root.get("activation", "gelu"),
        "dropout": float(cnn_root.get("dropout", 0.1)),
        "bn_eps": float(cnn_root.get("bn_eps", 1e-5)),
        "classifier_hidden_dim": int(cnn_root.get("classifier_hidden_dim", 0) or 0),
        "classifier_dropout": float(cnn_root.get("classifier_dropout", 0.1)),
        "embedding_from_centroids": bool(
            init_root.get("embedding_from_centroids", True)
        ),
        "centroid_requires_grad": bool(init_root.get("centroid_requires_grad", True)),
        "weight_init": init_root.get("weight_init", "xavier_uniform"),
    }
    return cfg


def build_phase2_cnn_model(
    cfg_dict: Dict[str, Any],
    num_labels: int,
    primitive_centroids: Optional["torch.Tensor | Any"] = None,
) -> Phase2CNGlyphClassifier:
    """
    Build Phase 2 CNN model from config dict (mirroring phase2.yaml).
    """
    cnn_cfg_raw = _extract_cnn_cfg(cfg_dict)
    cfg = Phase2CNNConfig(**cnn_cfg_raw)

    # Convert centroids if numpy
    if primitive_centroids is not None and not torch.is_tensor(primitive_centroids):
        primitive_centroids = torch.as_tensor(primitive_centroids, dtype=torch.float32)

    model = Phase2CNGlyphClassifier(
        cfg=cfg,
        num_labels=num_labels,
        primitive_centroids=primitive_centroids,
    )
    return model


__all__ = [
    "Phase2CNGlyphClassifier",
    "Phase2CNNConfig",
    "build_phase2_cnn_model",
]
