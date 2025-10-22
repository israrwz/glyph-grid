#!/usr/bin/env python3
"""
Phase 2 Sequence-Aware CNN (Glyph Classification with Context)
==============================================================

This module extends the Phase 2 CNN classifier to incorporate sequential context
from neighboring glyphs in the font's glyph order. This helps disambiguate
visually similar glyphs (e.g., Arabic digit 1 vs Aleph, English O vs Arabic heh).

Key Differences from phase2_cnn.py:
-----------------------------------
1. **Context Window**: Processes N neighbors before/after the center glyph
2. **Dual-Stream Architecture**:
   - Visual stream: Processes center glyph appearance (main CNN)
   - Sequence stream: Processes neighbor embeddings (lightweight LSTM/attention)
3. **Multi-Objective Loss**:
   - Primary: Visual classification loss (center glyph)
   - Auxiliary: Sequence consistency loss (predicting center from neighbors)
4. **Fusion**: Combines visual + sequence features for final prediction

Architecture Overview:
---------------------
Input:
  - center_grid: (B, 16, 16) - center glyph primitive IDs
  - context_grids: (B, K, 16, 16) - K neighbor grids (K=2*window_size)
  - context_deltas: (B, K) - glyph_id deltas (relative positions)

Visual Stream:
  - Same CNN as phase2_cnn.py → visual_features (B, C)

Sequence Stream:
  - Lightweight encoder for each neighbor → context_embeddings (B, K, D)
  - LSTM/Attention over context → sequence_features (B, D)
  - Positional encoding from context_deltas

Fusion:
  - Concatenate [visual_features, sequence_features]
  - Gated fusion (learnable weight between visual/sequence)
  - Final classifier

Multi-Objective Training:
------------------------
Loss = λ_visual * L_visual + λ_sequence * L_sequence

Where:
  - L_visual: Cross-entropy on final fused prediction
  - L_sequence: Predict center glyph class from sequence features alone
  - λ_visual = 1.0 (shape matching is primary)
  - λ_sequence = 0.3 (sequence is auxiliary signal)

This ensures the model learns both visual recognition and sequential patterns
without sacrificing basic shape matching ability.

Usage:
------
from models.phase2_cnn_sequence import build_phase2_sequence_model

model = build_phase2_sequence_model(
    cfg_dict,
    num_labels=1216,
    context_window=2,  # 2 before, 2 after
    sequence_weight=0.3
)

# Training
logits, aux_outputs = model(center_grid, context_grids, context_deltas)
loss = criterion(logits, labels, aux_outputs)

# Inference (fallback to visual-only if context unavailable)
logits, _ = model(center_grid)  # context optional
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from models.phase2_cnn import (
    Phase2CNNConfig,
    ConvBNAct,
    ResidualBlock,
    _make_activation,
)


# ---------------------------------------------------------------------------
# Sequence Encoder Modules
# ---------------------------------------------------------------------------


class ContextEncoder(nn.Module):
    """
    Lightweight encoder for processing neighbor glyph grids.
    Shares embedding with main model but has separate lightweight CNN.
    """

    def __init__(
        self,
        embedding: nn.Embedding,
        embedding_dim: int,
        hidden_dim: int,
        activation: str = "gelu",
    ):
        super().__init__()
        self.embedding = embedding  # Shared with main model
        self.normalize_embeddings = False

        # Lightweight CNN: 2 conv blocks with aggressive pooling
        self.conv1 = ConvBNAct(
            embedding_dim,
            hidden_dim // 2,
            kernel_size=3,
            stride=2,
            activation=activation,
        )
        self.conv2 = ConvBNAct(
            hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, activation=activation
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, grid_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid_ids: (B, 16, 16) or (B*K, 16, 16)
        Returns:
            features: (B, hidden_dim) or (B*K, hidden_dim)
        """
        emb = self.embedding(grid_ids)  # (B, 16, 16, E)
        if self.normalize_embeddings:
            emb = F.normalize(emb, p=2, dim=-1)
        x = emb.permute(0, 3, 1, 2)  # (B, E, 16, 16)
        x = self.conv1(x)  # (B, H/2, 8, 8)
        x = self.conv2(x)  # (B, H, 4, 4)
        x = self.pool(x).squeeze(-1).squeeze(-1)  # (B, H)
        return x


class PositionalDeltaEncoder(nn.Module):
    """
    Encodes relative position deltas between center and context glyphs.
    Uses sinusoidal encoding + learned projection.
    """

    def __init__(self, hidden_dim: int, max_delta: int = 10):
        super().__init__()
        self.max_delta = max_delta
        self.projection = nn.Linear(2 * max_delta + 1, hidden_dim)

    def forward(self, deltas: torch.Tensor) -> torch.Tensor:
        """
        Args:
            deltas: (B, K) - relative glyph_id deltas (can be negative)
        Returns:
            encodings: (B, K, hidden_dim)
        """
        # Clip and normalize deltas
        deltas_clipped = torch.clamp(deltas, -self.max_delta, self.max_delta)

        # One-hot style encoding (bucket into 2*max_delta+1 bins)
        bins = deltas_clipped + self.max_delta  # Shift to [0, 2*max_delta]
        one_hot = F.one_hot(bins.long(), num_classes=2 * self.max_delta + 1).float()

        return self.projection(one_hot)


class SequenceProcessor(nn.Module):
    """
    Processes sequence of context embeddings with LSTM or attention.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        mode: str = "lstm",
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mode = mode

        if mode == "lstm":
            self.lstm = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=True,
            )
            self.projection = nn.Linear(2 * hidden_dim, hidden_dim)
        elif mode == "attention":
            self.self_attn = nn.MultiheadAttention(
                input_dim, num_heads=4, dropout=dropout, batch_first=True
            )
            self.norm = nn.LayerNorm(input_dim)
            self.projection = nn.Linear(input_dim, hidden_dim)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, K, D) - sequence of context embeddings
            mask: (B, K) - optional mask for padding
        Returns:
            features: (B, hidden_dim) - aggregated sequence features
        """
        if self.mode == "lstm":
            out, (h_n, _) = self.lstm(x)  # out: (B, K, 2*H), h_n: (2*layers, B, H)
            # Use final hidden state (forward + backward)
            final = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, 2*H)
            return self.projection(final)
        else:  # attention
            attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
            normed = self.norm(attn_out + x)
            # Mean pooling over sequence
            if mask is not None:
                mask_expanded = (~mask).float().unsqueeze(-1)  # (B, K, 1)
                pooled = (normed * mask_expanded).sum(dim=1) / mask_expanded.sum(
                    dim=1
                ).clamp(min=1)
            else:
                pooled = normed.mean(dim=1)  # (B, D)
            return self.projection(pooled)


# ---------------------------------------------------------------------------
# Main Sequence-Aware Model
# ---------------------------------------------------------------------------


@dataclass
class Phase2SequenceConfig:
    """Extended config for sequence-aware model."""

    base_config: Phase2CNNConfig
    context_window: int  # Number of neighbors on each side
    sequence_hidden_dim: int
    sequence_processor: str  # 'lstm' or 'attention'
    sequence_layers: int
    sequence_dropout: float
    fusion_mode: str  # 'concat', 'gated', 'weighted_sum'
    sequence_loss_weight: float
    max_delta: int


class Phase2SequenceAwareGlyphClassifier(nn.Module):
    """
    Sequence-aware glyph classifier that combines visual and sequential context.

    Forward modes:
        1. Full: (center_grid, context_grids, context_deltas) → logits + aux
        2. Visual-only: (center_grid) → logits (no aux, fallback for inference)
    """

    def __init__(
        self,
        cfg: Phase2SequenceConfig,
        num_labels: int,
        primitive_centroids: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.num_labels = num_labels
        self.context_window = cfg.context_window

        # Shared embedding
        vocab_size = cfg.base_config.vocab_size
        embedding_dim = cfg.base_config.embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Optional centroid initialization (same as base model)
        if (
            cfg.base_config.embedding_from_centroids
            and primitive_centroids is not None
            and primitive_centroids.shape[0] == vocab_size
        ):
            with torch.no_grad():
                centroids = primitive_centroids.float()
                if centroids.shape[1] != embedding_dim:
                    proj = nn.Linear(centroids.shape[1], embedding_dim, bias=False)
                    nn.init.xavier_uniform_(proj.weight)
                    centroids = proj(centroids)
                self.embedding.weight.data.copy_(centroids)
            if not cfg.base_config.centroid_requires_grad:
                self.embedding.weight.requires_grad_(False)

        # Visual stream (main CNN)
        self._build_visual_stream(cfg.base_config)

        # Sequence stream
        self.context_encoder = ContextEncoder(
            self.embedding,
            embedding_dim,
            cfg.sequence_hidden_dim,
            activation=cfg.base_config.activation,
        )
        self.delta_encoder = PositionalDeltaEncoder(
            cfg.sequence_hidden_dim, max_delta=cfg.max_delta
        )
        self.sequence_processor = SequenceProcessor(
            input_dim=cfg.sequence_hidden_dim * 2,  # context + delta
            hidden_dim=cfg.sequence_hidden_dim,
            mode=cfg.sequence_processor,
            num_layers=cfg.sequence_layers,
            dropout=cfg.sequence_dropout,
        )

        # Fusion
        visual_dim = cfg.base_config.stages[-1]
        fused_dim = self._build_fusion_layer(
            visual_dim, cfg.sequence_hidden_dim, cfg.fusion_mode
        )

        # Fused classifier (for when context is available)
        if cfg.base_config.classifier_hidden_dim > 0:
            self.fused_classifier = nn.Sequential(
                nn.LayerNorm(fused_dim),
                nn.Linear(fused_dim, cfg.base_config.classifier_hidden_dim),
                _make_activation(cfg.base_config.activation),
                nn.Dropout(cfg.base_config.classifier_dropout),
                nn.Linear(cfg.base_config.classifier_hidden_dim, num_labels),
            )
        else:
            self.fused_classifier = nn.Sequential(
                nn.LayerNorm(fused_dim),
                nn.Dropout(cfg.base_config.classifier_dropout),
                nn.Linear(fused_dim, num_labels),
            )

        # Visual-only classifier (for when context is unavailable)
        if cfg.base_config.classifier_hidden_dim > 0:
            self.visual_only_classifier = nn.Sequential(
                nn.LayerNorm(visual_dim),
                nn.Linear(visual_dim, cfg.base_config.classifier_hidden_dim),
                _make_activation(cfg.base_config.activation),
                nn.Dropout(cfg.base_config.classifier_dropout),
                nn.Linear(cfg.base_config.classifier_hidden_dim, num_labels),
            )
        else:
            self.visual_only_classifier = nn.Sequential(
                nn.LayerNorm(visual_dim),
                nn.Dropout(cfg.base_config.classifier_dropout),
                nn.Linear(visual_dim, num_labels),
            )

        # Auxiliary sequence classifier (for sequence loss)
        self.aux_sequence_classifier = nn.Sequential(
            nn.LayerNorm(cfg.sequence_hidden_dim),
            nn.Linear(cfg.sequence_hidden_dim, cfg.sequence_hidden_dim),
            _make_activation(cfg.base_config.activation),
            nn.Dropout(cfg.sequence_dropout),
            nn.Linear(cfg.sequence_hidden_dim, num_labels),
        )

        self.normalize_embeddings = cfg.base_config.normalize_embeddings
        self._apply_weight_init(cfg.base_config.weight_init)

    def _build_visual_stream(self, cfg: Phase2CNNConfig):
        """Build main CNN (same as Phase2CNGlyphClassifier)"""
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

        self.visual_features = nn.Sequential(*blocks)

    def _build_fusion_layer(self, visual_dim: int, sequence_dim: int, mode: str) -> int:
        """Build fusion mechanism"""
        self.fusion_mode = mode
        if mode == "concat":
            return visual_dim + sequence_dim
        elif mode == "gated":
            # Learnable gate to weight visual vs sequence
            self.gate = nn.Sequential(
                nn.Linear(visual_dim + sequence_dim, 1),
                nn.Sigmoid(),
            )
            return max(visual_dim, sequence_dim)
        elif mode == "weighted_sum":
            # Simple weighted addition (requires same dims)
            if visual_dim != sequence_dim:
                self.sequence_proj = nn.Linear(sequence_dim, visual_dim)
            else:
                self.sequence_proj = nn.Identity()
            self.alpha = nn.Parameter(
                torch.tensor(0.7)
            )  # Initial: 70% visual, 30% sequence
            return visual_dim
        else:
            raise ValueError(f"Unsupported fusion mode: {mode}")

    def _apply_weight_init(self, strategy: str):
        """Initialize weights"""
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

    def forward(
        self,
        center_grid: torch.Tensor,
        context_grids: Optional[torch.Tensor] = None,
        context_deltas: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Args:
            center_grid: (B, 16, 16) - center glyph
            context_grids: (B, K, 16, 16) - optional context (K=2*window)
            context_deltas: (B, K) - optional glyph_id deltas

        Returns:
            logits: (B, num_labels) - final predictions
            aux_outputs: dict with 'sequence_logits' for auxiliary loss
        """
        B = center_grid.shape[0]

        # Visual stream (center glyph)
        emb = self.embedding(center_grid)  # (B, 16, 16, E)
        if self.normalize_embeddings:
            emb = F.normalize(emb, p=2, dim=-1)
        x = emb.permute(0, 3, 1, 2)  # (B, E, 16, 16)
        x = self.stem(x)
        x = self.visual_features(x)  # (B, C, H, W)
        visual_feat = x.mean(dim=(2, 3))  # (B, C) - global avg pool

        # Sequence stream (if context available)
        aux_outputs = None
        if context_grids is not None and context_deltas is not None:
            K = context_grids.shape[1]
            # Flatten batch and context dims
            context_flat = context_grids.view(B * K, 16, 16)
            context_feat_flat = self.context_encoder(context_flat)  # (B*K, H)
            context_feat = context_feat_flat.view(B, K, -1)  # (B, K, H)

            # Add positional delta encoding
            delta_enc = self.delta_encoder(context_deltas)  # (B, K, H)
            combined = torch.cat([context_feat, delta_enc], dim=-1)  # (B, K, 2*H)

            # Process sequence
            sequence_feat = self.sequence_processor(combined)  # (B, H)

            # Fused features (apply fusion based on mode)
            if self.fusion_mode == "concat":
                fused_feat = torch.cat([visual_feat, sequence_feat], dim=1)
            elif self.fusion_mode == "gated":
                gate_val = self.gate(torch.cat([visual_feat, sequence_feat], dim=1))
                fused_feat = gate_val * visual_feat + (1 - gate_val) * sequence_feat
            elif self.fusion_mode == "weighted_sum":
                fused_feat = self.alpha * visual_feat + (
                    1 - self.alpha
                ) * self.sequence_proj(sequence_feat)

            # Auxiliary sequence prediction (for training only)
            sequence_logits = self.aux_sequence_classifier(sequence_feat)
            aux_outputs = {"sequence_logits": sequence_logits}

            # Final prediction with fused features
            logits = self.fused_classifier(fused_feat)
        else:
            # Visual-only mode (inference fallback)
            logits = self.visual_only_classifier(visual_feat)

        return logits, aux_outputs


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_phase2_sequence_model(
    cfg_dict: Dict[str, Any],
    num_labels: int,
    primitive_centroids: Optional[torch.Tensor] = None,
    context_window: Optional[int] = None,
    sequence_hidden_dim: Optional[int] = None,
    sequence_processor: Optional[str] = None,
    sequence_layers: Optional[int] = None,
    sequence_dropout: Optional[float] = None,
    fusion_mode: Optional[str] = None,
    sequence_loss_weight: Optional[float] = None,
    max_delta: Optional[int] = None,
) -> Phase2SequenceAwareGlyphClassifier:
    """
    Build sequence-aware Phase 2 model from config dict.

    Args:
        cfg_dict: Base config (same as phase2_cnn) with 'sequence' section
        num_labels: Number of glyph classes
        primitive_centroids: Optional centroid initialization
        context_window: Number of neighbors on each side (K=2*window) - overrides config
        sequence_hidden_dim: Hidden dim for sequence stream - overrides config
        sequence_processor: 'lstm' or 'attention' - overrides config
        sequence_layers: Number of LSTM/attention layers - overrides config
        sequence_dropout: Dropout for sequence stream - overrides config
        fusion_mode: 'concat', 'gated', or 'weighted_sum' - overrides config
        sequence_loss_weight: Weight for auxiliary sequence loss - overrides config
        max_delta: Maximum glyph_id delta to encode - overrides config
    """
    # Import base config extraction
    from models.phase2_cnn import _extract_cnn_cfg, Phase2CNNConfig

    base_cfg_raw = _extract_cnn_cfg(cfg_dict)
    base_cfg = Phase2CNNConfig(**base_cfg_raw)

    # Extract sequence config from cfg_dict
    model_root = cfg_dict.get("model", {})
    seq_root = model_root.get("sequence", {})
    loss_root = cfg_dict.get("loss", {})

    # Read from config with fallback to defaults
    context_window = (
        context_window
        if context_window is not None
        else int(seq_root.get("context_window", 2))
    )
    sequence_hidden_dim = (
        sequence_hidden_dim
        if sequence_hidden_dim is not None
        else seq_root.get("hidden_dim")
    )
    sequence_processor = (
        sequence_processor
        if sequence_processor is not None
        else seq_root.get("processor", "lstm")
    )
    sequence_layers = (
        sequence_layers
        if sequence_layers is not None
        else int(seq_root.get("num_layers", 2))
    )
    sequence_dropout = (
        sequence_dropout
        if sequence_dropout is not None
        else float(seq_root.get("dropout", 0.1))
    )
    fusion_mode = (
        fusion_mode
        if fusion_mode is not None
        else seq_root.get("fusion_mode", "concat")
    )
    max_delta = (
        max_delta if max_delta is not None else int(seq_root.get("max_delta", 10))
    )
    sequence_loss_weight = (
        sequence_loss_weight
        if sequence_loss_weight is not None
        else float(loss_root.get("sequence_weight", 0.3))
    )

    # Default sequence hidden dim to half of final stage width
    if sequence_hidden_dim is None:
        sequence_hidden_dim = base_cfg.stages[-1] // 2

    seq_cfg = Phase2SequenceConfig(
        base_config=base_cfg,
        context_window=context_window,
        sequence_hidden_dim=sequence_hidden_dim,
        sequence_processor=sequence_processor,
        sequence_layers=sequence_layers,
        sequence_dropout=sequence_dropout,
        fusion_mode=fusion_mode,
        sequence_loss_weight=sequence_loss_weight,
        max_delta=max_delta,
    )

    # Convert centroids if numpy
    if primitive_centroids is not None and not torch.is_tensor(primitive_centroids):
        primitive_centroids = torch.as_tensor(primitive_centroids, dtype=torch.float32)

    model = Phase2SequenceAwareGlyphClassifier(
        cfg=seq_cfg,
        num_labels=num_labels,
        primitive_centroids=primitive_centroids,
    )
    return model


__all__ = [
    "Phase2SequenceAwareGlyphClassifier",
    "Phase2SequenceConfig",
    "build_phase2_sequence_model",
]
