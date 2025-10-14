#!/usr/bin/env python3
"""
Phase 2 Transformer (Glyph Classification from Primitive ID Grid)

Implements the lightweight transformer architecture described in NEW_PLAN.md
Sections 7.1–7.3 and aligned with configs/phase2.yaml.

Core Responsibilities:
  * Embed a 16x16 grid (256 tokens) of primitive IDs (0..V-1; 0 = EMPTY).
  * Add 2D positional encoding (sinusoidal or learnable) to primitive embeddings.
  * Optional CLS token (prepended) used for classification pooling when enabled.
  * Apply a stack of Transformer encoder layers (pre-norm by default).
  * Produce glyph label logits via a classifier head (single Linear or small MLP).
  * Optional patch grouping (4x4) to reduce token length (disabled in baseline).
  * Optional initialization of primitive embedding weights from centroid vectors.
  * Optional L2 normalization of primitive embeddings (for prototype analysis).

Input Shapes:
  grid_ids: (B, H, W) int64  OR (B, H*W) flattened
            H=W=16 by default (plan). When patch grouping enabled, still pass
            the raw 16x16 grid; grouping occurs internally.

Output:
  logits: (B, num_labels)

Key Config Parameters (mapping to phase2.yaml):
  vocab_size                  -> input.primitive_vocab_size
  embedding_dim               -> input.embedding_dim
  positional_encoding         -> input.positional_encoding (sinusoidal_2d | learnable_2d)
  combine_mode                -> input.combine_mode (add | concat)
  patch_grouping.enabled      -> input.patch_grouping.enabled
  patch_rows / patch_cols     -> input.patch_grouping.patch_rows/patch_cols
  use_cls_token               -> input.use_cls_token
  token_pooling               -> input.token_pooling (cls | mean)
  normalize_embeddings        -> input.normalize_embeddings
  transformer.num_layers      -> model.transformer.num_layers
  transformer.d_model         -> model.transformer.d_model
  transformer.num_heads       -> model.transformer.num_heads
  transformer.mlp_hidden_dim  -> model.transformer.mlp_hidden_dim
  transformer.dropout         -> model.transformer.dropout
  transformer.attention_dropout -> model.transformer.attention_dropout
  transformer.pre_norm        -> model.transformer.pre_norm
  classifier.hidden_dim       -> model.classifier.hidden_dim
  classifier.dropout          -> model.classifier.dropout
  init.embedding_from_centroids -> model.init.embedding_from_centroids
  init.centroid_requires_grad -> model.init.centroid_requires_grad
  init.cls_init               -> model.init.cls_init
  init.weight_init            -> model.init.weight_init (xavier_uniform | kaiming_normal)

Assumptions:
  * The calling training script is responsible for providing num_labels
    (size of the glyph label set) and reading centroid vectors if needed.
  * Centroids (if used) must be float32 in [0,1], shape (vocab_size, 64) matching
    flattened 8x8 binary prototypes (row 0 = EMPTY).
  * If embedding_dim != centroid_dim (64), projection is performed automatically.

Limitations / Deferred:
  * No built-in masking / dropout of random tokens (could be added).
  * No gradient checkpointing (enable manually for deeper models if needed).
  * No relative positional encoding variant (future enhancement).
  * Attention heatmap extraction: expose last layer attn via forward(..., return_attn=True).

Usage (minimal):
  from models.phase2_transformer import build_phase2_model
  model = build_phase2_model(cfg_dict, num_labels=NUM_LABELS, primitive_centroids=centroids_array)

License: Follows project root license.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import math
import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------


class Sinusoidal2DPositionalEncoding(nn.Module):
    """
    2D extension of standard sine/cosine positional encoding.

    Generates a tensor of shape (H*W, D) where D must be even.
    Splits channels: first D/2 encode row, second D/2 encode column.

    If required_dim != model_dim and combine_mode='concat', caller will project.
    """

    def __init__(self, height: int, width: int, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Sinusoidal2DPositionalEncoding requires even dim.")
        self.height = height
        self.width = width
        self.dim = dim
        self.register_buffer("pe", self._build(), persistent=False)

    def _build(self) -> torch.Tensor:
        h, w, d = self.height, self.width, self.dim
        d_half = d // 2
        pe = torch.zeros(h, w, d)
        # Row encoding
        row_pos = torch.arange(h, dtype=torch.float32).unsqueeze(1)  # (h,1)
        col_pos = torch.arange(w, dtype=torch.float32).unsqueeze(1)  # (w,1)

        div_term_row = torch.exp(
            torch.arange(0, d_half, 2, dtype=torch.float32)
            * -(math.log(10_000.0) / d_half)
        )  # (d_half/2,)

        div_term_col = torch.exp(
            torch.arange(0, d_half, 2, dtype=torch.float32)
            * -(math.log(10_000.0) / d_half)
        )

        # Row (broadcast w)
        pe[:, :, 0:d_half:2] = torch.sin(row_pos * div_term_row).unsqueeze(1)
        pe[:, :, 1:d_half:2] = torch.cos(row_pos * div_term_row).unsqueeze(1)
        # Column (broadcast h)
        start = d_half
        pe[:, :, start : start + d_half : 2] = torch.sin(
            col_pos * div_term_col
        ).unsqueeze(0)
        pe[:, :, start + 1 : start + d_half : 2] = torch.cos(
            col_pos * div_term_col
        ).unsqueeze(0)
        return pe.view(h * w, d)  # (H*W, D)

    def forward(self, device=None) -> torch.Tensor:
        return self.pe.to(device=device)


class Learnable2DPositionalEncoding(nn.Module):
    """
    Learnable 2D positional encoding: (H*W, D) parameters.
    """

    def __init__(self, height: int, width: int, dim: int):
        super().__init__()
        self.height = height
        self.width = width
        self.dim = dim
        self.pe = nn.Parameter(torch.zeros(height * width, dim))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, device=None) -> torch.Tensor:
        return self.pe


# ---------------------------------------------------------------------------
# Transformer Building Blocks
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    def __init__(
        self, d_model: int, hidden: int, dropout: float, activation: str = "gelu"
    ):
        super().__init__()
        if activation not in {"gelu", "relu"}:
            raise ValueError("Unsupported activation: choose 'gelu' or 'relu'")
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        if self.activation == "gelu":
            x = F.gelu(x)
        else:
            x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Custom encoder layer (pre-norm optional). Exposes attention weights if requested.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        attn_dropout: float,
        layer_norm_eps: float = 1e-5,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=attn_dropout, batch_first=True
        )
        self.ff = FeedForward(d_model, dim_feedforward, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(
        self, x: torch.Tensor, attn_mask=None, key_padding_mask=None, need_weights=False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.pre_norm:
            # Pre-norm
            x_norm = self.norm1(x)
            attn_out, attn_w = self.self_attn(
                x_norm,
                x_norm,
                x_norm,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
            )
            x = x + self.dropout1(attn_out)
            x_norm2 = self.norm2(x)
            ff_out = self.ff(x_norm2)
            x = x + self.dropout2(ff_out)
        else:
            # Post-norm
            attn_out, attn_w = self.self_attn(
                x,
                x,
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
            )
            x = self.norm1(x + self.dropout1(attn_out))
            ff_out = self.ff(x)
            x = self.norm2(x + self.dropout2(ff_out))

        return x, (attn_w if need_weights else None)


# ---------------------------------------------------------------------------
# Phase 2 Model
# ---------------------------------------------------------------------------


@dataclass
class Phase2TransformerConfig:
    vocab_size: int
    embedding_dim: int
    d_model: int
    num_layers: int
    num_heads: int
    mlp_hidden_dim: int
    dropout: float
    attention_dropout: float
    layer_norm_eps: float
    pre_norm: bool
    use_cls_token: bool
    token_pooling: str  # 'cls' | 'mean'
    positional_encoding: str  # 'sinusoidal_2d' | 'learnable_2d'
    combine_mode: str  # 'add' | 'concat'
    patch_grouping: bool
    patch_rows: int
    patch_cols: int
    normalize_embeddings: bool
    classifier_hidden_dim: int
    classifier_dropout: float
    classifier_activation: str
    embedding_from_centroids: bool
    centroid_requires_grad: bool
    weight_init: str
    cls_init: str


class Phase2GlyphTransformer(nn.Module):
    """
    Transformer from primitive ID grid → glyph class logits.
    """

    def __init__(
        self,
        cfg: Phase2TransformerConfig,
        num_labels: int,
        primitive_centroids: Optional[
            torch.Tensor
        ] = None,  # (vocab_size, centroid_dim)
    ):
        super().__init__()
        self.cfg = cfg
        self.num_labels = num_labels

        if cfg.token_pooling not in {"cls", "mean"}:
            raise ValueError("token_pooling must be 'cls' or 'mean'")
        if cfg.combine_mode not in {"add", "concat"}:
            raise ValueError("combine_mode must be 'add' or 'concat'")

        # Embedding
        self.primitive_embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)

        # Optional centroid initialization
        if (
            cfg.embedding_from_centroids
            and primitive_centroids is not None
            and primitive_centroids.shape[0] == cfg.vocab_size
        ):
            with torch.no_grad():
                c = primitive_centroids
                if c.dtype != torch.float32:
                    c = c.float()
                # If centroid dim mismatch, project or slice
                if c.shape[1] != cfg.embedding_dim:
                    # Simple linear projection (centroids frozen or not depending on cfg)
                    proj = nn.Linear(c.shape[1], cfg.embedding_dim, bias=False)
                    nn.init.xavier_uniform_(proj.weight)
                    c = proj(c)
                self.primitive_embedding.weight.data.copy_(c)
            if not cfg.centroid_requires_grad:
                self.primitive_embedding.weight.requires_grad_(False)

        # Positional encoding
        if cfg.positional_encoding == "sinusoidal_2d":
            self.positional = Sinusoidal2DPositionalEncoding(16, 16, cfg.embedding_dim)
        elif cfg.positional_encoding == "learnable_2d":
            self.positional = Learnable2DPositionalEncoding(16, 16, cfg.embedding_dim)
        else:
            raise ValueError(
                f"Unsupported positional_encoding {cfg.positional_encoding}"
            )

        # If combine_mode == concat, increase model input dim
        combined_dim = (
            cfg.embedding_dim if cfg.combine_mode == "add" else cfg.embedding_dim * 2
        )

        # Project to d_model if needed
        if combined_dim != cfg.d_model:
            self.input_proj = nn.Linear(combined_dim, cfg.d_model)
        else:
            self.input_proj = nn.Identity()

        # CLS token
        self.use_cls = cfg.use_cls_token
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
            if cfg.cls_init == "normal":
                nn.init.trunc_normal_(self.cls_token, std=0.02)
            else:
                nn.init.zeros_(self.cls_token)

        # Patch grouping (optional)
        self.patch_grouping = cfg.patch_grouping
        if self.patch_grouping:
            # Precompute a pooling mapping by block averaging / flatten inside forward.
            if 16 % cfg.patch_rows != 0 or 16 % cfg.patch_cols != 0:
                raise ValueError("Patch rows/cols must divide 16.")
            self.group_rows = cfg.patch_rows
            self.group_cols = cfg.patch_cols
            self.group_factor = cfg.patch_rows * cfg.patch_cols
        else:
            self.group_rows = self.group_cols = 1
            self.group_factor = 1

        # Transformer layers
        layers = []
        for _ in range(cfg.num_layers):
            layers.append(
                TransformerEncoderLayer(
                    d_model=cfg.d_model,
                    nhead=cfg.num_heads,
                    dim_feedforward=cfg.mlp_hidden_dim,
                    dropout=cfg.dropout,
                    attn_dropout=cfg.attention_dropout,
                    layer_norm_eps=cfg.layer_norm_eps,
                    pre_norm=cfg.pre_norm,
                )
            )
        self.encoder = nn.ModuleList(layers)
        self.final_norm = (
            nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
            if cfg.pre_norm
            else nn.Identity()
        )

        # Classifier head
        if cfg.classifier_hidden_dim and cfg.classifier_hidden_dim != cfg.d_model:
            act = (
                nn.GELU()
                if cfg.classifier_activation == "gelu"
                else nn.ReLU(inplace=True)
            )
            self.classifier = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.classifier_hidden_dim),
                act,
                nn.Dropout(cfg.classifier_dropout),
                nn.Linear(cfg.classifier_hidden_dim, num_labels),
            )
        else:
            # Single linear
            self.classifier = nn.Sequential(
                nn.Dropout(cfg.classifier_dropout),
                nn.Linear(cfg.d_model, num_labels),
            )

        # Optional embedding normalization
        self.normalize_embeddings = cfg.normalize_embeddings

        # Weight init (for non-embedding layers)
        self._apply_weight_init(cfg.weight_init)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _apply_weight_init(self, strategy: str):
        for m in self.modules():
            if isinstance(m, (nn.Linear,)):
                if strategy == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight)
                elif strategy == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                else:
                    # Default PyTorch init fallback
                    pass
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _prepare_tokens(self, grid_ids: torch.Tensor) -> torch.Tensor:
        """
        grid_ids: (B,16,16) or (B,256)
        Returns token embeddings (B, T, d_model) before transformer layers.
        """
        if grid_ids.dim() == 3:
            B, H, W = grid_ids.shape
            if (H, W) != (16, 16):
                raise ValueError(f"Expected grid (16,16); got {(H, W)}")
            flat = grid_ids.view(B, H * W)  # (B,256)
        elif grid_ids.dim() == 2:
            B, N = grid_ids.shape
            if N != 256:
                raise ValueError("Flattened grid must have length 256 (16*16).")
            flat = grid_ids
        else:
            raise ValueError("grid_ids must have shape (B,16,16) or (B,256).")

        # Primitive embeddings
        emb = self.primitive_embedding(flat)  # (B,256,E)

        if self.normalize_embeddings:
            emb = F.normalize(emb, p=2, dim=-1)

        # Positional
        pos = self.positional(device=emb.device)  # (256,E)
        if self.cfg.combine_mode == "add":
            tokens = emb + pos.unsqueeze(0)
        else:  # concat
            tokens = torch.cat([emb, pos.unsqueeze(0).expand_as(emb)], dim=-1)

        tokens = self.input_proj(tokens)  # (B,256,d_model)

        # Patch grouping (pooling tokens into bigger spatial patches)
        if self.patch_grouping:
            B = tokens.size(0)
            d_model = tokens.size(-1)
            tokens_2d = tokens.view(B, 16, 16, d_model)
            pr, pc = self.group_rows, self.group_cols
            new_h = 16 // pr
            new_w = 16 // pc
            # Average pooling over patch blocks
            tokens = (
                tokens_2d.view(B, new_h, pr, new_w, pc, d_model)
                .mean(dim=(2, 4))
                .contiguous()
                .view(B, new_h * new_w, d_model)
            )
        return tokens  # (B,T,d_model)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        grid_ids: torch.Tensor,
        *,
        return_attn: bool = False,
    ):
        """
        grid_ids: (B,16,16) int64 primitive IDs (0..vocab_size-1)
        return_attn: if True, returns (logits, attn_list)
        """
        x = self._prepare_tokens(grid_ids)  # (B,T,d_model)
        B, T, D = x.shape

        attn_list = [] if return_attn else None

        if self.use_cls:
            cls_tok = self.cls_token.expand(B, -1, -1)  # (B,1,D)
            x = torch.cat([cls_tok, x], dim=1)  # (B,1+T,D)

        for layer in self.encoder:
            x, attn = layer(
                x, attn_mask=None, key_padding_mask=None, need_weights=return_attn
            )
            if return_attn:
                attn_list.append(attn)  # (B, heads, seq, seq) in PyTorch's shape

        x = self.final_norm(x)

        if self.cfg.token_pooling == "cls":
            if not self.use_cls:
                raise ValueError("CLS pooling requested but use_cls_token is False.")
            pooled = x[:, 0]  # (B,D)
        else:  # mean (exclude CLS if present)
            if self.use_cls:
                pooled = x[:, 1:].mean(dim=1)
            else:
                pooled = x.mean(dim=1)

        logits = self.classifier(pooled)

        if return_attn:
            return logits, attn_list
        return logits


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_phase2_model(
    cfg_dict: Dict[str, Any],
    num_labels: int,
    primitive_centroids: Optional["torch.Tensor | Any"] = None,
) -> Phase2GlyphTransformer:
    """
    Build a Phase2GlyphTransformer from a nested config dict (mirroring phase2.yaml).

    Parameters
    ----------
    cfg_dict : dict
        Parsed YAML (or subset) with 'input', 'model', etc.
    num_labels : int
        Number of glyph classes.
    primitive_centroids : torch.Tensor (optional)
        Centroid prototypes (vocab_size, centroid_dim=64). May be numpy -> auto-converted.

    Returns
    -------
    Phase2GlyphTransformer
    """
    input_cfg = cfg_dict.get("input", {})
    model_cfg = cfg_dict.get("model", {}).get("transformer", {})
    classifier_cfg = cfg_dict.get("model", {}).get("classifier", {})
    init_cfg = cfg_dict.get("model", {}).get("init", {})

    vocab_size = int(input_cfg.get("primitive_vocab_size", 1024))
    embedding_dim = int(input_cfg.get("embedding_dim", 64))
    positional_encoding = input_cfg.get("positional_encoding", "sinusoidal_2d")
    combine_mode = input_cfg.get("combine_mode", "add")
    patch_grouping_cfg = input_cfg.get("patch_grouping", {}) or {}
    patch_grouping = bool(patch_grouping_cfg.get("enabled", False))
    patch_rows = int(patch_grouping_cfg.get("patch_rows", 4))
    patch_cols = int(patch_grouping_cfg.get("patch_cols", 4))
    token_pooling = input_cfg.get("token_pooling", "cls")
    use_cls_token = bool(input_cfg.get("use_cls_token", True))
    normalize_embeddings = bool(input_cfg.get("normalize_embeddings", False))

    d_model = int(model_cfg.get("d_model", 256))
    num_layers = int(model_cfg.get("num_layers", 5))
    num_heads = int(model_cfg.get("num_heads", 8))
    mlp_hidden_dim = int(model_cfg.get("mlp_hidden_dim", 512))
    dropout = float(model_cfg.get("dropout", 0.1))
    attention_dropout = float(model_cfg.get("attention_dropout", 0.1))
    layer_norm_eps = float(model_cfg.get("layer_norm_eps", 1e-5))
    pre_norm = bool(model_cfg.get("pre_norm", True))

    classifier_hidden_dim = int(classifier_cfg.get("hidden_dim", d_model))
    classifier_dropout = float(classifier_cfg.get("dropout", 0.1))
    classifier_activation = classifier_cfg.get("activation", "gelu")

    embedding_from_centroids = bool(init_cfg.get("embedding_from_centroids", True))
    centroid_requires_grad = bool(init_cfg.get("centroid_requires_grad", True))
    weight_init = init_cfg.get("weight_init", "xavier_uniform")
    cls_init = init_cfg.get("cls_init", "normal")

    cfg = Phase2TransformerConfig(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_hidden_dim=mlp_hidden_dim,
        dropout=dropout,
        attention_dropout=attention_dropout,
        layer_norm_eps=layer_norm_eps,
        pre_norm=pre_norm,
        use_cls_token=use_cls_token,
        token_pooling=token_pooling,
        positional_encoding=positional_encoding,
        combine_mode=combine_mode,
        patch_grouping=patch_grouping,
        patch_rows=patch_rows,
        patch_cols=patch_cols,
        normalize_embeddings=normalize_embeddings,
        classifier_hidden_dim=classifier_hidden_dim,
        classifier_dropout=classifier_dropout,
        classifier_activation=classifier_activation,
        embedding_from_centroids=embedding_from_centroids,
        centroid_requires_grad=centroid_requires_grad,
        weight_init=weight_init,
        cls_init=cls_init,
    )

    # Convert centroids if numpy supplied
    if primitive_centroids is not None and not torch.is_tensor(primitive_centroids):
        primitive_centroids = torch.as_tensor(primitive_centroids, dtype=torch.float32)

    model = Phase2GlyphTransformer(
        cfg=cfg,
        num_labels=num_labels,
        primitive_centroids=primitive_centroids,
    )
    return model


__all__ = [
    "Phase2GlyphTransformer",
    "Phase2TransformerConfig",
    "build_phase2_model",
]
