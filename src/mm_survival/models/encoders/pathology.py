"""Pathology feature encoders."""

from __future__ import annotations

import torch
from torch import nn


class GatedAttentionMIL(nn.Module):
    """Gated attention pooling for tile-level pathology features."""

    def __init__(self, in_dim: int = 1024, attn_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.v = nn.Linear(in_dim, attn_dim)
        self.u = nn.Linear(in_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if h.ndim != 2:
            raise ValueError(f"GatedAttentionMIL expects (tiles, dim), got {tuple(h.shape)}")
        h_norm = self.drop(self.norm(h.float()))
        logits = self.w(torch.tanh(self.v(h_norm)) * torch.sigmoid(self.u(h_norm))).squeeze(-1)
        weights = torch.softmax(logits - logits.max(), dim=0)
        pooled = torch.sum(weights.unsqueeze(1) * h_norm, dim=0)
        return pooled, weights


class HEEncoder(nn.Module):
    """Encode pathology tile features into one sample-level embedding."""

    def __init__(
        self,
        in_dim: int,
        emb_dim: int,
        aggregator: str = "gated",
        attn_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.aggregator = aggregator
        if aggregator == "gated":
            self.pool = GatedAttentionMIL(in_dim=in_dim, attn_dim=attn_dim, dropout=dropout)
            proj_in = in_dim
        elif aggregator == "mean":
            self.pool = None
            proj_in = in_dim
        elif aggregator == "mean+std":
            self.pool = None
            proj_in = in_dim * 2
        else:
            raise ValueError(f"Unknown HE aggregator: {aggregator}")

        self.proj = nn.Sequential(
            nn.LayerNorm(proj_in),
            nn.Linear(proj_in, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.aggregator == "gated":
            pooled, weights = self.pool(h)
        elif self.aggregator == "mean":
            pooled = h.float().mean(dim=0)
            weights = None
        else:
            pooled = torch.cat([h.float().mean(dim=0), h.float().std(dim=0, unbiased=False)], dim=0)
            weights = None
        return self.proj(pooled), weights
