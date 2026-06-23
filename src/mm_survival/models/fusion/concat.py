"""Concatenation fusion."""

from __future__ import annotations

import torch
from torch import nn


class ConcatFusion(nn.Module):
    """Concatenate RNA, clinical, and pathology embeddings."""

    def __init__(
        self,
        rna_dim: int = 256,
        clinical_dim: int = 128,
        pathology_dim: int = 256,
    ):
        super().__init__()
        self.output_dim = rna_dim + clinical_dim + pathology_dim

    def forward(
        self,
        rna_emb: torch.Tensor,
        clinical_emb: torch.Tensor,
        pathology_emb: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([rna_emb, clinical_emb, pathology_emb], dim=-1)
