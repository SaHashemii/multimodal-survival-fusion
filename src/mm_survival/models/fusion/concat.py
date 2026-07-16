"""
Concatenation fusion
====================

The simplest embedding-based fusion strategy: concatenate RNA, clinical, and
pathology patient-level embeddings and pass the result to the Cox head.

Missing RNA behavior
--------------------
When rna_mask is provided, RNA embeddings are multiplied by the mask before
concatenation. This keeps the fused representation dimension unchanged while
removing RNA information for patients with missing RNA.
"""

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
        rna_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if rna_mask is not None:

            # Shape [B] mask is expanded to [B, 1] so it zeros each patient's
            # full RNA embedding without changing the fused feature dimension.
            rna_emb = rna_emb * rna_mask.reshape(-1, 1).to(device=rna_emb.device, dtype=rna_emb.dtype)
        return torch.cat([rna_emb, clinical_emb, pathology_emb], dim=-1)
