"""
Low-rank pairwise bilinear fusion
=================================

Builds pairwise interactions between modality embeddings using low-rank
bilinear blocks, then concatenates the interaction features.

Trimodal interaction blocks
--------------------------
  pathology × RNA       → pathology_rna feature
  pathology × clinical  → pathology_clinical feature
  RNA × clinical        → rna_clinical feature

Missing RNA behavior
--------------------
When rna_mask = 0, RNA-dependent interaction blocks are explicitly zeroed:

  pathology × RNA = 0
  RNA × clinical  = 0

The pathology × clinical block remains active. Explicit masking is needed
because Linear layers have bias terms, so passing a zero RNA vector through the
projection layers does not guarantee a zero interaction output.
"""

from __future__ import annotations

import torch
from torch import nn


class LowRankBilinearFusion(nn.Module):
    """Low-rank bilinear interaction between two modality embeddings."""

    def __init__(self, dim1: int, dim2: int, rank: int, out_dim: int):
        super().__init__()
        self.proj1 = nn.Linear(dim1, rank)
        self.proj2 = nn.Linear(dim2, rank)
        self.out = nn.Linear(rank, out_dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:

        # Project both modalities to a shared low-rank space, multiply them
        # elementwise, then project the interaction to the requested output dim.
        interaction = self.proj1(x1) * self.proj2(x2)
        return self.out(interaction)


class PairwiseLowRankBilinearFusion(nn.Module):
    """Pairwise low-rank bilinear fusion for pathology, RNA, and clinical embeddings."""

    def __init__(
        self,
        pathology_dim: int = 256,
        rna_dim: int = 256,
        clinical_dim: int = 128,
        rank: int = 64,
        pathology_rna_out_dim: int = 256,
        pathology_clinical_out_dim: int = 192,
        rna_clinical_out_dim: int = 192,
    ):
        super().__init__()
        self.pathology_rna = LowRankBilinearFusion(pathology_dim, rna_dim, rank, out_dim=pathology_rna_out_dim)
        self.pathology_clinical = LowRankBilinearFusion(
            pathology_dim,
            clinical_dim,
            rank,
            out_dim=pathology_clinical_out_dim,
        )
        self.rna_clinical = LowRankBilinearFusion(rna_dim, clinical_dim, rank, out_dim=rna_clinical_out_dim)
        self.output_dim = pathology_rna_out_dim + pathology_clinical_out_dim + rna_clinical_out_dim

    def forward(
        self,
        rna_emb: torch.Tensor,
        clinical_emb: torch.Tensor,
        pathology_emb: torch.Tensor,
        rna_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pathology_rna = self.pathology_rna(pathology_emb, rna_emb)
        pathology_clinical = self.pathology_clinical(pathology_emb, clinical_emb)
        rna_clinical = self.rna_clinical(rna_emb, clinical_emb)
        if rna_mask is not None:

            # Keep pathology-clinical information available while removing only
            # the interaction terms that depend on unavailable RNA.
            mask = rna_mask.reshape(-1, 1).to(device=rna_emb.device, dtype=rna_emb.dtype)
            pathology_rna = pathology_rna * mask
            rna_clinical = rna_clinical * mask
        return torch.cat([pathology_rna, pathology_clinical, rna_clinical], dim=-1)
