"""Low-rank pairwise bilinear fusion."""

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
    ) -> torch.Tensor:
        pathology_rna = self.pathology_rna(pathology_emb, rna_emb)
        pathology_clinical = self.pathology_clinical(pathology_emb, clinical_emb)
        rna_clinical = self.rna_clinical(rna_emb, clinical_emb)
        return torch.cat([pathology_rna, pathology_clinical, rna_clinical], dim=-1)
