"""Scalar-gated concatenation fusion."""

from __future__ import annotations

import torch
from torch import nn


class ScalarModalityGate(nn.Module):
    """Learn one scalar gate for a modality embedding."""

    def __init__(self, in_dim: int):
        super().__init__()
        hidden_dim = max(1, in_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = self.net(x)
        gated_x = x * gate
        return gated_x, gate.squeeze(-1)


class ModalityGatedConcatFusion(nn.Module):
    """Scalar-gated concat fusion for RNA, clinical, and pathology embeddings."""

    def __init__(
        self,
        rna_dim: int = 256,
        clinical_dim: int = 128,
        pathology_dim: int = 256,
    ):
        super().__init__()
        self.rna_gate = ScalarModalityGate(rna_dim)
        self.clinical_gate = ScalarModalityGate(clinical_dim)
        self.pathology_gate = ScalarModalityGate(pathology_dim)
        self.output_dim = rna_dim + clinical_dim + pathology_dim

    def forward(
        self,
        rna_emb: torch.Tensor,
        clinical_emb: torch.Tensor,
        pathology_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        rna_gated, rna_gate = self.rna_gate(rna_emb)
        clinical_gated, clinical_gate = self.clinical_gate(clinical_emb)
        pathology_gated, pathology_gate = self.pathology_gate(pathology_emb)

        fused = torch.cat([rna_gated, clinical_gated, pathology_gated], dim=-1)
        gates = {
            "rna": rna_gate,
            "clinical": clinical_gate,
            "pathology": pathology_gate,
        }
        return fused, gates
