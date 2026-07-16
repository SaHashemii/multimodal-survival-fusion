"""
Embedding-based trimodal Cox model
==================================

Wraps the modality-specific encoders, fusion module, and Cox survival head for
RNA + clinical + pathology experiments.

Pipeline
--------
  RNA matrix          → RNA extractor              → RNA embedding
  clinical input      → clinical encoder           → clinical embedding
  pathology tile bags → pathology MIL encoder      → pathology embedding
                                                        ↓
                                 concat/gated/low-rank fusion
                                                        ↓
                                             Cox head log-risk

Design rationale
----------------
* The fusion module is selected from the experiment config.
* Clinical inputs can be tabular covariates or token embeddings.
* Pathology bags can have different tile counts, so they are encoded one sample
  at a time before stacking sample-level embeddings.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from mm_survival.models.encoders.clinical import ClinicalEmbeddingEncoder, ClinicalEncoder
from mm_survival.models.encoders.pathology import HEEncoder
from mm_survival.models.fusion import build_fusion
from mm_survival.models.heads import CoxHead


class MultimodalCoxModel(nn.Module):
    """RNA + clinical + pathology Cox model with configurable fusion.

    This model covers the embedding-based fusion methods:
    ``concat``, ``gated_concat``, and ``lowrank_bilinear``.
    """

    def __init__(
        self,
        rna_extractor: nn.Module,
        clinical_source: str,
        clinical_dim: int,
        pathology_in_dim: int,
        clinical_token_count: int | None = None,
        clinical_hidden_dims: list[int] | None = None,
        clinical_emb_dim: int = 128,
        clinical_token_hidden_dim: int = 256,
        clinical_token_out_dim: int = 128,
        clinical_dropout: float = 0.20,
        clinical_activation: str = "selu",
        pathology_emb_dim: int = 256,
        pathology_aggregator: str = "gated",
        pathology_attn_dim: int = 128,
        pathology_dropout: float = 0.20,
        fusion_name: str = "concat",
        fusion_config: dict[str, Any] | None = None,
        head_hidden_dims: list[int] | None = None,
        head_dropout: float = 0.30,
        head_activation: str = "selu",
    ):
        super().__init__()
        self.rna_extractor = rna_extractor
        self.clinical_source = clinical_source

        clinical_hidden_dims = clinical_hidden_dims or [256, 128]
        if clinical_source == "embedding":
            if clinical_token_count is None:
                raise ValueError("clinical_token_count is required when clinical_source='embedding'.")

            # Clinical embeddings are first projected token-wise, flattened, and
            # then compressed to one patient-level clinical embedding.
            self.clinical_encoder = ClinicalEmbeddingEncoder(
                token_count=clinical_token_count,
                in_dim=clinical_dim,
                token_hidden_dim=clinical_token_hidden_dim,
                token_out_dim=clinical_token_out_dim,
                emb_dim=clinical_emb_dim,
                dropout=clinical_dropout,
                activation=clinical_activation,
            )
        elif clinical_source == "tabular":

            # Tabular clinical features are already sample-level vectors, so an
            # MLP encoder maps them directly to the shared clinical embedding.
            self.clinical_encoder = ClinicalEncoder(
                in_dim=clinical_dim,
                hidden_dims=clinical_hidden_dims,
                emb_dim=clinical_emb_dim,
                dropout=clinical_dropout,
                activation=clinical_activation,
            )
        else:
            raise ValueError(f"Unknown clinical_source: {clinical_source}")

        self.pathology_encoder = HEEncoder(
            in_dim=pathology_in_dim,
            emb_dim=pathology_emb_dim,
            aggregator=pathology_aggregator,
            attn_dim=pathology_attn_dim,
            dropout=pathology_dropout,
        )
        self.fusion = build_fusion(
            name=fusion_name,
            rna_dim=rna_extractor.output_dim,
            clinical_dim=clinical_emb_dim,
            pathology_dim=pathology_emb_dim,
            config=fusion_config,
        )
        self.head = CoxHead(
            in_dim=self.fusion.output_dim,
            hidden_dims=head_hidden_dims or [256, 64],
            dropout=head_dropout,
            activation=head_activation,
        )

    def encode_modalities(
        self,
        rna: torch.Tensor,
        clinical: torch.Tensor,
        pathology_bags: list[torch.Tensor],
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode each modality into aligned sample-level embeddings."""
        device = device or rna.device
        rna_emb = self.rna_extractor(rna)
        clinical_emb = self.clinical_encoder(clinical)
        pathology_embs = []
        for bag in pathology_bags:

            # Pathology bags are processed independently because each patient
            # can have a different number of tile features.
            pathology_emb, _ = self.pathology_encoder(bag.to(device))
            pathology_embs.append(pathology_emb)
        pathology_emb = torch.stack(pathology_embs)
        return rna_emb, clinical_emb, pathology_emb

    def fuse(
        self,
        rna_emb: torch.Tensor,
        clinical_emb: torch.Tensor,
        pathology_emb: torch.Tensor,
        rna_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse modality embeddings, ignoring optional interpretability payloads."""
        fused = self.fusion(rna_emb, clinical_emb, pathology_emb, rna_mask=rna_mask)

        # Some fusion modules return auxiliary outputs such as modality gates;
        # the Cox head receives only the fused patient representation.
        if isinstance(fused, tuple):
            fused = fused[0]
        return fused

    def forward_all(
        self,
        rna: torch.Tensor,
        clinical: torch.Tensor,
        pathology_bags: list[torch.Tensor],
        device: torch.device | None = None,
        rna_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one log-risk score per sample."""
        rna_emb, clinical_emb, pathology_emb = self.encode_modalities(rna, clinical, pathology_bags, device=device)
        fused = self.fuse(rna_emb, clinical_emb, pathology_emb, rna_mask=rna_mask)
        return self.head(fused)

    def forward(
        self,
        rna: torch.Tensor,
        clinical: torch.Tensor,
        pathology_bags: list[torch.Tensor],
    ) -> torch.Tensor:
        return self.forward_all(rna, clinical, pathology_bags)
