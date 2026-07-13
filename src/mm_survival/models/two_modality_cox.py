"""Two-modality Cox models for concat ablation experiments."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from mm_survival.models.encoders.clinical import ClinicalEmbeddingEncoder, ClinicalEncoder
from mm_survival.models.encoders.pathology import HEEncoder
from mm_survival.models.fusion.gated import ScalarModalityGate
from mm_survival.models.fusion.lowrank_bilinear import LowRankBilinearFusion
from mm_survival.models.heads import CoxHead


class TwoModalityConcatCoxModel(nn.Module):
    """Cox model that concatenates any two selected modality embeddings."""

    def __init__(
        self,
        *,
        modalities: Iterable[str],
        rna_extractor: nn.Module | None,
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
        head_hidden_dims: list[int] | None = None,
        head_dropout: float = 0.30,
        head_activation: str = "selu",
    ):
        super().__init__()
        self.modalities = tuple(modalities)
        valid = {"rna", "clinical", "pathology"}
        unknown = sorted(set(self.modalities) - valid)
        if unknown:
            raise ValueError(f"Unknown modalities: {unknown}. Expected values from {sorted(valid)}")
        if len(self.modalities) != 2:
            raise ValueError(f"TwoModalityConcatCoxModel expects exactly two modalities, got {self.modalities}")

        self.rna_extractor = None
        self.clinical_encoder = None
        self.pathology_encoder = None
        output_dims: dict[str, int] = {}

        if "rna" in self.modalities:
            if rna_extractor is None:
                raise ValueError("rna_extractor is required when using RNA.")
            self.rna_extractor = rna_extractor
            output_dims["rna"] = int(rna_extractor.output_dim)

        if "clinical" in self.modalities:
            clinical_hidden_dims = clinical_hidden_dims or [256, 128]
            if clinical_source == "embedding":
                if clinical_token_count is None:
                    raise ValueError("clinical_token_count is required when clinical_source='embedding'.")
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
                self.clinical_encoder = ClinicalEncoder(
                    in_dim=clinical_dim,
                    hidden_dims=clinical_hidden_dims,
                    emb_dim=clinical_emb_dim,
                    dropout=clinical_dropout,
                    activation=clinical_activation,
                )
            else:
                raise ValueError(f"Unknown clinical_source: {clinical_source}")
            output_dims["clinical"] = clinical_emb_dim

        if "pathology" in self.modalities:
            self.pathology_encoder = HEEncoder(
                in_dim=pathology_in_dim,
                emb_dim=pathology_emb_dim,
                aggregator=pathology_aggregator,
                attn_dim=pathology_attn_dim,
                dropout=pathology_dropout,
            )
            output_dims["pathology"] = pathology_emb_dim

        self.output_dims = output_dims
        self.output_dim = sum(output_dims[name] for name in self.modalities)
        self.head = CoxHead(
            in_dim=self.output_dim,
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
    ) -> dict[str, torch.Tensor]:
        device = device or rna.device
        embeddings: dict[str, torch.Tensor] = {}
        if "rna" in self.modalities:
            embeddings["rna"] = self.rna_extractor(rna)
        if "clinical" in self.modalities:
            embeddings["clinical"] = self.clinical_encoder(clinical)
        if "pathology" in self.modalities:
            pathology_embs = []
            for bag in pathology_bags:
                pathology_emb, _ = self.pathology_encoder(bag.to(device))
                pathology_embs.append(pathology_emb)
            embeddings["pathology"] = torch.stack(pathology_embs)
        return embeddings

    def forward_all(
        self,
        rna: torch.Tensor,
        clinical: torch.Tensor,
        pathology_bags: list[torch.Tensor],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        embeddings = self.encode_modalities(rna, clinical, pathology_bags, device=device)
        pieces = [embeddings[name] for name in self.modalities]
        return self.head(torch.cat(pieces, dim=-1))

    def forward(
        self,
        rna: torch.Tensor,
        clinical: torch.Tensor,
        pathology_bags: list[torch.Tensor],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        return self.forward_all(rna, clinical, pathology_bags, device=device)


class TwoModalityGatedCoxModel(TwoModalityConcatCoxModel):
    """Scalar-gated concat Cox model for any two selected modalities."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gates = nn.ModuleDict(
            {name: ScalarModalityGate(self.output_dims[name]) for name in self.modalities}
        )

    def forward_all(
        self,
        rna: torch.Tensor,
        clinical: torch.Tensor,
        pathology_bags: list[torch.Tensor],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        embeddings = self.encode_modalities(rna, clinical, pathology_bags, device=device)
        gated_pieces = []
        for name in self.modalities:
            gated_embedding, _ = self.gates[name](embeddings[name])
            gated_pieces.append(gated_embedding)
        return self.head(torch.cat(gated_pieces, dim=-1))


class TwoModalityLowRankBilinearCoxModel(TwoModalityConcatCoxModel):
    """Low-rank bilinear Cox model for any two selected modality embeddings."""

    def __init__(
        self,
        *,
        fusion_rank: int = 64,
        fusion_out_dim: int = 64,
        head_hidden_dims: list[int] | None = None,
        head_dropout: float = 0.30,
        head_activation: str = "selu",
        **kwargs,
    ):
        super().__init__(
            head_hidden_dims=[],
            head_dropout=head_dropout,
            head_activation=head_activation,
            **kwargs,
        )
        dim1 = self.output_dims[self.modalities[0]]
        dim2 = self.output_dims[self.modalities[1]]
        self.fusion = LowRankBilinearFusion(dim1, dim2, rank=fusion_rank, out_dim=fusion_out_dim)
        self.output_dim = fusion_out_dim
        self.head = CoxHead(
            in_dim=self.output_dim,
            hidden_dims=head_hidden_dims or [],
            dropout=head_dropout,
            activation=head_activation,
        )

    def forward_all(
        self,
        rna: torch.Tensor,
        clinical: torch.Tensor,
        pathology_bags: list[torch.Tensor],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        embeddings = self.encode_modalities(rna, clinical, pathology_bags, device=device)
        fused = self.fusion(embeddings[self.modalities[0]], embeddings[self.modalities[1]])
        return self.head(fused)
