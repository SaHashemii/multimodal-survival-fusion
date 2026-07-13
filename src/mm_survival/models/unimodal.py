"""Unimodal Cox model wrappers."""

from __future__ import annotations

import torch
from torch import nn

from mm_survival.models.encoders.clinical import ClinicalEmbeddingEncoder, ClinicalEmbeddingPoolingEncoder, ClinicalEncoder
from mm_survival.models.encoders.pathology import HEEncoder
from mm_survival.models.heads import CoxHead


class RNACoxModel(nn.Module):
    """RNA-only Cox model."""

    def __init__(
        self,
        rna_extractor: nn.Module,
        head_hidden_dims: list[int] | None = None,
        head_dropout: float = 0.30,
        head_activation: str = "selu",
    ):
        super().__init__()
        self.rna_extractor = rna_extractor
        self.head = CoxHead(
            in_dim=rna_extractor.output_dim,
            hidden_dims=head_hidden_dims or [256, 64],
            dropout=head_dropout,
            activation=head_activation,
        )

    def forward_all(self, rna: torch.Tensor) -> torch.Tensor:
        return self.head(self.rna_extractor(rna))

    def forward(self, rna: torch.Tensor) -> torch.Tensor:
        return self.forward_all(rna)


class ClinicalCoxModel(nn.Module):
    """Clinical-only Cox model for tabular or embedding inputs."""

    def __init__(
        self,
        clinical_source: str,
        clinical_dim: int,
        clinical_token_count: int | None = None,
        clinical_hidden_dims: list[int] | None = None,
        clinical_emb_dim: int = 128,
        clinical_token_hidden_dim: int = 256,
        clinical_token_out_dim: int = 128,
        clinical_dropout: float = 0.30,
        clinical_activation: str = "selu",
        clinical_pooling: str | None = None,
        clinical_projection_dim: int = 512,
        clinical_attention_hidden_dim: int = 128,
        head_hidden_dims: list[int] | None = None,
        head_dropout: float = 0.30,
        head_activation: str = "selu",
    ):
        super().__init__()
        clinical_hidden_dims = clinical_hidden_dims or [128]
        if clinical_source == "embedding":
            if clinical_token_count is None:
                raise ValueError("clinical_token_count is required when clinical_source='embedding'.")
            if clinical_pooling is None:
                self.encoder = ClinicalEmbeddingEncoder(
                    token_count=clinical_token_count,
                    in_dim=clinical_dim,
                    token_hidden_dim=clinical_token_hidden_dim,
                    token_out_dim=clinical_token_out_dim,
                    emb_dim=clinical_emb_dim,
                    dropout=clinical_dropout,
                    activation=clinical_activation,
                )
                head_in_dim = clinical_emb_dim
            else:
                self.encoder = ClinicalEmbeddingPoolingEncoder(
                    token_count=clinical_token_count,
                    in_dim=clinical_dim,
                    pooling=clinical_pooling,
                    attention_hidden_dim=clinical_attention_hidden_dim,
                    projection_dim=clinical_projection_dim,
                    dropout=clinical_dropout,
                )
                head_in_dim = self.encoder.output_dim
        elif clinical_source == "tabular":
            self.encoder = ClinicalEncoder(
                in_dim=clinical_dim,
                hidden_dims=clinical_hidden_dims,
                emb_dim=clinical_emb_dim,
                dropout=clinical_dropout,
                activation=clinical_activation,
            )
            head_in_dim = clinical_emb_dim
        else:
            raise ValueError(f"Unknown clinical_source: {clinical_source}")

        self.head = CoxHead(
            in_dim=head_in_dim,
            hidden_dims=head_hidden_dims or [],
            dropout=head_dropout,
            activation=head_activation,
        )

    def forward_all(self, clinical: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(clinical))

    def forward(self, clinical: torch.Tensor) -> torch.Tensor:
        return self.forward_all(clinical)


class PathologyCoxModel(nn.Module):
    """Pathology-only Cox model using MIL aggregation over tile features."""

    def __init__(
        self,
        pathology_in_dim: int,
        pathology_emb_dim: int = 256,
        pathology_aggregator: str = "gated",
        pathology_attn_dim: int = 128,
        pathology_dropout: float = 0.20,
        head_hidden_dims: list[int] | None = None,
        head_dropout: float = 0.30,
        head_activation: str = "selu",
    ):
        super().__init__()
        self.encoder = HEEncoder(
            in_dim=pathology_in_dim,
            emb_dim=pathology_emb_dim,
            aggregator=pathology_aggregator,
            attn_dim=pathology_attn_dim,
            dropout=pathology_dropout,
        )
        self.head = CoxHead(
            in_dim=pathology_emb_dim,
            hidden_dims=head_hidden_dims or [256, 64],
            dropout=head_dropout,
            activation=head_activation,
        )

    def forward_all(
        self,
        pathology_bags: list[torch.Tensor],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        risks = []
        for bag in pathology_bags:
            if device is not None:
                bag = bag.to(device)
            embedding, _ = self.encoder(bag)
            risks.append(self.head(embedding))
        return torch.stack(risks)

    def forward(self, pathology_bags: list[torch.Tensor]) -> torch.Tensor:
        return self.forward_all(pathology_bags)


class PathologySlideCoxModel(nn.Module):
    """Pathology-only Cox model for slide-level embeddings such as PRISM."""

    def __init__(
        self,
        pathology_in_dim: int,
        pathology_hidden_dims: list[int] | None = None,
        pathology_emb_dim: int = 256,
        pathology_dropout: float = 0.30,
        pathology_activation: str = "selu",
        head_hidden_dims: list[int] | None = None,
        head_dropout: float = 0.30,
        head_activation: str = "selu",
    ):
        super().__init__()
        pathology_hidden_dims = pathology_hidden_dims or [512]
        drop_cls = nn.AlphaDropout if pathology_activation == "selu" else nn.Dropout
        act_cls = nn.SELU if pathology_activation == "selu" else nn.ReLU

        layers: list[nn.Module] = [nn.LayerNorm(pathology_in_dim), drop_cls(pathology_dropout)]
        prev = pathology_in_dim
        for hidden_dim in pathology_hidden_dims:
            layers.extend([nn.Linear(prev, hidden_dim), nn.LayerNorm(hidden_dim), act_cls(), drop_cls(pathology_dropout)])
            prev = hidden_dim
        layers.extend([nn.Linear(prev, pathology_emb_dim), nn.LayerNorm(pathology_emb_dim), act_cls(), drop_cls(pathology_dropout)])
        self.encoder = nn.Sequential(*layers)
        self.head = CoxHead(
            in_dim=pathology_emb_dim,
            hidden_dims=head_hidden_dims or [256, 64],
            dropout=head_dropout,
            activation=head_activation,
        )

    def forward_all(self, pathology: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(pathology))

    def forward(self, pathology: torch.Tensor) -> torch.Tensor:
        return self.forward_all(pathology)
