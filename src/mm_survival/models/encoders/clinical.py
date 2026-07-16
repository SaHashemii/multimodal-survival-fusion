"""
Clinical encoders for tabular covariates and token embeddings
=============================================================

Transforms clinical inputs into one fixed-length patient-level embedding.

Supported clinical inputs
-------------------------
  tabular covariates:
    preprocessed feature vector → MLP → clinical embedding

  frozen text/token embeddings:
    [tokens, dim] embeddings → token projection → flatten/pool → clinical embedding

Design rationale
----------------
* Tabular clinical data are already patient-level features.
* BERT/CONCH clinical embeddings are token-level features, so each token is
  projected to a lower dimension before all token features are combined.
* Zero-padded clinical tokens are masked so padding does not contribute to the
  patient representation.
"""

from __future__ import annotations

import torch
from torch import nn


class ClinicalEncoder(nn.Module):
    """MLP encoder for tabular clinical covariates."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        emb_dim: int,
        dropout: float,
        activation: str = "selu",
    ):
        super().__init__()
        drop_cls = nn.AlphaDropout if activation == "selu" else nn.Dropout
        act_cls = nn.SELU if activation == "selu" else nn.ReLU

        layers: list[nn.Module] = []
        prev = in_dim
        for hidden_dim in hidden_dims:

            # Tabular features are encoded by a standard MLP before being passed
            # to fusion or the unimodal Cox head.
            layers.extend([nn.Linear(prev, hidden_dim), nn.LayerNorm(hidden_dim), act_cls(), drop_cls(dropout)])
            prev = hidden_dim
        layers.extend([nn.Linear(prev, emb_dim), act_cls(), drop_cls(dropout)])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClinicalEmbeddingEncoder(nn.Module):
    """Encoder for frozen clinical token embeddings."""

    def __init__(
        self,
        token_count: int,
        in_dim: int,
        token_hidden_dim: int,
        token_out_dim: int,
        emb_dim: int,
        dropout: float,
        activation: str = "selu",
    ):
        super().__init__()
        drop_cls = nn.AlphaDropout if activation == "selu" else nn.Dropout
        act_cls = nn.SELU if activation == "selu" else nn.ReLU

        self.token_net = nn.Sequential(
            nn.Linear(in_dim, token_hidden_dim),
            nn.LayerNorm(token_hidden_dim),
            act_cls(),
            drop_cls(dropout),
            nn.Linear(token_hidden_dim, token_out_dim),
            nn.LayerNorm(token_out_dim),
            act_cls(),
            drop_cls(dropout),
        )
        self.fused_net = nn.Sequential(
            nn.Linear(token_count * token_out_dim, emb_dim),
            act_cls(),
            drop_cls(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"ClinicalEmbeddingEncoder expects [batch, tokens, dim], got {tuple(x.shape)}")

        # Padding tokens are all-zero rows. Mask them after token projection so
        # shorter clinical prompts do not add artificial signal when flattened.
        mask = (x.abs().sum(dim=-1, keepdim=True) > 0).float()
        token_emb = self.token_net(x) * mask

        # Original clinical-embedding design: project each token to a lower
        # dimension, flatten all projected tokens, then compress to emb_dim.
        return self.fused_net(token_emb.flatten(start_dim=1))


class ClinicalEmbeddingPoolingEncoder(nn.Module):
    """Legacy clinical embedding pooling used by the original unimodal baseline."""

    def __init__(
        self,
        token_count: int,
        in_dim: int,
        pooling: str,
        attention_hidden_dim: int,
        projection_dim: int,
        dropout: float,
    ):
        super().__init__()
        if pooling == "auto":

            # For a single token, mean pooling and flattening are equivalent in
            # shape; for multiple tokens, mean is the simpler default.
            pooling = "mean" if token_count > 1 else "flatten"
        if pooling == "attention" and token_count == 1:
            pooling = "mean"
        if projection_dim < 4:
            raise ValueError("projection_dim must be at least 4.")

        self.pooling = pooling
        self.attention = None
        self.feature_net = None

        if pooling == "attention":

            # Attention pooling learns token weights before summing token
            # embeddings into one patient-level clinical representation.
            self.attention = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, attention_hidden_dim),
                nn.Tanh(),
                nn.Linear(attention_hidden_dim, 1, bias=False),
            )
            self.output_dim = in_dim
        elif pooling == "mean":
            self.output_dim = in_dim
        elif pooling == "flatten":
            self.output_dim = token_count * in_dim
        elif pooling == "project-concat":

            # Legacy project-concat path mirrors the older clinical unimodal
            # model: project token embeddings down, then concatenate tokens.
            self.feature_net = nn.Sequential(
                nn.Linear(in_dim, projection_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(projection_dim // 2, projection_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.output_dim = token_count * (projection_dim // 4)
        else:
            raise ValueError(f"Unsupported clinical embedding pooling: {pooling}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"ClinicalEmbeddingPoolingEncoder expects [batch, tokens, dim], got {tuple(x.shape)}")
        mask = (x.abs().sum(dim=-1, keepdim=True) > 0).float()
        if self.pooling == "attention":
            logits = self.attention(x).squeeze(-1)

            # Mask padded tokens before softmax so they receive zero attention.
            logits = logits.masked_fill(mask.squeeze(-1) == 0, torch.finfo(logits.dtype).min)
            weights = torch.softmax(logits, dim=1)
            return torch.sum(weights.unsqueeze(-1) * x, dim=1)
        if self.pooling == "mean":
            denom = mask.sum(dim=1).clamp_min(1.0)
            return (x * mask).sum(dim=1) / denom
        if self.pooling == "project-concat":
            token_features = self.feature_net(x) * mask
            return token_features.flatten(start_dim=1)
        return x.flatten(start_dim=1)
