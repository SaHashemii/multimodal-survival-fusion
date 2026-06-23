"""Factory for multimodal fusion modules."""

from __future__ import annotations

from typing import Any

from torch import nn

from .concat import ConcatFusion
from .gated import ModalityGatedConcatFusion
from .lowrank_bilinear import PairwiseLowRankBilinearFusion


FUSION_REGISTRY = {
    "concat": ConcatFusion,
    "gated_concat": ModalityGatedConcatFusion,
    "lowrank_bilinear": PairwiseLowRankBilinearFusion,
}


def build_fusion(
    name: str,
    rna_dim: int,
    clinical_dim: int,
    pathology_dim: int,
    config: dict[str, Any] | None = None,
) -> nn.Module:
    """Build a fusion module by registry name."""
    config = dict(config or {})
    key = name.lower().strip()
    if key not in FUSION_REGISTRY:
        raise ValueError(f"Unknown fusion method '{name}'. Choose from: {list(FUSION_REGISTRY)}")
    cls = FUSION_REGISTRY[key]
    return cls(
        rna_dim=rna_dim,
        clinical_dim=clinical_dim,
        pathology_dim=pathology_dim,
        **config,
    )


def list_fusion_methods() -> list[str]:
    """Return available fusion method names."""
    return list(FUSION_REGISTRY)
