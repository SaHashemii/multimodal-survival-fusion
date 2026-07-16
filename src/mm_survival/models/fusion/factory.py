"""
Factory for multimodal fusion modules
=====================================

Maps the fusion name in an experiment YAML file to the corresponding PyTorch
module.

Supported names
---------------
  concat
  gated_concat
  lowrank_bilinear

Design rationale
----------------
* Training scripts should not need fusion-specific construction logic.
* Adding a new fusion method only requires registering the module here and
  defining its config fields in the experiment YAML.
"""

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

    # Common modality dimensions come from the encoders; method-specific values
    # such as rank or interaction output dims come from config.
    return cls(
        rna_dim=rna_dim,
        clinical_dim=clinical_dim,
        pathology_dim=pathology_dim,
        **config,
    )


def list_fusion_methods() -> list[str]:
    """Return available fusion method names."""
    return list(FUSION_REGISTRY)
