"""
Factory for RNA extractors.
Instantiate the right extractor from a config dict.

Usage
-----
    from rna_extractors import build_rna_extractor

    extractor = build_rna_extractor(
        strategy="variance_filter",
        input_dim=20531,
        config={
            "top_k": 2000,
            "out_dim": 256,
            "hidden_dims": [512, 256],
        }
    )
    extractor.fit(x_train)          # only needed for variance_filter
    feat = extractor(x_batch)       # (B, 256)
    loss += extractor.regularization_loss()

Strategies
----------
  "all_features"      → AllFeaturesExtractor
  "pathway"           → PathwayAggregationExtractor
  "variance_filter"   → VarianceFilterExtractor
"""

from typing import Dict, Any, Optional
from .base import BaseRNAExtractor
from .all_features import AllFeaturesExtractor
from .pathway import PathwayAggregationExtractor
from .variance_filter import VarianceFilterExtractor


_REGISTRY: Dict[str, type] = {
    "all_features":    AllFeaturesExtractor,
    "pathway":         PathwayAggregationExtractor,
    "variance_filter": VarianceFilterExtractor,
}


def build_rna_extractor(
    strategy: str,
    input_dim: int,
    config: Optional[Dict[str, Any]] = None,
) -> BaseRNAExtractor:
    """
    Construct and return the requested RNA extractor.

    Args:
        strategy:  One of "all_features" | "pathway" | "variance_filter"
        input_dim: Number of genes in the input expression vector.
        config:    Strategy-specific hyperparameters (see each module's docstring).

    Returns:
        An unfit (or stateless) BaseRNAExtractor subclass instance.
    """
    strategy = strategy.lower().strip()
    if strategy not in _REGISTRY:
        raise ValueError(
            f"Unknown RNA extractor strategy '{strategy}'. "
            f"Choose from: {list(_REGISTRY.keys())}"
        )

    cls = _REGISTRY[strategy]
    return cls(input_dim=input_dim, config=config or {})


def list_strategies() -> list:
    return list(_REGISTRY.keys())
