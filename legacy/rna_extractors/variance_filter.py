"""
Option 3 – Variance Filter Extractor (Coefficient of Variation)
================================================================
Selects the top-K most variable genes measured by coefficient of variation
(CV = σ / μ) across the training cohort, then feeds the reduced expression
vector through an MLP.

Pipeline
--------
  fit(x_train):
    compute per-gene CV on training data → rank → keep top-K indices

  forward(x):
    x[:, top_k_idx]            (B, K)  ← hard gene selection
         ↓  MLP
    embedding  (B, out_dim)

Design rationale
----------------
* CV normalises variance by mean expression, preventing highly expressed
  housekeeping genes from dominating the selection.
* Hard selection keeps K << G, drastically reducing parameter count.
* The threshold can be set by top_k (count) or cv_percentile (percentile).
* Optional: also supports plain variance ranking when use_cv=False,
  useful when data is already z-scored.

Config keys
-----------
  top_k           : int       – number of genes to keep          (default 2000)
  use_cv          : bool      – use CV instead of raw variance   (default True)
  cv_percentile   : float     – alternative: keep genes above
                                this CV percentile (0–100).
                                Overrides top_k if set.          (default None)
  out_dim         : int       – output embedding dim             (default 256)
  hidden_dims     : list[int] – MLP hidden dims                  (default [512, 256])
  dropout         : float     – dropout                          (default 0.25)
  activation      : str       – "relu"|"gelu"|"silu"             (default "relu")
  norm            : str       – "layer"|"batch"|"none"           (default "layer")
  input_norm      : bool      – BatchNorm on selected genes      (default True)
  gene_names      : list[str] – optional; enables reporting      (default None)
  eps             : float     – stability constant for CV denom  (default 1e-8)
"""

from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import numpy as np

from .base import BaseRNAExtractor
from .mlp_utils import build_mlp


class VarianceFilterExtractor(BaseRNAExtractor):
    """
    Top-K CV gene filter → MLP → embedding.
    Requires fit(x_train) before forward().
    """

    def __init__(self, input_dim: int, config: Dict[str, Any] = {}):
        super().__init__(input_dim, config)

        self.top_k: int               = config.get("top_k", 2000)
        self.use_cv: bool             = config.get("use_cv", True)
        self.cv_percentile: Optional[float] = config.get("cv_percentile", None)
        self.eps: float               = config.get("eps", 1e-8)
        self.gene_names: Optional[List[str]] = config.get("gene_names", None)

        out_dim: int           = config.get("out_dim", 256)
        hidden_dims: List[int] = config.get("hidden_dims", [512, 256])
        dropout: float         = config.get("dropout", 0.25)
        activation: str        = config.get("activation", "relu")
        norm: str              = config.get("norm", "layer")
        use_input_norm: bool   = config.get("input_norm", True)

        self._out_dim = out_dim

        # Buffers populated by fit()
        self.register_buffer("selected_idx", torch.zeros(self.top_k, dtype=torch.long))
        self.register_buffer("gene_scores", torch.zeros(input_dim))
        self._actual_k: int = self.top_k   # may differ if cv_percentile used

        # Input norm on selected genes (dim set after fit; placeholder here)
        self._use_input_norm = use_input_norm
        self._norm_cfg = norm

        # MLP — built lazily in fit() once we know the actual K
        self._dropout   = dropout
        self._activation = activation
        self._mlp: Optional[nn.Module] = None

        self._is_fitted = False

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, x_train: torch.Tensor) -> "VarianceFilterExtractor":
        """
        Compute per-gene CV (or variance) on training data and select top-K.

        Args:
            x_train: (N, G) training-set expression matrix (log1p-TPM)
        """
        x = x_train.float()
        mean = x.mean(dim=0)          # (G,)
        std  = x.std(dim=0)           # (G,)

        if self.use_cv:
            scores = std / (mean.abs() + self.eps)
        else:
            scores = std ** 2          # plain variance

        self.gene_scores = scores

        # Determine threshold
        if self.cv_percentile is not None:
            threshold = torch.quantile(scores, self.cv_percentile / 100.0)
            idx = torch.where(scores >= threshold)[0]
            # Sort by score descending
            idx = idx[scores[idx].argsort(descending=True)]
        else:
            k = min(self.top_k, scores.numel())
            idx = scores.topk(k).indices

        self._actual_k = len(idx)

        # Pad or resize buffer if needed
        if self._actual_k != self.selected_idx.numel():
            self.register_buffer(
                "selected_idx",
                torch.zeros(self._actual_k, dtype=torch.long)
            )

        self.selected_idx.copy_(idx)

        # Build MLP now that we know K
        self._mlp = self._build_mlp(self._actual_k)
        self._is_fitted = True

        print(
            f"[VarianceFilterExtractor] fit complete: "
            f"selected {self._actual_k} / {self.input_dim} genes "
            f"({'CV' if self.use_cv else 'variance'} ranking)"
        )
        return self

    def _build_mlp(self, k: int) -> nn.Sequential:
        from .mlp_utils import build_mlp

        hidden_dims = self.config.get("hidden_dims", [512, 256])
        norm = self._norm_cfg

        layers: List[nn.Module] = []

        if self._use_input_norm:
            layers.append(nn.BatchNorm1d(k))

        layers.append(
            build_mlp(
                in_dim=k,
                hidden_dims=hidden_dims,
                out_dim=self._out_dim,
                dropout=self._dropout,
                activation=self._activation,
                norm=norm,
            )
        )
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    @property
    def output_dim(self) -> int:
        return self._out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) log1p-TPM expression matrix
        Returns:
            feat: (B, out_dim)
        """
        if not self._is_fitted or self._mlp is None:
            raise RuntimeError(
                "VarianceFilterExtractor must be fitted before forward(). "
                "Call extractor.fit(x_train) first."
            )
        x_sel = x[:, self.selected_idx]   # (B, K) — hard gene selection
        feat  = self._mlp(x_sel)           # (B, out_dim)
        return feat

    # ------------------------------------------------------------------
    # Inspection utilities
    # ------------------------------------------------------------------

    def get_selected_genes(self) -> List[str]:
        """Return names of selected genes (requires gene_names in config)."""
        if self.gene_names is None:
            return [f"gene_{i.item()}" for i in self.selected_idx]
        return [self.gene_names[i.item()] for i in self.selected_idx]

    def get_score_summary(self, top_n: int = 20) -> Dict[str, Any]:
        """
        Return a dict summarising the top-scoring genes.
        Useful for sanity-checking biological relevance.
        """
        scores = self.gene_scores
        top_idx = scores.topk(min(top_n, len(scores))).indices
        names = (
            [self.gene_names[i] for i in top_idx]
            if self.gene_names else [f"gene_{i}" for i in top_idx]
        )
        return {
            "metric": "CV" if self.use_cv else "variance",
            "top_genes": names,
            "top_scores": scores[top_idx].tolist(),
            "num_selected": self._actual_k,
            "score_threshold": scores[self.selected_idx].min().item(),
        }

    # Type hint helpers
    from typing import Dict, Any
