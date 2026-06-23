"""
Option 1 – All Features + L1-regularised MLP
=============================================
Takes the full RNA expression vector as input and passes it through a
trainable MLP with explicit L1 weight regularisation.

Design rationale
----------------
* No feature pre-selection: the model learns which genes are relevant.
* L1 penalty on the first Linear layer promotes sparsity in gene usage,
  acting as a soft gene-selection gate.
* Optional input BatchNorm stabilises training when expression scales differ.

Config keys
-----------
  hidden_dims   : list[int]  – MLP hidden layer widths  (default [512, 256])
  out_dim       : int        – output embedding dim      (default 256)
  dropout       : float      – dropout rate              (default 0.25)
  l1_lambda     : float      – weight on L1 penalty      (default 1e-4)
  input_norm    : bool       – BatchNorm on raw input    (default True)
  activation    : str        – "relu"|"gelu"|"silu"      (default "relu")
  norm          : str        – "layer"|"batch"|"none"    (default "layer")
"""

from typing import Dict, Any, List
import torch
import torch.nn as nn

from .base import BaseRNAExtractor
from .mlp_utils import build_mlp, l1_weight_penalty


class AllFeaturesExtractor(BaseRNAExtractor):
    """All RNA features → L1-regularised MLP → fixed-dim embedding."""

    def __init__(self, input_dim: int, config: Dict[str, Any] = {}):
        super().__init__(input_dim, config)

        hidden_dims: List[int] = config.get("hidden_dims", [512, 256])
        out_dim: int           = config.get("out_dim", 256)
        dropout: float         = config.get("dropout", 0.25)
        self.l1_lambda: float  = config.get("l1_lambda", 1e-4)
        activation: str        = config.get("activation", "relu")
        norm: str              = config.get("norm", "layer")
        use_input_norm: bool   = config.get("input_norm", True)

        self._out_dim = out_dim

        # Optional input normalisation
        self.input_norm = nn.BatchNorm1d(input_dim) if use_input_norm else nn.Identity()

        # First linear layer is kept separate so L1 targets it specifically
        # (acts as a soft-feature-selection gate)
        self.gate = nn.Linear(input_dim, hidden_dims[0] if hidden_dims else out_dim)

        # Remaining MLP body (starts from hidden_dims[0] or directly out_dim)
        body_in = hidden_dims[0] if hidden_dims else out_dim
        self.body = build_mlp(
            in_dim=body_in,
            hidden_dims=hidden_dims[1:],
            out_dim=out_dim,
            dropout=dropout,
            activation=activation,
            norm=norm,
        )

        self._is_fitted = True  # stateless, no fit() needed

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
        x = self.input_norm(x)
        x = self.gate(x)
        feat = self.body(x)
        return feat

    def regularization_loss(self) -> torch.Tensor:
        """
        Explicit L1 penalty on the gate layer weights.
        Call this inside your training loop and add to task loss:
            loss = task_loss + extractor.regularization_loss()
        """
        if self.l1_lambda == 0.0:
            return torch.zeros(1, device=self.gate.weight.device)
        return self.l1_lambda * self.gate.weight.abs().sum()
