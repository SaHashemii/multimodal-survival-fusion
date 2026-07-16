"""
Prediction heads for survival models
====================================

Defines the shared Cox MLP head used by unimodal, bimodal, and multimodal
models.

Output convention
-----------------
  input patient representation → scalar log-risk score

The head does not output a probability. The scalar is optimized with the
negative Cox partial log-likelihood, where higher log-risk means higher relative
risk according to the learned model.
"""

from __future__ import annotations

from torch import nn


class CoxHead(nn.Module):
    """MLP head that outputs one log-risk score per sample."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.30,
        activation: str = "selu",
    ):
        super().__init__()
        hidden_dims = hidden_dims or []
        drop_cls = nn.AlphaDropout if activation == "selu" else nn.Dropout
        act_cls = nn.SELU if activation == "selu" else nn.ReLU

        # Start with LayerNorm so different encoder/fusion outputs enter the Cox
        # head on a comparable scale.
        layers: list[nn.Module] = [nn.LayerNorm(in_dim), drop_cls(dropout)]
        prev = in_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev, hidden_dim), nn.LayerNorm(hidden_dim), act_cls(), drop_cls(dropout)])
            prev = hidden_dim

        # Final linear layer produces the single Cox log-risk value.
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
