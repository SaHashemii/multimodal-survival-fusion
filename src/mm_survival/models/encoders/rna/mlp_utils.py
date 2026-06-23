"""
Shared MLP utility used by all RNA extractor strategies.
"""

from typing import List, Optional
import torch
import torch.nn as nn


def build_mlp(
    in_dim: int,
    hidden_dims: List[int],
    out_dim: int,
    dropout: float = 0.25,
    activation: str = "relu",
    norm: str = "layer",    # "layer" | "batch" | "none"
    final_activation: bool = False,
) -> nn.Sequential:
    """
    Constructs a fully-connected MLP block.

    Args:
        in_dim:           Input feature size.
        hidden_dims:      List of hidden layer sizes (can be empty for linear proj).
        out_dim:          Output feature size.
        dropout:          Dropout probability (applied after each hidden layer).
        activation:       "relu" | "gelu" | "silu" | "selu"
        norm:             Normalization after each hidden layer.
        final_activation: Whether to apply activation after the last linear layer.

    Returns:
        nn.Sequential MLP.
    """
    act_map = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "selu": nn.SELU}
    Act = act_map[activation]

    layers: List[nn.Module] = []
    prev = in_dim

    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))

        if norm == "layer":
            layers.append(nn.LayerNorm(h))
        elif norm == "batch":
            layers.append(nn.BatchNorm1d(h))

        layers.append(Act())

        if dropout > 0:
            Drop = nn.AlphaDropout if activation == "selu" else nn.Dropout
            layers.append(Drop(dropout))

        prev = h

    layers.append(nn.Linear(prev, out_dim))

    if final_activation:
        layers.append(Act())

    return nn.Sequential(*layers)


def l1_weight_penalty(module: nn.Module) -> torch.Tensor:
    """
    Computes the sum of absolute values of all Linear layer weights in `module`.
    Use as an explicit L1 regularization term on top of the task loss.
    """
    penalty = torch.tensor(0.0)
    for m in module.modules():
        if isinstance(m, nn.Linear):
            penalty = penalty + m.weight.abs().sum()
    return penalty
