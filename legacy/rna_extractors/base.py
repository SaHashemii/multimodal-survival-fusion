"""
Base class for RNA feature extractors.
All extractors expose a unified interface for the multimodal pipeline.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class BaseRNAExtractor(ABC, nn.Module):
    """
    Abstract base for RNA-seq feature extractors.

    Every subclass must implement:
      - forward(x) → (B, out_dim) feature tensor
      - output_dim property
      - fit(x_train) for any statistics that require training-set data
        (e.g. variance estimates, pathway loading matrices)

    The extractor is responsible for its own L1 / regularization losses;
    call extractor.regularization_loss() inside your training loop and add
    it to the task loss.
    """

    def __init__(self, input_dim: int, config: Dict[str, Any]):
        super().__init__()
        self.input_dim = input_dim
        self.config = config
        self._is_fitted = False

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the output feature vector."""
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) raw RNA expression tensor (log1p-TPM recommended)
        Returns:
            feat: (B, output_dim)
        """
        ...

    def fit(self, x_train: torch.Tensor) -> "BaseRNAExtractor":
        """
        Compute any dataset-level statistics from training data.
        Must be called before forward() for extractors that need it.
        Default: no-op (stateless extractors skip this).
        """
        self._is_fitted = True
        return self

    def regularization_loss(self) -> torch.Tensor:
        """
        Optional per-extractor regularization term to add to task loss.
        Default: zero (no extra regularization).
        """
        return torch.tensor(0.0, device=next(self.parameters()).device
                            if len(list(self.parameters())) > 0 else torch.device("cpu"))

    def get_config(self) -> Dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            **self.config,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"input_dim={self.input_dim}, "
            f"output_dim={self.output_dim})"
        )
