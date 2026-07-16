"""
Utilities for deterministic missing-RNA experiments
===================================================

Creates and applies binary RNA-observed masks for robustness experiments.

Mask convention
---------------
  rna_mask = 1  → RNA is observed and the real RNA tensor is used
  rna_mask = 0  → RNA is missing and the RNA tensor is replaced by zeros

Design rationale
----------------
* Training dropout is deterministic from the fold seed, so different fusion
  methods see the same missing-RNA patients under the same configuration.
* Validation/test missing-RNA evaluation uses an all-zero mask to simulate the
  setting where RNA is unavailable for every evaluated patient.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def make_rna_observed_mask(n_samples: int, dropout_prob: float, seed: int, device: torch.device) -> torch.Tensor:
    """Return 1 for observed RNA and 0 for dropped RNA."""
    if dropout_prob <= 0:
        return torch.ones(n_samples, dtype=torch.float32, device=device)
    if dropout_prob >= 1:
        return torch.zeros(n_samples, dtype=torch.float32, device=device)

    # NumPy's Generator gives a stable mask for a given seed without depending
    # on PyTorch's global RNG state or DataLoader ordering.
    rng = np.random.default_rng(seed)
    observed = rng.random(n_samples) >= dropout_prob
    return torch.tensor(observed.astype(np.float32), dtype=torch.float32, device=device)


def apply_rna_mask(rna: torch.Tensor, rna_mask: torch.Tensor | None) -> torch.Tensor:
    """Zero RNA rows where rna_mask is 0."""
    if rna_mask is None:
        return rna

    # Reshape [B] into [B, 1, ...] so the same mask works for flat RNA
    # embeddings and token-shaped RNA tensors.
    view_shape = (rna.shape[0],) + (1,) * (rna.ndim - 1)
    return rna * rna_mask.reshape(view_shape).to(device=rna.device, dtype=rna.dtype)


def missing_rna_mask_like(rna: torch.Tensor) -> torch.Tensor:
    """Return an all-zero observed mask for evaluating missing RNA."""
    return torch.zeros(rna.shape[0], dtype=torch.float32, device=rna.device)


def save_rna_mask(sample_ids: list[str], rna_mask: torch.Tensor, path) -> None:
    """Save fold-specific RNA mask assignments."""
    pd.DataFrame(
        {
            "sample_id": sample_ids,
            "rna_mask": rna_mask.detach().cpu().numpy().astype(int),
        }
    ).to_csv(path, index=False)
