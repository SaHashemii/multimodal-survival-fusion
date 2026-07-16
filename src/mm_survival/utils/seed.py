"""
Reproducibility helpers
=======================

Seeds the random number generators used by Python, NumPy, and PyTorch.

Design rationale
----------------
* Cross-validation splits, model initialization, batching, and RNA dropout all
  depend on random number generation.
* Centralizing seed setup makes experiments easier to reproduce across scripts.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:

        # Deterministic cuDNN can improve reproducibility, but may reduce GPU
        # performance for some operations.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
