"""Cross-validation helpers shared across experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def make_fold_assignments(
    sample_ids: list[str],
    labels: pd.DataFrame,
    n_splits: int,
    seed: int,
    event_col: str = "Event",
) -> pd.DataFrame:
    """Create stratified outer-fold assignments.

    The returned dataframe has columns ``sample_id`` and ``fold``.
    """
    if len(sample_ids) < n_splits:
        raise ValueError(f"Need at least {n_splits} samples, got {len(sample_ids)}.")
    y = labels.loc[sample_ids, event_col].astype(int).values
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = pd.DataFrame({"sample_id": sample_ids, "fold": -1})
    for fold, (_, test_idx) in enumerate(splitter.split(np.zeros(len(sample_ids)), y)):
        folds.loc[test_idx, "fold"] = fold
    return folds


def make_inner_validation_split(
    train_ids: list[str],
    labels: pd.DataFrame,
    seed: int,
    val_size: float,
    event_col: str = "Event",
) -> tuple[list[str], list[str]]:
    """Split outer-training IDs into fit and validation IDs."""
    y = labels.loc[train_ids, event_col].astype(int).values
    class_counts = np.bincount(y, minlength=2)
    if class_counts.min() < 2:
        raise ValueError(
            "Cannot create a stratified validation split: "
            f"event counts are {class_counts.tolist()}."
        )
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    fit_idx, val_idx = next(splitter.split(np.zeros(len(train_ids)), y))
    return [train_ids[i] for i in fit_idx], [train_ids[i] for i in val_idx]


def event_aware_batch_indices(
    events,
    batch_size: int,
    min_events: int = 3,
) -> list:
    """Build mini-batches with at least a target number of events when possible."""
    import torch

    n = len(events)
    event_idx = torch.where(events == 1)[0]
    censored_idx = torch.where(events == 0)[0]
    event_idx = event_idx[torch.randperm(len(event_idx), device=events.device)]
    censored_idx = censored_idx[torch.randperm(len(censored_idx), device=events.device)]

    n_event_per_batch = max(min_events, int(batch_size * (len(event_idx) / max(n, 1))))
    n_event_per_batch = min(n_event_per_batch, max(len(event_idx), 1))
    n_censored_per_batch = max(batch_size - n_event_per_batch, 0)

    batches = []
    event_ptr = censored_ptr = 0
    while event_ptr < len(event_idx):
        event_chunk = event_idx[event_ptr : event_ptr + n_event_per_batch]
        if len(event_chunk) == 0:
            break
        if len(censored_idx) > 0 and n_censored_per_batch > 0:
            censored_chunk = censored_idx[censored_ptr : censored_ptr + n_censored_per_batch]
            if len(censored_chunk) < n_censored_per_batch:
                extra = censored_idx[: n_censored_per_batch - len(censored_chunk)]
                censored_chunk = torch.cat([censored_chunk, extra])
            censored_ptr = (censored_ptr + n_censored_per_batch) % len(censored_idx)
        else:
            censored_chunk = torch.tensor([], dtype=torch.long, device=events.device)
        batch = torch.cat([event_chunk, censored_chunk])
        if len(batch) > 1:
            batch = batch[torch.randperm(len(batch), device=events.device)]
        batches.append(batch)
        event_ptr += n_event_per_batch
    return batches
