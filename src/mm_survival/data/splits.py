"""Fold split helpers for reproducible experiments."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mm_survival.training.cross_validation import make_inner_validation_split


@dataclass(frozen=True)
class FoldSplit:
    """Sample IDs for one outer CV fold and its inner validation split."""

    fold: int
    train_ids: list[str]
    fit_ids: list[str]
    val_ids: list[str]
    test_ids: list[str]


def validate_fold_assignments(fold_assignments: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a fold assignment dataframe."""
    if not {"sample_id", "fold"}.issubset(fold_assignments.columns):
        raise ValueError("fold_assignments must contain sample_id and fold columns.")
    out = fold_assignments[["sample_id", "fold"]].copy()
    out["sample_id"] = out["sample_id"].astype(str)
    out["fold"] = out["fold"].astype(int)
    if out["sample_id"].duplicated().any():
        duplicated = out.loc[out["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"Duplicate sample IDs in fold assignments: {duplicated[:10]}")
    return out


def prepare_fold_split(
    sample_ids: list[str],
    labels: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    fold: int,
    *,
    seed: int,
    val_size: float,
    event_col: str = "Event",
) -> FoldSplit:
    """Prepare shared train/fit/val/test IDs for one fold."""
    folds = validate_fold_assignments(fold_assignments)
    retained = set(map(str, sample_ids))
    folds = folds[folds["sample_id"].isin(retained)].copy()
    if folds.empty:
        raise ValueError("No retained samples are present in fold_assignments.")
    if fold not in set(folds["fold"]):
        raise ValueError(f"Fold {fold} is not present in fold_assignments.")

    train_ids = folds.loc[folds["fold"] != fold, "sample_id"].tolist()
    test_ids = folds.loc[folds["fold"] == fold, "sample_id"].tolist()
    if not train_ids:
        raise ValueError(f"Fold {fold} has no training samples.")
    if not test_ids:
        raise ValueError(f"Fold {fold} has no test samples.")

    fit_ids, val_ids = make_inner_validation_split(
        train_ids,
        labels,
        seed=seed,
        val_size=val_size,
        event_col=event_col,
    )
    return FoldSplit(
        fold=fold,
        train_ids=train_ids,
        fit_ids=fit_ids,
        val_ids=val_ids,
        test_ids=test_ids,
    )


def summarize_fold_split(labels: pd.DataFrame, split: FoldSplit, event_col: str = "Event") -> dict[str, int]:
    """Return sample and event counts for a fold split."""
    return {
        "fold": split.fold,
        "train_n": len(split.train_ids),
        "fit_n": len(split.fit_ids),
        "val_n": len(split.val_ids),
        "test_n": len(split.test_ids),
        "train_events": int(labels.loc[split.train_ids, event_col].sum()),
        "fit_events": int(labels.loc[split.fit_ids, event_col].sum()),
        "val_events": int(labels.loc[split.val_ids, event_col].sum()),
        "test_events": int(labels.loc[split.test_ids, event_col].sum()),
    }
