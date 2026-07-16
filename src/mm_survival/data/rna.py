"""
RNA matrix loading and preprocessing helpers
===========================================

Loads bulk RNA-seq matrices and applies simple train-only median imputation.

Expected CSV format
-------------------
  rows: genes
  columns: samples

The loader transposes the matrix to the modeling format:

  rows: samples
  columns: genes

Design rationale
----------------
* RNA imputation medians are fit on the training/fit split only.
* Validation and test RNA values are transformed with the fitted medians, which
  avoids leaking validation/test distributions into preprocessing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_rna_matrix(path: str | Path) -> pd.DataFrame:
    """Load a gene-by-sample RNA CSV as a sample-by-gene dataframe.

    Public input format:
        rows = genes, columns = samples.
    """
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)

    # Public RNA files are gene-by-sample; models use sample-by-gene tensors.
    rna = df.T
    rna.index.name = "sample_id"
    rna.columns = rna.columns.astype(str)
    return rna.apply(pd.to_numeric, errors="coerce")


def fit_rna_medians(train: pd.DataFrame) -> pd.Series:
    """Fit per-gene medians on the training/fit split only."""
    return train.median(axis=0, skipna=True).fillna(0.0)


def transform_rna_with_medians(data: pd.DataFrame, medians: pd.Series) -> np.ndarray:
    """Impute RNA data with previously fitted medians and return float32 array."""
    return data.fillna(medians).values.astype(np.float32)


def impute_rna_from_train(
    train: pd.DataFrame,
    test: pd.DataFrame,
    val: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, float]]:
    """Fit medians on train and apply them to train/test/optional validation."""

    # This helper is kept for code paths that work with explicit train/test/val
    # dataframes instead of the FoldSplit-based tensor preparation utilities.
    medians = fit_rna_medians(train)
    x_train = transform_rna_with_medians(train, medians)
    x_test = transform_rna_with_medians(test, medians)
    x_val = transform_rna_with_medians(val, medians) if val is not None else None
    return x_train, x_test, x_val, {str(k): float(v) for k, v in medians.items()}
