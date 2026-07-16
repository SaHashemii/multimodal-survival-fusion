"""
Clinical tabular and embedding loaders
======================================

Supports two clinical representations used by the survival models:

  1. Tabular clinical covariates loaded from a CSV file.
  2. Precomputed clinical text embeddings loaded from one .pt file per sample.

Tabular preprocessing
---------------------
  fit_clinical_preprocessor(fit):
    normalize missing sentinels → infer numeric/categorical columns
    impute/standardize numeric features → one-hot encode categorical features
    remove zero-variance columns

  transform_clinical_table(df, meta):
    apply the fitted metadata to validation/test data without refitting

Design rationale
----------------
* Missing values are normalized from legacy -1 / "-1" sentinels to NaN.
* Preprocessing metadata is fit on the training split only to avoid leakage.
* Categorical levels and kept feature columns are frozen after fitting, so every
  split has the same clinical feature dimension.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def load_clinical_table(path: str | Path) -> pd.DataFrame:
    """Load tabular clinical covariates with a normalized sample_id column."""
    df = pd.read_csv(path)
    if "case_id" in df.columns and "sample_id" not in df.columns:
        df = df.rename(columns={"case_id": "sample_id"})
    if "sample_id" not in df.columns:
        raise ValueError(f"{path} must contain sample_id or case_id.")
    df["sample_id"] = df["sample_id"].astype(str)
    return df


def infer_numeric(col: pd.Series) -> bool:
    """Return True when a column is mostly parseable as numeric."""
    values = pd.to_numeric(col.replace({-1: np.nan, "-1": np.nan}), errors="coerce")
    return np.isfinite(values).mean() >= 0.8


def _clean_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return clinical feature columns with legacy missing-value sentinels normalized."""
    features = df.drop(columns=["sample_id"], errors="ignore").copy()
    for col in features.columns:
        features[col] = features[col].replace({-1: np.nan, "-1": np.nan})
    return features


def fit_clinical_preprocessor(fit: pd.DataFrame) -> dict[str, Any]:
    """Fit tabular clinical preprocessing metadata on the fit split only."""
    feature_df = _clean_clinical_features(fit)

    # Column type inference is done on the fit split and then frozen for all
    # validation/test transforms.
    numeric_cols = [col for col in feature_df.columns if infer_numeric(feature_df[col])]
    categorical_cols = [col for col in feature_df.columns if col not in numeric_cols]

    numeric_medians: dict[str, float] = {}
    numeric_means: dict[str, float] = {}
    numeric_stds: dict[str, float] = {}
    numeric_outputs = []
    for col in numeric_cols:
        values = pd.to_numeric(feature_df[col], errors="coerce")

        # Median imputation and standardization parameters are estimated from
        # the fit split only, preventing validation/test distribution leakage.
        median = float(values.median()) if values.notna().any() else 0.0
        filled = values.fillna(median)
        mean = float(filled.mean())
        std = float(filled.std())
        numeric_medians[col] = median
        numeric_means[col] = mean
        numeric_stds[col] = std if np.isfinite(std) and std > 1e-8 else 1.0
        numeric_outputs.append(((filled - numeric_means[col]) / numeric_stds[col]).rename(col))
    fit_num = pd.concat(numeric_outputs, axis=1) if numeric_outputs else pd.DataFrame(index=feature_df.index)

    categorical_levels: dict[str, list[str]] = {}
    categorical_outputs = []
    for col in categorical_cols:
        values = feature_df[col].fillna("Missing").astype("string")

        # Store fit-split levels so unseen validation/test levels map to all
        # zeros rather than changing the feature dimension.
        levels = sorted(values.unique().tolist())
        categorical_levels[col] = levels
        for level in levels:
            categorical_outputs.append((values == level).astype(float).rename(f"{col}={level}"))
    fit_cat = pd.concat(categorical_outputs, axis=1) if categorical_outputs else pd.DataFrame(index=feature_df.index)

    fit_processed = pd.concat([fit_num, fit_cat], axis=1)

    # Drop constant features after encoding; they carry no information and can
    # make very small clinical designs less stable.
    kept_features = fit_processed.columns[np.var(fit_processed.values.astype(float), axis=0) > 0.0].tolist()

    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "numeric_medians": numeric_medians,
        "numeric_means": numeric_means,
        "numeric_stds": numeric_stds,
        "categorical_levels": categorical_levels,
        "kept_features": kept_features,
    }


def transform_clinical_table(df: pd.DataFrame, meta: dict[str, Any]) -> np.ndarray:
    """Transform tabular clinical covariates with fitted metadata."""
    feature_df = _clean_clinical_features(df)
    numeric_outputs = []

    for col in meta["numeric_cols"]:
        values = pd.to_numeric(feature_df.get(col, pd.Series(index=df.index)), errors="coerce")
        filled = values.fillna(meta["numeric_medians"][col])
        scaled = (filled - meta["numeric_means"][col]) / meta["numeric_stds"][col]
        numeric_outputs.append(scaled.astype(float).rename(col))
    out_num = pd.concat(numeric_outputs, axis=1) if numeric_outputs else pd.DataFrame(index=feature_df.index)

    categorical_outputs = []
    for col in meta["categorical_cols"]:
        values = feature_df.get(col, pd.Series(index=df.index)).fillna("Missing").astype("string")
        for level in meta["categorical_levels"][col]:
            categorical_outputs.append((values == level).astype(float).rename(f"{col}={level}"))
    out_cat = pd.concat(categorical_outputs, axis=1) if categorical_outputs else pd.DataFrame(index=feature_df.index)

    processed = pd.concat([out_num, out_cat], axis=1).reindex(columns=meta["kept_features"], fill_value=0)
    if processed.shape[1] == 0:
        return np.zeros((len(df), 0), dtype=np.float32)
    return processed.values.astype(np.float32)


def load_clinical_embedding_file(path: str | Path) -> torch.Tensor:
    """Load one clinical embedding/token tensor from a .pt file."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    emb = obj["embeddings"] if isinstance(obj, dict) and "embeddings" in obj else obj
    if isinstance(emb, np.ndarray):
        emb = torch.from_numpy(emb)
    if not torch.is_tensor(emb):
        raise TypeError(f"Unsupported clinical embedding object in {path}")
    emb = emb.float().contiguous()
    if emb.ndim == 1:
        emb = emb.unsqueeze(0)
    if emb.ndim != 2:
        raise ValueError(f"Expected [tokens, dim] or [dim] embedding in {path}, got {tuple(emb.shape)}")
    return emb


def load_clinical_embeddings(embedding_dir: str | Path) -> dict[str, torch.Tensor]:
    """Load one .pt clinical embedding file per sample_id."""
    directory = Path(embedding_dir)
    files = sorted(directory.glob("*.pt"))
    if not files:
        raise ValueError(f"No .pt files found in clinical embedding dir: {directory}")
    return {path.stem: load_clinical_embedding_file(path) for path in files}


def stack_clinical_embeddings(
    embeddings: dict[str, torch.Tensor],
    sample_ids: list[str],
    token_count: int | None = None,
    embedding_dim: int | None = None,
) -> np.ndarray:
    """Pad and stack clinical embeddings in sample order."""
    if token_count is None:
        token_count = max(int(embeddings[sid].shape[0]) for sid in sample_ids)
    if embedding_dim is None:
        embedding_dim = int(next(iter(embeddings.values())).shape[1])

    out = np.zeros((len(sample_ids), token_count, embedding_dim), dtype=np.float32)
    for row, sample_id in enumerate(sample_ids):
        emb = embeddings[sample_id].detach().cpu().numpy().astype(np.float32)

        # Keep a fixed [tokens, dim] shape across samples by truncating longer
        # embeddings and zero-padding shorter embeddings.
        n_tokens = min(emb.shape[0], token_count)
        n_dim = min(emb.shape[1], embedding_dim)
        out[row, :n_tokens, :n_dim] = emb[:n_tokens, :n_dim]
    return out
