"""
Fold tensor preparation for embedding-based multimodal models
=============================================================

Converts one FoldSplit into tensors used by concat, gated, low-rank, and
two-modality embedding-based Cox models.

Pipeline
--------
  split sample IDs → RNA median imputation → clinical tensor preparation
  attach pathology feature bags → pack labels/times/events into tensors

Design rationale
----------------
* RNA medians are fit on the fit split only, then reused for train/val/test.
* Clinical preprocessing is also fit on the fit split only for tabular data.
* Clinical embeddings are padded/truncated to one fixed [tokens, dim] shape.
* Pathology features stay as a list of per-patient tensors because each slide
  can contain a different number of tile embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from mm_survival.data.clinical import (
    fit_clinical_preprocessor,
    stack_clinical_embeddings,
    transform_clinical_table,
)
from mm_survival.data.multimodal import MultimodalDataset
from mm_survival.data.rna import fit_rna_medians, transform_rna_with_medians
from mm_survival.data.splits import FoldSplit


EmbeddingTensors = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass
class EmbeddingFoldData:
    """Prepared tensors and metadata for one embedding-based multimodal fold."""

    train_tensors: EmbeddingTensors
    fit_tensors: EmbeddingTensors
    val_tensors: EmbeddingTensors
    test_tensors: EmbeddingTensors
    pathology_train: list[torch.Tensor]
    pathology_fit: list[torch.Tensor]
    pathology_val: list[torch.Tensor]
    pathology_test: list[torch.Tensor]
    rna_gene_names: list[str]
    rna_medians: dict[str, float]
    clinical_dim: int
    clinical_token_count: int | None
    clinical_metadata: dict[str, Any]


def _to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(array, dtype=torch.float32, device=device)


def _label_tensors(dataset: MultimodalDataset, sample_ids: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    time = _to_tensor(dataset.labels.loc[sample_ids, "Time"].values.astype(np.float32), device)
    event = _to_tensor(dataset.labels.loc[sample_ids, "Event"].values.astype(np.float32), device)
    return time, event


def _pack(
    x_rna: np.ndarray,
    x_clinical: np.ndarray,
    dataset: MultimodalDataset,
    sample_ids: list[str],
    device: torch.device,
) -> EmbeddingTensors:
    time, event = _label_tensors(dataset, sample_ids, device)
    return _to_tensor(x_rna, device), _to_tensor(x_clinical, device), time, event


def prepare_embedding_fold_data(
    dataset: MultimodalDataset,
    split: FoldSplit,
    *,
    clinical_source: str,
    device: torch.device,
) -> EmbeddingFoldData:
    """Prepare tensors for concat/gated/low-rank multimodal models."""
    gene_names = dataset.rna.columns.astype(str).tolist()
    fit_rna = dataset.rna.loc[split.fit_ids, gene_names]

    # Fit RNA imputation on the fit split only so validation/test missingness
    # does not influence preprocessing statistics.
    medians = fit_rna_medians(fit_rna)

    x_rna_fit = transform_rna_with_medians(fit_rna, medians)
    x_rna_train = transform_rna_with_medians(dataset.rna.loc[split.train_ids, gene_names], medians)
    x_rna_val = transform_rna_with_medians(dataset.rna.loc[split.val_ids, gene_names], medians)
    x_rna_test = transform_rna_with_medians(dataset.rna.loc[split.test_ids, gene_names], medians)
    rna_medians = {str(gene): float(value) for gene, value in medians.items()}

    clinical_token_count = None
    if clinical_source == "embedding":
        clinical_embeddings = dataset.clinical
        if not isinstance(clinical_embeddings, dict):
            raise TypeError("Expected clinical embeddings dictionary when clinical_source='embedding'.")

        # Embedding-based clinical inputs are token tensors. The maximum token
        # count and embedding dimension define a fixed tensor shape per fold.
        clinical_token_count = max(int(emb.shape[0]) for emb in clinical_embeddings.values())
        clinical_dim = int(next(iter(clinical_embeddings.values())).shape[1])
        x_clinical_train = stack_clinical_embeddings(
            clinical_embeddings,
            split.train_ids,
            token_count=clinical_token_count,
            embedding_dim=clinical_dim,
        )
        x_clinical_fit = stack_clinical_embeddings(
            clinical_embeddings,
            split.fit_ids,
            token_count=clinical_token_count,
            embedding_dim=clinical_dim,
        )
        x_clinical_val = stack_clinical_embeddings(
            clinical_embeddings,
            split.val_ids,
            token_count=clinical_token_count,
            embedding_dim=clinical_dim,
        )
        x_clinical_test = stack_clinical_embeddings(
            clinical_embeddings,
            split.test_ids,
            token_count=clinical_token_count,
            embedding_dim=clinical_dim,
        )
        clinical_metadata = {
            "clinical_source": "embedding",
            "clinical_token_count_max": clinical_token_count,
            "clinical_embedding_dim": clinical_dim,
            "clinical_token_counts": {
                sample_id: int(clinical_embeddings[sample_id].shape[0])
                for sample_id in split.train_ids + split.test_ids
            },
        }
    elif clinical_source == "tabular":
        clinical_table = dataset.clinical
        if isinstance(clinical_table, dict):
            raise TypeError("Expected clinical dataframe when clinical_source='tabular'.")
        clinical_index = clinical_table.set_index("sample_id")

        # Tabular preprocessing is learned on fit IDs and then applied to every
        # split with frozen numeric/categorical metadata.
        clinical_fit = clinical_index.loc[split.fit_ids].reset_index()
        clinical_train = clinical_index.loc[split.train_ids].reset_index()
        clinical_val = clinical_index.loc[split.val_ids].reset_index()
        clinical_test = clinical_index.loc[split.test_ids].reset_index()
        clinical_metadata = fit_clinical_preprocessor(clinical_fit)
        x_clinical_fit = transform_clinical_table(clinical_fit, clinical_metadata)
        x_clinical_train = transform_clinical_table(clinical_train, clinical_metadata)
        x_clinical_val = transform_clinical_table(clinical_val, clinical_metadata)
        x_clinical_test = transform_clinical_table(clinical_test, clinical_metadata)
        clinical_dim = x_clinical_train.shape[1]
        clinical_metadata = {
            **clinical_metadata,
            "clinical_source": "tabular",
            "clinical_feature_count": clinical_dim,
        }
    else:
        raise ValueError(f"Unknown clinical_source: {clinical_source}")

    return EmbeddingFoldData(
        train_tensors=_pack(x_rna_train, x_clinical_train, dataset, split.train_ids, device),
        fit_tensors=_pack(x_rna_fit, x_clinical_fit, dataset, split.fit_ids, device),
        val_tensors=_pack(x_rna_val, x_clinical_val, dataset, split.val_ids, device),
        test_tensors=_pack(x_rna_test, x_clinical_test, dataset, split.test_ids, device),
        pathology_train=[dataset.pathology_features[sample_id] for sample_id in split.train_ids],
        pathology_fit=[dataset.pathology_features[sample_id] for sample_id in split.fit_ids],
        pathology_val=[dataset.pathology_features[sample_id] for sample_id in split.val_ids],
        pathology_test=[dataset.pathology_features[sample_id] for sample_id in split.test_ids],
        rna_gene_names=gene_names,
        rna_medians=rna_medians,
        clinical_dim=clinical_dim,
        clinical_token_count=clinical_token_count,
        clinical_metadata=clinical_metadata,
    )
