"""Assembly utilities for common multimodal cohorts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from mm_survival.data.clinical import load_clinical_embeddings, load_clinical_table
from mm_survival.data.labels import load_labels
from mm_survival.data.pathology import load_pathology_features, load_pathology_index
from mm_survival.data.rna import load_rna_matrix
from mm_survival.utils.config import resolve_path


@dataclass
class MultimodalDataset:
    """Loaded data for patients with all required modalities."""

    sample_ids: list[str]
    labels: pd.DataFrame
    rna: pd.DataFrame
    clinical: pd.DataFrame | dict[str, torch.Tensor]
    pathology_features: dict[str, torch.Tensor]
    missing_summary: dict[str, Any]


def _required_path(data_root: Path, data_cfg: dict[str, Any], key: str) -> Path:
    if key not in data_cfg:
        raise ValueError(f"Data config must define data.{key}")
    path = resolve_path(data_root, data_cfg[key])
    if path is None:
        raise ValueError(f"Data config path data.{key} resolved to None")
    return path


def load_common_multimodal_dataset(
    data_cfg: dict[str, Any],
    *,
    clinical_source: str = "tabular",
    seed: int = 42,
    pathology_tile_cap: int | None = None,
) -> MultimodalDataset:
    """Load labels, RNA, clinical, and pathology for common samples only."""
    if "root" not in data_cfg:
        raise ValueError("Data config must define data.root")

    data_root = Path(data_cfg["root"]).expanduser()
    labels_path = _required_path(data_root, data_cfg, "labels")
    rna_path = _required_path(data_root, data_cfg, "rna")
    pathology_index_path = _required_path(data_root, data_cfg, "pathology_index")
    pathology_features_root = resolve_path(data_root, data_cfg.get("pathology_features_root"))
    if pathology_features_root is None:
        pathology_features_root = pathology_index_path.parent

    labels = load_labels(labels_path)
    rna = load_rna_matrix(rna_path)
    pathology = load_pathology_index(pathology_index_path)

    if clinical_source == "embedding":
        clinical_embeddings_path = _required_path(data_root, data_cfg, "clinical_embeddings")
        clinical_data = load_clinical_embeddings(clinical_embeddings_path)
        clinical_ids = set(clinical_data)
        clinical_label = "clinical_embeddings"
    elif clinical_source == "tabular":
        clinical_table_path = _required_path(data_root, data_cfg, "clinical_tabular")
        clinical_data = load_clinical_table(clinical_table_path)
        clinical_ids = set(clinical_data["sample_id"].astype(str))
        clinical_label = "clinical_tabular"
    else:
        raise ValueError(f"Unknown clinical_source: {clinical_source}")

    modality_ids = {
        "labels": set(labels.index.astype(str)),
        "rna": set(rna.index.astype(str)),
        clinical_label: clinical_ids,
        "pathology_index": set(pathology.index.astype(str)),
    }
    common = sorted(set.intersection(*modality_ids.values()))
    missing_summary: dict[str, Any] = {
        "modality_counts": {name: len(ids) for name, ids in modality_ids.items()},
        "common_before_pathology_load": len(common),
    }

    pathology_features, invalid_pathology = load_pathology_features(
        pathology,
        common,
        pathology_features_root=pathology_features_root,
        seed=seed,
        tile_cap=pathology_tile_cap,
    )
    sample_ids = sorted(set(common) & set(pathology_features))
    missing_summary["invalid_pathology_features"] = invalid_pathology
    missing_summary["retained_samples"] = len(sample_ids)

    if clinical_source == "embedding":
        clinical_out = {sample_id: clinical_data[sample_id] for sample_id in sample_ids}
    else:
        clinical_out = clinical_data[clinical_data["sample_id"].isin(sample_ids)].copy()

    return MultimodalDataset(
        sample_ids=sample_ids,
        labels=labels.loc[sample_ids],
        rna=rna.loc[sample_ids],
        clinical=clinical_out,
        pathology_features={sample_id: pathology_features[sample_id] for sample_id in sample_ids},
        missing_summary=missing_summary,
    )
