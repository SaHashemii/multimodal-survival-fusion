"""Configuration and path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at top level of YAML config: {path}")
    return data


def resolve_path(root: str | Path, value: str | Path | None) -> Path | None:
    """Resolve a path relative to an explicit root.

    Absolute paths are returned unchanged. ``None`` values stay ``None``.
    """
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(root).expanduser() / path


def resolve_repo_path(repo_root: str | Path, value: str | Path | None) -> Path | None:
    """Resolve a path relative to the repository root."""
    return resolve_path(repo_root, value)


def _select_config_value(data_cfg: dict[str, Any], exp_data_cfg: dict[str, Any], key: str, selector_key: str) -> Any:
    value = data_cfg.get(key)
    if not isinstance(value, dict):
        return value

    selected = exp_data_cfg.get(selector_key, data_cfg.get(selector_key))
    if selected is None:
        raise ValueError(f"Data config {key} is a mapping; define data.{selector_key} in the data or experiment config.")
    if selected not in value:
        options = ", ".join(sorted(str(option) for option in value))
        raise ValueError(f"Unknown {selector_key}={selected!r}. Available values: {options}")
    return value[selected]


def materialize_data_config(data_cfg: dict[str, Any], exp_data_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve selectable data config entries into the concrete paths used by loaders."""
    exp_data_cfg = exp_data_cfg or {}
    resolved = dict(data_cfg)
    resolved["labels"] = _select_config_value(data_cfg, exp_data_cfg, "labels", "label_name")
    resolved["clinical_embeddings"] = _select_config_value(
        data_cfg,
        exp_data_cfg,
        "clinical_embeddings",
        "clinical_embedding_name",
    )
    resolved["omics_embeddings"] = _select_config_value(data_cfg, exp_data_cfg, "omics_embeddings", "omics_embedding_name")

    pathology = data_cfg.get("pathology")
    if isinstance(pathology, dict):
        selected = exp_data_cfg.get("pathology_feature_name", data_cfg.get("pathology_feature_name"))
        if selected is None:
            raise ValueError("Data config pathology is a mapping; define data.pathology_feature_name in the data or experiment config.")
        if selected not in pathology:
            options = ", ".join(sorted(str(option) for option in pathology))
            raise ValueError(f"Unknown pathology_feature_name={selected!r}. Available values: {options}")
        pathology_cfg = pathology[selected] or {}
        resolved["pathology_index"] = pathology_cfg.get("index_csv")
        resolved["pathology_features_root"] = pathology_cfg.get("features_root")
        resolved["pathology_representation"] = pathology_cfg.get("representation", "tile_bag")
        resolved["pathology_file_suffix"] = pathology_cfg.get("file_suffix", "_HE.h5")
        resolved["pathology_feature_key"] = pathology_cfg.get("feature_key", "features")

    return resolved
