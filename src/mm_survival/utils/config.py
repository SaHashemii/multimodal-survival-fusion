"""
Configuration and path helpers
==============================

Loads YAML configs and resolves selectable data entries into concrete paths.

Config pattern
--------------
  configs/data/local.yaml:
    defines available labels, clinical embeddings, pathology features, and
    omics embeddings for a local machine or server

  configs/experiments/*.yaml:
    selects which named data source to use through fields such as label_name,
    clinical_embedding_name, pathology_feature_name, and omics_embedding_name

Design rationale
----------------
* Local paths live in the data config, not in every experiment config.
* Experiment configs can switch datasets/embeddings by name while sharing the
  same training code.
* Relative paths are resolved against the configured data root or repo root.
"""

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

    # If a data field is a mapping, the selector can come from the experiment
    # config first, then fall back to the default in the data config.
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

    # Convert named choices such as labels.without_urolife or conch5 clinical
    # embeddings into the actual relative/absolute paths used by data loaders.
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

        # Pathology has extra metadata beyond a single path because UNI and
        # PRISM use different file layouts and representations.
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
