"""Pathology feature index and tensor loading utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_pathology_index(path: str | Path) -> pd.DataFrame:
    """Load pathology feature index with columns sample_id and feature_path."""
    df = pd.read_csv(path)
    if "source_path_sol" in df.columns and "feature_path" not in df.columns:
        df = df.rename(columns={"source_path_sol": "feature_path"})
    missing = {"sample_id", "feature_path"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required pathology index columns: {sorted(missing)}")
    df["sample_id"] = df["sample_id"].astype(str)
    df["feature_path"] = df["feature_path"].astype(str)
    return df.drop_duplicates("sample_id").set_index("sample_id")


def resolve_pathology_feature_path(
    feature_path: str | Path,
    pathology_features_root: str | Path | None = None,
    sample_id: str | None = None,
) -> Path:
    """Resolve a pathology feature path from the index.

    If the resolved path is a directory, this tries ``{sample_id}.pt`` first
    when a sample id is available, then falls back to a single ``*.pt`` file in
    that directory.
    """
    path = Path(feature_path).expanduser()
    if not path.is_absolute() and pathology_features_root is not None:
        path = Path(pathology_features_root).expanduser() / path

    if path.is_dir():
        if sample_id is not None:
            sample_file = path / f"{sample_id}.pt"
            if sample_file.is_file():
                return sample_file
        pt_files = sorted(path.glob("*.pt"))
        if len(pt_files) == 1:
            return pt_files[0]
        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in pathology feature directory: {path}")
        raise ValueError(f"Multiple .pt files found in {path}; cannot infer which one to use.")

    return path


def load_pathology_feature_file(path: str | Path) -> torch.Tensor | None:
    """Load one pathology feature tensor from a .pt file.

    Supported payloads are a tensor directly, a dict with ``feats``, or a dict
    with ``embeddings``.
    """
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    except OSError as exc:
        print(f"[Pathology] WARN: cannot load {path}: {exc}")
        return None

    feats = obj.get("feats") if isinstance(obj, dict) and "feats" in obj else obj
    feats = obj.get("embeddings") if isinstance(obj, dict) and "embeddings" in obj else feats
    if isinstance(feats, np.ndarray):
        feats = torch.from_numpy(feats)
    if not torch.is_tensor(feats):
        print(f"[Pathology] WARN: unsupported feature object in {path}")
        return None
    feats = feats.float()
    if feats.ndim != 2 or feats.shape[0] == 0:
        print(f"[Pathology] WARN: invalid feature shape in {path}: {tuple(feats.shape)}")
        return None
    return feats.contiguous()


def cap_tiles(
    feats: torch.Tensor,
    sample_id: str,
    seed: int,
    tile_cap: int | None,
) -> torch.Tensor:
    """Deterministically subsample tiles for a sample when over the cap."""
    if tile_cap is None or tile_cap <= 0 or feats.shape[0] <= tile_cap:
        return feats
    digest = hashlib.sha256(f"{sample_id}:{seed}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little") % (2**32))
    idx = rng.choice(feats.shape[0], size=tile_cap, replace=False)
    idx.sort()
    return feats[idx].contiguous()


def load_pathology_features(
    pathology_index: pd.DataFrame,
    sample_ids: list[str],
    pathology_features_root: str | Path | None = None,
    seed: int = 42,
    tile_cap: int | None = None,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Load pathology features for ordered sample ids.

    Returns a mapping of loaded tensors and a list of sample ids that were
    missing or invalid.
    """
    features: dict[str, torch.Tensor] = {}
    missing_or_invalid: list[str] = []
    for sample_id in sample_ids:
        try:
            feature_path = resolve_pathology_feature_path(
                pathology_index.loc[sample_id, "feature_path"],
                pathology_features_root=pathology_features_root,
                sample_id=sample_id,
            )
        except (FileNotFoundError, OSError, ValueError) as exc:
            print(f"[Pathology] WARN: cannot resolve feature path for {sample_id}: {exc}")
            missing_or_invalid.append(sample_id)
            continue
        feats = load_pathology_feature_file(feature_path)
        if feats is None:
            missing_or_invalid.append(sample_id)
            continue
        features[sample_id] = cap_tiles(feats, sample_id, seed, tile_cap)
    return features, missing_or_invalid
