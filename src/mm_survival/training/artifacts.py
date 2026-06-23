"""Helpers for saving training artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_history(history: list[dict[str, Any]] | pd.DataFrame, path: str | Path) -> None:
    """Write training history to CSV."""
    df = history if isinstance(history, pd.DataFrame) else pd.DataFrame(history)
    df.to_csv(path, index=False)


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    """Write a JSON artifact with readable indentation."""
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_risk_scores(
    sample_ids: list[str],
    risk,
    time,
    event,
    path: str | Path,
) -> None:
    """Write per-sample risk scores to CSV."""
    pd.DataFrame(
        {
            "sample_id": sample_ids,
            "log_risk": risk,
            "Event": event,
            "Time": time,
        }
    ).to_csv(path, index=False)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    summary: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a model checkpoint with summary metadata."""
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "summary": summary,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
