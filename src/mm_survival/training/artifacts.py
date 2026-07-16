"""
Helpers for saving training artifacts
=====================================

Centralizes the output formats written by training scripts.

Common artifacts
----------------
  history.csv:
    per-epoch train/validation losses and C-index values

  summary.json:
    fold-level or experiment-level metrics and metadata

  test_risk_scores.csv:
    sample_id, log_risk, Event, and Time for downstream C-index/KM analysis

  checkpoint.pt:
    model state dict plus summary metadata

Design rationale
----------------
* Keeping output schemas consistent lets unimodal, bimodal, trimodal, SurvPGC,
  and late-fusion scripts share plotting and summary code.
"""

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

    # This schema is reused by Kaplan-Meier plotting, result summaries, and
    # late-fusion models that combine unimodal risk scores.
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

    # Store both learned parameters and the run summary so a checkpoint remains
    # interpretable without separately opening summary.json.
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "summary": summary,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
