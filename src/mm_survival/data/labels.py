"""
Label loading utilities for survival prediction
===============================================

Normalizes survival label CSV files to the schema expected by all training
scripts.

Required model schema
---------------------
  sample_id: patient/sample identifier
  Time: follow-up or event time
  Event: 1 for observed event, 0 for censored

Design rationale
----------------
* Source label files may use different column names across cohorts or previous
  scripts.
* Normalization keeps downstream data loaders independent of those naming
  differences.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


LABEL_RENAME_CANDIDATES = {
    "sample": "sample_id",
    "case_id": "sample_id",
    "time": "Time",
    "Time_to_prog_or_FUend": "Time",
    "time_to_prog_or_FUend": "Time",
    "time_to_HG_recur_or_FUend": "Time",
    "duration": "Time",
    "survival_time": "Time",
    "event": "Event",
}

EVENT_TRUE_STRINGS = {
    "1",
    "yes",
    "y",
    "true",
    "t",
    "progression",
    "progressed",
}


def to_event(value: object) -> int:
    """Convert common event encodings to 0/1."""
    if isinstance(value, str):

        # Treat known positive strings as observed events; other strings become
        # censored/non-event labels.
        return 1 if value.strip().lower() in EVENT_TRUE_STRINGS else 0
    return int(value)


def normalize_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize label dataframe columns to sample_id, Time, Event."""
    out = df.copy()
    for old, new in LABEL_RENAME_CANDIDATES.items():
        if old in out.columns and new not in out.columns:

            # Accept common legacy column names without requiring each labels
            # file to be manually edited.
            out = out.rename(columns={old: new})
    if "progression" in out.columns and "Event" not in out.columns:
        out["Event"] = out["progression"].apply(to_event)
    if "sample_id" not in out.columns:

        # If no explicit sample_id column exists, assume the first column stores
        # sample identifiers.
        out = out.rename(columns={out.columns[0]: "sample_id"})

    missing = {"sample_id", "Time", "Event"} - set(out.columns)
    if missing:
        raise ValueError(f"Labels file is missing required columns: {sorted(missing)}")

    out["sample_id"] = out["sample_id"].astype(str)
    labels = out.set_index("sample_id")[["Time", "Event"]].copy()
    labels["Time"] = pd.to_numeric(labels["Time"], errors="coerce")
    labels["Event"] = labels["Event"].apply(to_event)
    return labels.dropna(subset=["Time", "Event"])


def load_labels(path: str | Path) -> pd.DataFrame:
    """Load labels CSV and return an index-by-sample dataframe."""
    return normalize_label_columns(pd.read_csv(path))
