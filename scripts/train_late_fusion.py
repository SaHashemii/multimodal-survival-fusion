#!/usr/bin/env python3
"""Fuse saved unimodal risk scores at the decision level."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd

from mm_survival.training.artifacts import ensure_dir, write_json
from mm_survival.training.metrics import concordance_index
from mm_survival.training.plots import write_kaplan_meier_plot
from mm_survival.utils.config import load_yaml, resolve_path


MODALITIES = ("rna", "pathology", "clinical")
RISK_COLUMNS = {"sample_id", "log_risk", "Event", "Time"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run late fusion from saved unimodal risk scores.")
    parser.add_argument("--experiment", type=Path, required=True, help="Late-fusion experiment config YAML.")
    parser.add_argument("--data", type=Path, required=True, help="Data config YAML. Used for consistent CLI shape.")
    parser.add_argument("--rna-output-dir", type=Path, default=None, help="Override RNA unimodal output directory.")
    parser.add_argument("--pathology-output-dir", type=Path, default=None, help="Override pathology unimodal output directory.")
    parser.add_argument("--clinical-output-dir", type=Path, default=None, help="Override clinical unimodal output directory.")
    parser.add_argument(
        "--method",
        choices=("mean", "learned_cox"),
        default=None,
        help="Optional method override. If omitted, all methods listed in the YAML are run.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Override late-fusion output directory.")
    return parser.parse_args()


def read_risk_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing risk-score file: {path}")
    table = pd.read_csv(path)
    missing = RISK_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    table = table[["sample_id", "log_risk", "Event", "Time"]].copy()
    table["sample_id"] = table["sample_id"].astype(str)
    return table


def rename_modality_risk(table: pd.DataFrame, modality: str) -> pd.DataFrame:
    return table.rename(columns={"log_risk": f"{modality}_risk"})


def merge_modality_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for modality in MODALITIES:
        table = rename_modality_risk(tables[modality], modality)
        keep_cols = ["sample_id", f"{modality}_risk", "Event", "Time"]
        if merged is None:
            merged = table[keep_cols]
        else:
            merged = merged.merge(
                table[keep_cols],
                on=["sample_id", "Event", "Time"],
                how="inner",
                validate="one_to_one",
            )
    if merged is None or merged.empty:
        raise ValueError("No shared samples found across modality risk-score tables.")
    return merged.sort_values("sample_id").reset_index(drop=True)


def available_folds(input_dirs: dict[str, Path]) -> list[int]:
    fold_sets = []
    for path in input_dirs.values():
        folds = set()
        for fold_dir in path.glob("fold_*"):
            if not fold_dir.is_dir():
                continue
            try:
                folds.add(int(fold_dir.name.split("_", 1)[1]))
            except ValueError:
                continue
        fold_sets.append(folds)
    common = set.intersection(*fold_sets) if fold_sets else set()
    return sorted(common)


def load_fold_tables(input_dirs: dict[str, Path], fold: int, split: str) -> pd.DataFrame:
    tables = {
        modality: read_risk_table(input_dirs[modality] / f"fold_{fold}" / f"{split}_risk_scores.csv")
        for modality in MODALITIES
    }
    return merge_modality_tables(tables)


def cox_gradient(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> tuple[float, np.ndarray]:
    exp_risk = np.exp(risk - np.max(risk))
    event_idx = np.where(event == 1)[0]
    if len(event_idx) == 0:
        return 0.0, np.zeros_like(risk)

    loss = 0.0
    grad = np.zeros_like(risk)
    for i in event_idx:
        at_risk = time >= time[i]
        denom = exp_risk[at_risk].sum()
        if denom <= 0:
            continue
        loss -= risk[i] - (np.log(denom) + np.max(risk))
        grad[i] -= 1.0
        grad[at_risk] += exp_risk[at_risk] / denom
    scale = float(len(event_idx))
    return loss / scale, grad / scale


def fit_learned_cox(
    train_features: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    *,
    epochs: int = 1000,
    lr: float = 0.05,
    weight_decay: float = 0.01,
) -> tuple[np.ndarray, dict[str, float]]:
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std == 0.0] = 1.0
    x = (train_features - mean) / std
    weights = np.zeros(x.shape[1], dtype=float)

    best_loss = float("inf")
    best_weights = weights.copy()
    m = np.zeros_like(weights)
    v = np.zeros_like(weights)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    for step in range(1, epochs + 1):
        risk = x @ weights
        loss, grad_risk = cox_gradient(risk, time, event)
        grad = x.T @ grad_risk + weight_decay * weights
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        weights -= lr * m_hat / (np.sqrt(v_hat) + eps)
        if loss < best_loss:
            best_loss = loss
            best_weights = weights.copy()

    packed = np.concatenate([best_weights, mean, std])
    metadata = {
        "train_loss": float(best_loss),
        "rna_weight": float(best_weights[0]),
        "pathology_weight": float(best_weights[1]),
        "clinical_weight": float(best_weights[2]),
    }
    return packed, metadata


def predict_learned_cox(features: np.ndarray, packed: np.ndarray) -> np.ndarray:
    n_features = features.shape[1]
    weights = packed[:n_features]
    mean = packed[n_features : 2 * n_features]
    std = packed[2 * n_features :]
    return ((features - mean) / std) @ weights


def make_risk_output(merged: pd.DataFrame, risk: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": merged["sample_id"].astype(str),
            "log_risk": risk.astype(float),
            "Event": merged["Event"].astype(int),
            "Time": merged["Time"].astype(float),
            "rna_risk": merged["rna_risk"].astype(float),
            "pathology_risk": merged["pathology_risk"].astype(float),
            "clinical_risk": merged["clinical_risk"].astype(float),
        }
    )


def summarize_method(method_dir: Path, method: str, fold_results: list[dict], test_tables: list[pd.DataFrame]) -> None:
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(method_dir / "results_per_fold.csv", index=False)
    all_test = pd.concat(test_tables, ignore_index=True)
    all_test.to_csv(method_dir / "test_risk_scores_all_folds.csv", index=False)
    km_stats = write_kaplan_meier_plot(all_test, method_dir / "kaplan_meier_by_risk.png", title=f"Late fusion {method}")
    summary = {
        "method": method,
        "folds": int(results_df.shape[0]),
        "mean_c_index": float(results_df["c_index"].mean()),
        "std_c_index": float(results_df["c_index"].std(ddof=1)) if results_df.shape[0] > 1 else 0.0,
        "kaplan_meier": km_stats,
    }
    write_json(summary, method_dir / "summary.json")


def run_mean(input_dirs: dict[str, Path], folds: list[int], output_dir: Path) -> dict:
    method_dir = ensure_dir(output_dir / "mean")
    fold_results = []
    test_tables = []
    for fold in folds:
        merged = load_fold_tables(input_dirs, fold, "test")
        risk = merged[["rna_risk", "pathology_risk", "clinical_risk"]].to_numpy(float).mean(axis=1)
        risk_table = make_risk_output(merged, risk)
        c_index = concordance_index(risk_table["log_risk"].to_numpy(float), risk_table["Time"].to_numpy(float), risk_table["Event"].to_numpy(int))

        fold_dir = ensure_dir(method_dir / f"fold_{fold}")
        risk_table.to_csv(fold_dir / "test_risk_scores.csv", index=False)
        summary = {
            "fold": int(fold),
            "test_n": int(risk_table.shape[0]),
            "test_events": int(risk_table["Event"].sum()),
            "c_index": float(c_index),
        }
        write_json(summary, fold_dir / "summary.json")
        fold_results.append(summary)
        test_tables.append(risk_table.assign(fold=fold))
    summarize_method(method_dir, "mean", fold_results, test_tables)
    return {"method": "mean", "output_dir": str(method_dir)}


def run_learned_cox(input_dirs: dict[str, Path], folds: list[int], output_dir: Path) -> dict:
    method_dir = ensure_dir(output_dir / "learned_cox")
    fold_results = []
    test_tables = []
    for fold in folds:
        train = load_fold_tables(input_dirs, fold, "train")
        test = load_fold_tables(input_dirs, fold, "test")
        feature_cols = ["rna_risk", "pathology_risk", "clinical_risk"]
        packed, fit_metadata = fit_learned_cox(
            train[feature_cols].to_numpy(float),
            train["Time"].to_numpy(float),
            train["Event"].to_numpy(int),
        )
        risk = predict_learned_cox(test[feature_cols].to_numpy(float), packed)
        risk_table = make_risk_output(test, risk)
        c_index = concordance_index(risk_table["log_risk"].to_numpy(float), risk_table["Time"].to_numpy(float), risk_table["Event"].to_numpy(int))

        fold_dir = ensure_dir(method_dir / f"fold_{fold}")
        risk_table.to_csv(fold_dir / "test_risk_scores.csv", index=False)
        with (fold_dir / "weights.json").open("w", encoding="utf-8") as handle:
            json.dump(fit_metadata, handle, indent=2)
        summary = {
            "fold": int(fold),
            "test_n": int(risk_table.shape[0]),
            "test_events": int(risk_table["Event"].sum()),
            "c_index": float(c_index),
            **fit_metadata,
        }
        write_json(summary, fold_dir / "summary.json")
        fold_results.append(summary)
        test_tables.append(risk_table.assign(fold=fold))
    summarize_method(method_dir, "learned_cox", fold_results, test_tables)
    return {"method": "learned_cox", "output_dir": str(method_dir)}


def main() -> None:
    args = parse_args()
    exp_cfg = load_yaml(args.experiment)
    load_yaml(args.data)

    exp_info = exp_cfg.get("experiment", {})
    if exp_info.get("model_type") != "late_fusion":
        raise ValueError("scripts/train_late_fusion.py requires model_type=late_fusion.")

    output_dir = ensure_dir(args.output_dir or resolve_path(REPO_ROOT, exp_info.get("output_dir", "outputs/late_fusion")))
    input_cfg = exp_cfg.get("inputs", {})
    input_dirs = {
        "rna": args.rna_output_dir or resolve_path(REPO_ROOT, input_cfg["rna_output_dir"]),
        "pathology": args.pathology_output_dir or resolve_path(REPO_ROOT, input_cfg["pathology_output_dir"]),
        "clinical": args.clinical_output_dir or resolve_path(REPO_ROOT, input_cfg["clinical_output_dir"]),
    }
    methods = [args.method] if args.method else list(exp_cfg.get("fusion", {}).get("methods", ["mean"]))
    folds = available_folds(input_dirs)
    if not folds:
        raise ValueError("No common fold_* directories found across the three unimodal output folders.")

    summaries = []
    for method in methods:
        if method == "mean":
            summaries.append(run_mean(input_dirs, folds, output_dir))
        elif method == "learned_cox":
            summaries.append(run_learned_cox(input_dirs, folds, output_dir))
        else:
            raise ValueError(f"Unsupported late-fusion method: {method}")

    write_json({"methods": summaries, "folds": folds}, output_dir / "summary.json")


if __name__ == "__main__":
    main()
