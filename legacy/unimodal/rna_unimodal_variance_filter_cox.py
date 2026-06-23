#!/usr/bin/env python3
"""Train a unimodal Cox model using variance-filtered RNA features."""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn


from rna_extractors.variance_filter import VarianceFilterExtractor  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT.parent / "5_CV"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RNA-only variance-filter Cox model with 5-fold CV.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--labels-csv", "--labels-master", dest="labels_csv", type=Path, default=None)
    parser.add_argument("--rna-csv", "--rna-master", dest="rna_csv", type=Path, default=None)
    parser.add_argument("--fold-assignments", type=Path, default=None)
    parser.add_argument("--output-dir", "--out-dir", dest="output_dir", type=Path, default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--val-size", type=float, default=0.2)

    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--use-cv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rna-hidden-dims", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--rna-out-dim", type=int, default=256)
    parser.add_argument("--rna-dropout", type=float, default=0.25)
    parser.add_argument("--rna-activation", choices=["relu", "gelu", "silu", "selu"], default="selu")
    parser.add_argument("--rna-input-norm", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--head-hidden-dims", type=int, nargs="*", default=[256, 64])
    parser.add_argument("--head-dropout", type=float, default=0.30)
    parser.add_argument("--head-activation", choices=["selu", "relu"], default="selu")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-events-per-batch", type=int, default=3)
    parser.add_argument("--training-style", choices=["full_batch", "event_batch", "random_batch"], default="full_batch")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.labels_master = args.labels_csv or args.data_root / "task3_combined_labels.csv"
    args.rna_master = args.rna_csv or args.data_root / "all_samples_RNA_matrix.csv"
    args.out_dir = args.output_dir or args.data_root / "outputs" / "rna_unimodal_variance_filter_cox"
    return args


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_event(value) -> int:
    if isinstance(value, str):
        return 1 if value.strip().lower() in {"1", "yes", "y", "true", "t", "progression", "progressed"} else 0
    return int(value)


def load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename_candidates = {
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
    for old, new in rename_candidates.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    if "progression" in df.columns and "Event" not in df.columns:
        df["Event"] = df["progression"].apply(to_event)
    if "sample_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "sample_id"})
    missing = {"sample_id", "Time", "Event"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required label columns: {sorted(missing)}")
    df["sample_id"] = df["sample_id"].astype(str)
    labels = df.set_index("sample_id")[["Time", "Event"]].copy()
    labels["Time"] = pd.to_numeric(labels["Time"], errors="coerce")
    labels["Event"] = labels["Event"].apply(to_event)
    return labels.dropna(subset=["Time", "Event"])


def load_rna(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    rna_t = df.T
    rna_t.index.name = "sample_id"
    rna_t.columns = rna_t.columns.astype(str)
    return rna_t.apply(pd.to_numeric, errors="coerce")


def make_fold_assignments(sample_ids: list[str], labels: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.fold_assignments and args.fold_assignments.is_file():
        folds = pd.read_csv(args.fold_assignments)
        if not {"sample_id", "fold"}.issubset(folds.columns):
            raise ValueError(f"{args.fold_assignments} must contain sample_id and fold columns.")
        folds["sample_id"] = folds["sample_id"].astype(str)
        folds = folds[folds["sample_id"].isin(sample_ids)].copy()
        missing = sorted(set(sample_ids) - set(folds["sample_id"]))
        if missing:
            print(f"[CV] using existing folds for {len(folds)} patients; {len(missing)} common patients are not in fold_assignments.")
        if folds.empty:
            raise ValueError("No patients remain after applying fold assignments.")
        return folds.sort_values("sample_id").reset_index(drop=True)

    y = labels.loc[sample_ids, "Event"].astype(int).values
    splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    fold_assignments = pd.DataFrame({"sample_id": sample_ids, "fold": -1})
    for fold, (_, test_idx) in enumerate(splitter.split(np.zeros(len(sample_ids)), y)):
        fold_assignments.loc[test_idx, "fold"] = fold
    return fold_assignments


def make_inner_validation_split(train_ids: list[str], labels: pd.DataFrame, seed: int, val_size: float) -> tuple[list[str], list[str]]:
    y = labels.loc[train_ids, "Event"].astype(int).values
    class_counts = np.bincount(y, minlength=2)
    if class_counts.min() < 2:
        raise ValueError(f"Cannot create stratified validation split; event counts are {class_counts.tolist()}.")
    rng = np.random.default_rng(seed)
    val_indices = []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        n_val = max(1, int(math.ceil(len(idx) * val_size)))
        n_val = min(n_val, len(idx) - 1)
        val_indices.extend(idx[:n_val].tolist())
    val_set = set(val_indices)
    fit_ids = [sid for i, sid in enumerate(train_ids) if i not in val_set]
    val_ids = [sid for i, sid in enumerate(train_ids) if i in val_set]
    return fit_ids, val_ids


def impute_rna_with_medians(data: pd.DataFrame, medians: pd.Series) -> np.ndarray:
    return data.fillna(medians).values.astype(np.float32)


def build_rna_extractor(args: argparse.Namespace, gene_names: list[str], xrna_fit: np.ndarray) -> VarianceFilterExtractor:
    extractor = VarianceFilterExtractor(
        input_dim=len(gene_names),
        config={
            "top_k": args.top_k,
            "use_cv": args.use_cv,
            "cv_percentile": None,
            "eps": 1e-8,
            "hidden_dims": args.rna_hidden_dims,
            "out_dim": args.rna_out_dim,
            "dropout": args.rna_dropout,
            "activation": args.rna_activation,
            "norm": "layer",
            "input_norm": args.rna_input_norm,
            "gene_names": gene_names,
        },
    )
    extractor.fit(torch.tensor(xrna_fit, dtype=torch.float32))
    return extractor


def cox_loss(risk: torch.Tensor, event: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(time, descending=True)
    r = risk[order]
    e = event[order]
    log_cumsum = torch.logcumsumexp(r, dim=0)
    observed = e == 1
    if observed.sum() == 0:
        return torch.zeros((), device=risk.device, requires_grad=True)
    return -((r[observed] - log_cumsum[observed]).sum() / observed.sum())


def concordance_index(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    concordant = 0.0
    permissible = 0.0
    for i in range(len(risk)):
        for j in range(len(risk)):
            if time[i] < time[j] and event[i] == 1:
                permissible += 1
                if risk[i] > risk[j]:
                    concordant += 1
                elif risk[i] == risk[j]:
                    concordant += 0.5
    return float(concordant / permissible) if permissible else 0.5


def event_aware_batch_indices(events: torch.Tensor, batch_size: int, min_events: int) -> list[torch.Tensor]:
    ev_idx = torch.where(events == 1)[0]
    no_idx = torch.where(events == 0)[0]
    ev_idx = ev_idx[torch.randperm(len(ev_idx), device=events.device)]
    no_idx = no_idx[torch.randperm(len(no_idx), device=events.device)]
    n = len(events)
    n_ev_per_batch = max(min_events, int(batch_size * (len(ev_idx) / max(n, 1))))
    n_ev_per_batch = min(n_ev_per_batch, max(len(ev_idx), 1))
    n_no_per_batch = max(batch_size - n_ev_per_batch, 0)
    batches = []
    ev_ptr = no_ptr = 0
    while ev_ptr < len(ev_idx):
        ev_chunk = ev_idx[ev_ptr : ev_ptr + n_ev_per_batch]
        if len(ev_chunk) == 0:
            break
        if len(no_idx) > 0 and n_no_per_batch > 0:
            no_chunk = no_idx[no_ptr : no_ptr + n_no_per_batch]
            if len(no_chunk) < n_no_per_batch:
                no_chunk = torch.cat([no_chunk, no_idx[: n_no_per_batch - len(no_chunk)]])
            no_ptr = (no_ptr + n_no_per_batch) % len(no_idx)
        else:
            no_chunk = torch.tensor([], dtype=torch.long, device=events.device)
        batch = torch.cat([ev_chunk, no_chunk])
        batches.append(batch[torch.randperm(len(batch), device=events.device)] if len(batch) > 1 else batch)
        ev_ptr += n_ev_per_batch
    return batches


class RNACox(nn.Module):
    def __init__(self, rna_extractor: VarianceFilterExtractor, hidden_dims: list[int], dropout: float, activation: str):
        super().__init__()
        self.rna_extractor = rna_extractor
        if activation == "selu":
            act_cls = nn.SELU
            drop_cls = nn.AlphaDropout
        elif activation == "relu":
            act_cls = nn.ReLU
            drop_cls = nn.Dropout
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        layers: list[nn.Module] = [nn.LayerNorm(rna_extractor.output_dim), nn.Dropout(dropout)]
        prev = rna_extractor.output_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev, hidden_dim), nn.LayerNorm(hidden_dim), act_cls(), drop_cls(dropout)])
            prev = hidden_dim
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, rna: torch.Tensor) -> torch.Tensor:
        return self.head(self.rna_extractor(rna)).squeeze(-1)


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def pack_tensors(xrna: np.ndarray, labels: pd.DataFrame, sample_ids: list[str], device: torch.device):
    return (
        to_tensor(xrna, device),
        torch.tensor(labels.loc[sample_ids, "Time"].values.astype(np.float32), dtype=torch.float32, device=device),
        torch.tensor(labels.loc[sample_ids, "Event"].values.astype(np.float32), dtype=torch.float32, device=device),
    )


def train_model(model, train_tensors, args, device, val_tensors=None):
    x_tr, time_tr, event_tr = train_tensors
    if val_tensors is not None:
        x_va, time_va, event_va = val_tensors
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = deepcopy(model.state_dict())
    best_loss = math.inf
    wait = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        if args.training_style == "full_batch":
            batches = [torch.arange(len(x_tr), device=device)]
        elif args.training_style == "event_batch":
            batches = event_aware_batch_indices(event_tr, args.batch_size, args.min_events_per_batch)
        else:
            perm = torch.randperm(len(x_tr), device=device)
            batches = [perm[start : start + args.batch_size] for start in range(0, len(x_tr), args.batch_size)]
        for idx in batches:
            opt.zero_grad()
            risk = model(x_tr[idx])
            loss = cox_loss(risk, event_tr[idx], time_tr[idx])
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            train_risk_tensor = model(x_tr)
            train_full_loss = float(cox_loss(train_risk_tensor, event_tr, time_tr).detach().cpu())
            train_ci = concordance_index(train_risk_tensor.cpu().numpy(), time_tr.cpu().numpy(), event_tr.cpu().numpy())
            val_loss = math.nan
            val_ci = math.nan
            if val_tensors is not None:
                val_risk_tensor = model(x_va)
                val_loss = float(cox_loss(val_risk_tensor, event_va, time_va).detach().cpu())
                val_ci = concordance_index(val_risk_tensor.cpu().numpy(), time_va.cpu().numpy(), event_va.cpu().numpy())
        mean_loss = float(np.mean(losses)) if losses else math.nan
        monitor_loss = val_loss if val_tensors is not None else mean_loss
        history.append(
            {
                "epoch": epoch,
                "train_loss": mean_loss,
                "train_full_loss": train_full_loss,
                "train_ci": train_ci,
                "val_loss": val_loss,
                "val_ci": val_ci,
                "monitor_loss": monitor_loss,
            }
        )
        if monitor_loss < best_loss:
            best_loss = monitor_loss
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def evaluate_model(model, tensors, labels: pd.DataFrame, sample_ids: list[str]) -> tuple[float, pd.DataFrame]:
    x, _, _ = tensors
    model.eval()
    with torch.no_grad():
        risk = model(x).detach().cpu().numpy()
    time = labels.loc[sample_ids, "Time"].values.astype(float)
    event = labels.loc[sample_ids, "Event"].values.astype(int)
    ci = concordance_index(risk, time, event)
    return ci, pd.DataFrame({"sample_id": sample_ids, "log_risk": risk, "Event": event, "Time": time})


def run_fold(fold: int, args: argparse.Namespace, device: torch.device, fold_assignments: pd.DataFrame, rna: pd.DataFrame, labels: pd.DataFrame) -> dict:
    train_ids = fold_assignments.loc[fold_assignments["fold"] != fold, "sample_id"].tolist()
    test_ids = fold_assignments.loc[fold_assignments["fold"] == fold, "sample_id"].tolist()
    fit_ids, val_ids = make_inner_validation_split(train_ids, labels, args.seed + fold, args.val_size)
    gene_names = rna.columns.astype(str).tolist()

    fit_rna = rna.loc[fit_ids, gene_names]
    rna_medians_series = fit_rna.median(axis=0, skipna=True).fillna(0.0)
    xrna_fit = impute_rna_with_medians(fit_rna, rna_medians_series)
    xrna_train = impute_rna_with_medians(rna.loc[train_ids, gene_names], rna_medians_series)
    xrna_test = impute_rna_with_medians(rna.loc[test_ids, gene_names], rna_medians_series)
    xrna_val = impute_rna_with_medians(rna.loc[val_ids, gene_names], rna_medians_series)
    rna_medians = {str(k): float(v) for k, v in rna_medians_series.items()}

    set_seed(args.seed + fold)
    rna_extractor = build_rna_extractor(args, gene_names, xrna_fit)
    selected_genes = [str(gene) for gene in rna_extractor.get_selected_genes()]
    model = RNACox(rna_extractor, args.head_hidden_dims, args.head_dropout, args.head_activation).to(device)

    train_tensors = pack_tensors(xrna_train, labels, train_ids, device)
    fit_tensors = pack_tensors(xrna_fit, labels, fit_ids, device)
    val_tensors = pack_tensors(xrna_val, labels, val_ids, device)
    test_tensors = pack_tensors(xrna_test, labels, test_ids, device)

    print(
        f"[FOLD {fold}] train_n={len(train_ids)} train_events={int(labels.loc[train_ids, 'Event'].sum())} "
        f"val_n={len(val_ids)} val_events={int(labels.loc[val_ids, 'Event'].sum())} "
        f"test_n={len(test_ids)} test_events={int(labels.loc[test_ids, 'Event'].sum())}"
    )
    model, history = train_model(model, fit_tensors, args, device, val_tensors=val_tensors)

    fold_dir = args.out_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(fold_dir / "history.csv", index=False)
    train_ci, train_risk = evaluate_model(model, train_tensors, labels, train_ids)
    test_ci, test_risk = evaluate_model(model, test_tensors, labels, test_ids)
    train_risk.to_csv(fold_dir / "train_risk_scores.csv", index=False)
    test_risk.to_csv(fold_dir / "test_risk_scores.csv", index=False)
    pd.Series(selected_genes).to_csv(args.out_dir / f"selected_genes_fold_{fold}.txt", index=False, header=False)
    with open(fold_dir / "rna_imputation_medians.json", "w") as handle:
        json.dump(rna_medians, handle, indent=2)

    fold_summary = {
        "fold": fold,
        "c_index": test_ci,
        "train_c_index": train_ci,
        "train_n": len(train_ids),
        "test_n": len(test_ids),
        "train_events": int(labels.loc[train_ids, "Event"].sum()),
        "test_events": int(labels.loc[test_ids, "Event"].sum()),
        "early_stop_fit_n": len(fit_ids),
        "early_stop_val_n": len(val_ids),
        "early_stop_fit_events": int(labels.loc[fit_ids, "Event"].sum()),
        "early_stop_val_events": int(labels.loc[val_ids, "Event"].sum()),
        "selected_genes": selected_genes,
        "selected_gene_count": len(selected_genes),
    }
    with open(fold_dir / "summary.json", "w") as handle:
        json.dump(fold_summary, handle, indent=2)
    torch.save(
        {
            "model_state": model.state_dict(),
            "summary": fold_summary,
            "selected_genes": selected_genes,
            "rna_hidden_dims": args.rna_hidden_dims,
            "rna_out_dim": args.rna_out_dim,
            "head_hidden_dims": args.head_hidden_dims,
        },
        fold_dir / "model.pt",
    )
    print(f"[FOLD {fold}] test_c_index={test_ci:.4f}")
    return fold_summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    labels = load_labels(args.labels_master)
    rna = load_rna(args.rna_master)
    sample_ids = sorted(set(labels.index) & set(rna.index))
    if len(sample_ids) < args.n_splits:
        raise ValueError(f"Too few patients with labels and RNA: {len(sample_ids)}")
    rna = rna.loc[sample_ids]
    labels = labels.loc[sample_ids]
    fold_assignments = make_fold_assignments(sample_ids, labels, args)
    sample_ids = fold_assignments["sample_id"].astype(str).tolist()
    rna = rna.loc[sample_ids]
    labels = labels.loc[sample_ids]
    fold_assignments.to_csv(args.out_dir / "fold_assignments.csv", index=False)

    print(f"[DATA] RNA: {args.rna_master}")
    print(f"[DATA] labels: {args.labels_master}")
    print(f"[DATA] patients used: {len(sample_ids)}")
    print(f"[DATA] input genes: {rna.shape[1]}")
    print(f"[RNA] top_k: {args.top_k} use_cv: {args.use_cv}")

    results = []
    for fold in sorted(fold_assignments["fold"].unique()):
        results.append(run_fold(int(fold), args, device, fold_assignments, rna, labels))

    results_df = pd.DataFrame(results)
    results_df.pop("selected_genes")
    results_df["selected_genes_file"] = [f"selected_genes_fold_{fold}.txt" for fold in results_df["fold"]]
    required_cols = ["fold", "train_n", "test_n", "train_events", "test_events", "c_index"]
    results_df[required_cols].to_csv(args.out_dir / "results_per_fold.csv", index=False)
    results_df.to_csv(args.out_dir / "results_per_fold_detailed.csv", index=False)

    c_indices = results_df["c_index"].astype(float).values
    summary = {
        "strategy": "rna_unimodal_variance_filter_cox",
        "rna_master": str(args.rna_master),
        "labels_master": str(args.labels_master),
        "fold_assignments": str(args.fold_assignments) if args.fold_assignments else None,
        "n_splits": int(len(results_df)),
        "n_patients": len(sample_ids),
        "input_genes": int(rna.shape[1]),
        "top_k": args.top_k,
        "use_cv": args.use_cv,
        "mean_c_index": float(np.mean(c_indices)),
        "std_c_index": float(np.std(c_indices, ddof=1)) if len(c_indices) > 1 else 0.0,
        "per_fold_results": results_df[required_cols].to_dict(orient="records"),
        "fold_c_indices": {str(int(row.fold)): float(row.c_index) for row in results_df.itertuples()},
        "selected_gene_counts": {str(int(row.fold)): int(row.selected_gene_count) for row in results_df.itertuples()},
        "rna_hidden_dims": args.rna_hidden_dims,
        "rna_out_dim": args.rna_out_dim,
        "rna_dropout": args.rna_dropout,
        "head_hidden_dims": args.head_hidden_dims,
        "head_dropout": args.head_dropout,
        "training_style": args.training_style,
        "seed": args.seed,
    }
    with open(args.out_dir / "cross_validation_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
