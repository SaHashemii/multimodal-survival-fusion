#!/usr/bin/env python3
"""Train a unimodal Cox model on frozen BioClinical ModernBERT embeddings."""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT.parent / "5_CV"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a survival Cox model using frozen BERT clinical embeddings.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--clinical-emb-dir", "--embedding-dir", dest="clinical_emb_dir", type=Path, default=None)
    parser.add_argument("--labels-csv", "--labels", dest="labels_csv", type=Path, default=None)
    parser.add_argument("--fold-assignments", type=Path, default=None)
    parser.add_argument("--output-dir", "--out-dir", dest="output_dir", type=Path, default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--pooling", choices=["auto", "mean", "attention", "flatten", "project-concat"], default="mean")
    parser.add_argument("--hidden-dims", type=int, nargs="*", default=[128])
    parser.add_argument("--attention-hidden-dim", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--activation", choices=["selu", "relu"], default="selu")
    parser.add_argument("--dropout", type=float, default=0.30)
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
    args.embedding_dir = args.clinical_emb_dir or args.data_root / "clinical_embeddings"
    args.labels = args.labels_csv or args.data_root / "task3_combined_labels.csv"
    args.fold_assignments = args.fold_assignments or args.data_root / "outputs" / "baseline_5cv_withUro" / "fold_assignments.csv"
    args.out_dir = args.output_dir or PROJECT_ROOT / "outputs" / "bert_embedding_unimodal_cox"
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


def load_embedding_file(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    emb = obj["embeddings"] if isinstance(obj, dict) and "embeddings" in obj else obj
    if isinstance(emb, np.ndarray):
        emb = torch.from_numpy(emb)
    if not torch.is_tensor(emb):
        raise TypeError(f"Unsupported embedding object in {path}")
    emb = emb.float().contiguous()
    if emb.ndim == 1:
        emb = emb.unsqueeze(0)
    if emb.ndim != 2:
        raise ValueError(f"Expected [tokens, dim] or [dim] embedding in {path}, got {tuple(emb.shape)}")
    return emb


def load_embeddings(embedding_dir: Path) -> dict[str, torch.Tensor]:
    files = sorted(embedding_dir.glob("*.pt"))
    if not files:
        raise ValueError(f"No .pt files found in {embedding_dir}")
    embeddings = {}
    for index, path in enumerate(files, start=1):
        embeddings[path.stem] = load_embedding_file(path)
        if index == 1 or index % 50 == 0 or index == len(files):
            print(f"[DATA] loaded {index}/{len(files)} embedding files", end="\r")
    print()
    shapes = sorted({tuple(t.shape) for t in embeddings.values()})
    if len(shapes) != 1:
        raise ValueError(f"Embedding shapes are inconsistent: {shapes}")
    return embeddings


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
        return folds.sort_values("sample_id").reset_index(drop=True)

    y = labels.loc[sample_ids, "Event"].astype(int).values
    folds = pd.DataFrame({"sample_id": sample_ids, "fold": -1})
    rng = np.random.default_rng(42)
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        for fold, split_idx in enumerate(np.array_split(idx, args.n_splits)):
            folds.loc[split_idx, "fold"] = fold
    return folds


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


def stack_embeddings(embeddings: dict[str, torch.Tensor], sample_ids: list[str], device: torch.device) -> torch.Tensor:
    return torch.stack([embeddings[sid] for sid in sample_ids], dim=0).to(device)


def pack_tensors(embeddings: dict[str, torch.Tensor], labels: pd.DataFrame, sample_ids: list[str], device: torch.device):
    return (
        stack_embeddings(embeddings, sample_ids, device),
        torch.tensor(labels.loc[sample_ids, "Time"].values.astype(np.float32), dtype=torch.float32, device=device),
        torch.tensor(labels.loc[sample_ids, "Event"].values.astype(np.float32), dtype=torch.float32, device=device),
    )


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


class BertEmbeddingCox(nn.Module):
    def __init__(
        self,
        token_count: int,
        embedding_dim: int,
        pooling: str,
        hidden_dims: list[int],
        attention_hidden_dim: int,
        projection_dim: int,
        activation: str,
        dropout: float,
    ):
        super().__init__()
        if pooling == "auto":
            pooling = "mean" if token_count > 1 else "flatten"
        self.pooling = pooling
        self.token_count = token_count
        self.embedding_dim = embedding_dim
        if projection_dim < 4:
            raise ValueError("--projection-dim must be at least 4.")
        if pooling == "attention" and token_count == 1:
            pooling = "mean"
            self.pooling = "mean"
        self.feature_net = None
        if pooling == "attention":
            self.attention = nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Linear(embedding_dim, attention_hidden_dim),
                nn.Tanh(),
                nn.Linear(attention_hidden_dim, 1, bias=False),
            )
            head_in = embedding_dim
        elif pooling == "mean":
            self.attention = None
            head_in = embedding_dim
        elif pooling == "flatten":
            self.attention = None
            head_in = token_count * embedding_dim
        elif pooling == "project-concat":
            self.attention = None
            self.feature_net = nn.Sequential(
                nn.Linear(embedding_dim, projection_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(projection_dim // 2, projection_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            head_in = token_count * (projection_dim // 4)
        else:
            raise ValueError(f"Unsupported pooling: {pooling}")

        if activation == "selu":
            act_cls = nn.SELU
            drop_cls = nn.AlphaDropout
        elif activation == "relu":
            act_cls = nn.ReLU
            drop_cls = nn.Dropout
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: list[nn.Module] = [nn.LayerNorm(head_in), nn.Dropout(dropout)]
        prev = head_in
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev, hidden_dim), nn.LayerNorm(hidden_dim), act_cls(), drop_cls(dropout)])
            prev = hidden_dim
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)

    def pooled(self, x: torch.Tensor) -> torch.Tensor:
        if self.pooling == "attention":
            logits = self.attention(x).squeeze(-1)
            weights = torch.softmax(logits, dim=1)
            return torch.sum(weights.unsqueeze(-1) * x, dim=1)
        if self.pooling == "mean":
            return x.mean(dim=1)
        if self.pooling == "project-concat":
            return self.feature_net(x).flatten(start_dim=1)
        return x.flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pooled(x)).squeeze(-1)


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


def run_fold(fold: int, args: argparse.Namespace, device: torch.device, fold_assignments: pd.DataFrame, labels: pd.DataFrame, embeddings: dict[str, torch.Tensor], shape: tuple[int, int]) -> dict:
    train_ids = fold_assignments.loc[fold_assignments["fold"] != fold, "sample_id"].tolist()
    test_ids = fold_assignments.loc[fold_assignments["fold"] == fold, "sample_id"].tolist()
    fit_ids, val_ids = make_inner_validation_split(train_ids, labels, args.seed + fold, args.val_size)

    set_seed(args.seed + fold)
    token_count, embedding_dim = shape
    model = BertEmbeddingCox(
        token_count=token_count,
        embedding_dim=embedding_dim,
        pooling=args.pooling,
        hidden_dims=args.hidden_dims,
        attention_hidden_dim=args.attention_hidden_dim,
        projection_dim=args.projection_dim,
        activation=args.activation,
        dropout=args.dropout,
    ).to(device)

    train_tensors = pack_tensors(embeddings, labels, train_ids, device)
    fit_tensors = pack_tensors(embeddings, labels, fit_ids, device)
    val_tensors = pack_tensors(embeddings, labels, val_ids, device)
    test_tensors = pack_tensors(embeddings, labels, test_ids, device)

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
    }
    with open(fold_dir / "summary.json", "w") as handle:
        json.dump(fold_summary, handle, indent=2)
    torch.save(
        {
            "model_state": model.state_dict(),
            "summary": fold_summary,
            "embedding_shape": shape,
            "pooling": model.pooling,
            "hidden_dims": args.hidden_dims,
            "projection_dim": args.projection_dim,
            "activation": args.activation,
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

    labels = load_labels(args.labels)
    embeddings = load_embeddings(args.embedding_dir)
    common_ids = sorted(set(labels.index) & set(embeddings))
    if len(common_ids) < args.n_splits:
        raise ValueError(f"Too few patients with labels and embeddings: {len(common_ids)}")
    fold_assignments = make_fold_assignments(common_ids, labels, args)
    sample_ids = fold_assignments["sample_id"].astype(str).tolist()
    embeddings = {sid: embeddings[sid] for sid in sample_ids}
    labels = labels.loc[sample_ids]
    shape = tuple(next(iter(embeddings.values())).shape)

    fold_assignments.to_csv(args.out_dir / "fold_assignments.csv", index=False)
    print(f"[DATA] embeddings: {args.embedding_dir}")
    print(f"[DATA] patients used: {len(sample_ids)}")
    print(f"[DATA] embedding shape per patient: {shape}")
    print(f"[MODEL] pooling: {args.pooling}")

    results = []
    for fold in sorted(fold_assignments["fold"].unique()):
        results.append(run_fold(int(fold), args, device, fold_assignments, labels, embeddings, shape))

    results_df = pd.DataFrame(results)
    required_cols = ["fold", "train_n", "test_n", "train_events", "test_events", "c_index"]
    results_df[required_cols].to_csv(args.out_dir / "results_per_fold.csv", index=False)
    results_df.to_csv(args.out_dir / "results_per_fold_detailed.csv", index=False)

    c_indices = results_df["c_index"].astype(float).values
    summary = {
        "strategy": "bioclinical_modernbert_embedding_unimodal_cox",
        "embedding_dir": str(args.embedding_dir),
        "embedding_shape": list(shape),
        "pooling": args.pooling,
        "hidden_dims": args.hidden_dims,
        "projection_dim": args.projection_dim,
        "activation": args.activation,
        "n_splits": int(len(results_df)),
        "n_patients": len(sample_ids),
        "fold_assignments": str(args.fold_assignments) if args.fold_assignments else None,
        "mean_c_index": float(np.mean(c_indices)),
        "std_c_index": float(np.std(c_indices, ddof=1)) if len(c_indices) > 1 else 0.0,
        "per_fold_results": results_df[required_cols].to_dict(orient="records"),
        "fold_c_indices": {str(int(row.fold)): float(row.c_index) for row in results_df.itertuples()},
        "training_style": args.training_style,
        "seed": args.seed,
    }
    with open(args.out_dir / "cross_validation_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
