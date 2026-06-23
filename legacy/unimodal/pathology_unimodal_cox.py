#!/usr/bin/env python3
"""Train a pathology-only Cox model using UNI tile features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT.parent / "5_CV"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pathology-only Cox model with 5-fold CV.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--labels-csv", "--labels-master", dest="labels_csv", type=Path, default=None)
    parser.add_argument("--pathology-index", type=Path, default=None)
    parser.add_argument("--fold-assignments", type=Path, default=None)
    parser.add_argument("--output-dir", "--out-dir", dest="output_dir", type=Path, default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--val-size", type=float, default=0.2)

    parser.add_argument("--he-aggregator", choices=["gated", "mean", "mean+std"], default="gated")
    parser.add_argument("--he-attn-dim", type=int, default=128)
    parser.add_argument("--he-emb-dim", type=int, default=256)
    parser.add_argument("--he-dropout", type=float, default=0.20)
    parser.add_argument("--he-tile-cap", type=int, default=4096)

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
    args.pathology_index = args.pathology_index or args.data_root / "he_paths_all_splits_minimal.csv"
    args.out_dir = args.output_dir or args.data_root / "outputs" / "pathology_unimodal_cox"
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


def load_pathology_index(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "source_path_sol" in df.columns and "feature_path" not in df.columns:
        df = df.rename(columns={"source_path_sol": "feature_path"})
    missing = {"sample_id", "feature_path"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required pathology index columns: {sorted(missing)}")
    df["sample_id"] = df["sample_id"].astype(str)
    original_sample_ids = df["sample_id"].copy()
    df["sample_id"] = df["sample_id"].str.replace(r"^2U_", "3U_", regex=True)
    df["sample_id"] = df["sample_id"].str.replace(r"^2B_", "3B_", regex=True)
    remapped = int((original_sample_ids != df["sample_id"]).sum())
    if remapped:
        print(f"[DATA] remapped {remapped} pathology sample IDs from 2U_/2B_ to 3U_/3B_ for matching")
    df["feature_path"] = df["feature_path"].astype(str)
    return df.drop_duplicates("sample_id").set_index("sample_id")


def load_pathology_bag(path: Path) -> torch.Tensor | None:
    try:
        obj = torch.load(path, map_location="cpu")
    except OSError as exc:
        print(f"[Pathology] WARN: cannot load {path}: {exc}")
        return None
    feats = obj["feats"] if isinstance(obj, dict) and "feats" in obj else obj
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


def cap_tiles(feats: torch.Tensor, sample_id: str, seed: int, tile_cap: int | None) -> torch.Tensor:
    if tile_cap is None or tile_cap <= 0 or feats.shape[0] <= tile_cap:
        return feats
    digest = hashlib.sha256(f"{sample_id}:{seed}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little") % (2**32))
    idx = rng.choice(feats.shape[0], size=tile_cap, replace=False)
    idx.sort()
    return feats[idx].contiguous()


def load_master_dataset(args: argparse.Namespace) -> tuple[list[str], pd.DataFrame, dict[str, torch.Tensor]]:
    labels = load_labels(args.labels_master)
    pathology = load_pathology_index(args.pathology_index)
    common = sorted(set(labels.index) & set(pathology.index))
    if len(common) < args.n_splits:
        raise ValueError(f"Too few patients with labels and pathology index rows: {len(common)}")

    he_bags: dict[str, torch.Tensor] = {}
    missing_features = []
    for sid in common:
        feature_path = Path(pathology.loc[sid, "feature_path"])
        if not feature_path.is_absolute():
            feature_path = args.pathology_index.parent / feature_path
        bag = load_pathology_bag(feature_path)
        if bag is None:
            missing_features.append(sid)
            continue
        he_bags[sid] = cap_tiles(bag, sid, args.seed, args.he_tile_cap)

    sample_ids = sorted(set(common) & set(he_bags))
    if len(sample_ids) < args.n_splits:
        raise ValueError(f"Too few patients with loadable pathology features: {len(sample_ids)}")
    print("[DATA] modality patient counts:")
    print(f"  Labels: {len(labels)}")
    print(f"  Pathology index: {len(pathology)}")
    print(f"  Retained pathology+labels: {len(sample_ids)}")
    print(f"  Dropped missing/invalid pathology bags: {len(missing_features)}")
    return sample_ids, labels.loc[sample_ids], he_bags


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


class GatedAttentionMIL(nn.Module):
    def __init__(self, in_dim: int = 1024, attn_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.v = nn.Linear(in_dim, attn_dim)
        self.u = nn.Linear(in_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor):
        if h.ndim != 2:
            raise ValueError(f"GatedAttentionMIL expects (tiles, dim), got {tuple(h.shape)}")
        hn = self.drop(self.norm(h.float()))
        logits = self.w(torch.tanh(self.v(hn)) * torch.sigmoid(self.u(hn))).squeeze(-1)
        weights = torch.softmax(logits - logits.max(), dim=0)
        z = torch.sum(weights.unsqueeze(1) * hn, dim=0)
        return z, weights


class HEEncoder(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int, aggregator: str, attn_dim: int, dropout: float):
        super().__init__()
        self.aggregator = aggregator
        if aggregator == "gated":
            self.pool = GatedAttentionMIL(in_dim=in_dim, attn_dim=attn_dim, dropout=dropout)
            proj_in = in_dim
        elif aggregator == "mean":
            self.pool = None
            proj_in = in_dim
        elif aggregator == "mean+std":
            self.pool = None
            proj_in = in_dim * 2
        else:
            raise ValueError(f"Unknown HE aggregator: {aggregator}")
        self.proj = nn.Sequential(
            nn.LayerNorm(proj_in),
            nn.Linear(proj_in, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, h: torch.Tensor):
        if self.aggregator == "gated":
            z, weights = self.pool(h)
        elif self.aggregator == "mean":
            z = h.float().mean(dim=0)
            weights = None
        else:
            z = torch.cat([h.float().mean(dim=0), h.float().std(dim=0, unbiased=False)], dim=0)
            weights = None
        return self.proj(z), weights


class PathologyCox(nn.Module):
    def __init__(
        self,
        he_in_dim: int,
        he_emb_dim: int,
        he_aggregator: str,
        he_attn_dim: int,
        he_dropout: float,
        head_hidden_dims: list[int],
        head_dropout: float,
        head_activation: str,
    ):
        super().__init__()
        self.he_encoder = HEEncoder(he_in_dim, he_emb_dim, he_aggregator, he_attn_dim, he_dropout)
        if head_activation == "selu":
            act_cls = nn.SELU
            drop_cls = nn.AlphaDropout
        elif head_activation == "relu":
            act_cls = nn.ReLU
            drop_cls = nn.Dropout
        else:
            raise ValueError(f"Unsupported activation: {head_activation}")
        layers: list[nn.Module] = [nn.LayerNorm(he_emb_dim), nn.Dropout(head_dropout)]
        prev = he_emb_dim
        for hidden_dim in head_hidden_dims:
            layers.extend([nn.Linear(prev, hidden_dim), nn.LayerNorm(hidden_dim), act_cls(), drop_cls(head_dropout)])
            prev = hidden_dim
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)

    def forward_all(self, he_bags: list[torch.Tensor], device: torch.device) -> torch.Tensor:
        risks = []
        for bag in he_bags:
            he_emb, _ = self.he_encoder(bag.to(device))
            risks.append(self.head(he_emb).squeeze())
        return torch.stack(risks)


def pack_labels(labels: pd.DataFrame, sample_ids: list[str], device: torch.device):
    return (
        torch.tensor(labels.loc[sample_ids, "Time"].values.astype(np.float32), dtype=torch.float32, device=device),
        torch.tensor(labels.loc[sample_ids, "Event"].values.astype(np.float32), dtype=torch.float32, device=device),
    )


def train_model(model, he_train, label_tensors, args, device, he_val=None, val_label_tensors=None):
    time_tr, event_tr = label_tensors
    if val_label_tensors is not None:
        time_va, event_va = val_label_tensors
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = deepcopy(model.state_dict())
    best_loss = math.inf
    wait = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        if args.training_style == "full_batch":
            batches = [torch.arange(len(he_train), device=device)]
        elif args.training_style == "event_batch":
            batches = event_aware_batch_indices(event_tr, args.batch_size, args.min_events_per_batch)
        else:
            perm = torch.randperm(len(he_train), device=device)
            batches = [perm[start : start + args.batch_size] for start in range(0, len(he_train), args.batch_size)]
        for idx in batches:
            batch_ids = idx.detach().cpu().tolist()
            batch_bags = [he_train[i] for i in batch_ids]
            opt.zero_grad()
            risk = model.forward_all(batch_bags, device)
            loss = cox_loss(risk, event_tr[idx], time_tr[idx])
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            train_risk_tensor = model.forward_all(he_train, device)
            train_full_loss = float(cox_loss(train_risk_tensor, event_tr, time_tr).detach().cpu())
            train_ci = concordance_index(train_risk_tensor.cpu().numpy(), time_tr.cpu().numpy(), event_tr.cpu().numpy())
            val_loss = math.nan
            val_ci = math.nan
            if he_val is not None and val_label_tensors is not None:
                val_risk_tensor = model.forward_all(he_val, device)
                val_loss = float(cox_loss(val_risk_tensor, event_va, time_va).detach().cpu())
                val_ci = concordance_index(val_risk_tensor.cpu().numpy(), time_va.cpu().numpy(), event_va.cpu().numpy())
        mean_loss = float(np.mean(losses)) if losses else math.nan
        monitor_loss = val_loss if he_val is not None and val_label_tensors is not None else mean_loss
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


def evaluate_model(model, he_bags: list[torch.Tensor], labels: pd.DataFrame, sample_ids: list[str], device: torch.device) -> tuple[float, pd.DataFrame]:
    model.eval()
    with torch.no_grad():
        risk = model.forward_all(he_bags, device).detach().cpu().numpy()
    time = labels.loc[sample_ids, "Time"].values.astype(float)
    event = labels.loc[sample_ids, "Event"].values.astype(int)
    ci = concordance_index(risk, time, event)
    return ci, pd.DataFrame({"sample_id": sample_ids, "log_risk": risk, "Event": event, "Time": time})


def run_fold(
    fold: int,
    args: argparse.Namespace,
    device: torch.device,
    fold_assignments: pd.DataFrame,
    labels: pd.DataFrame,
    he_bags_by_id: dict[str, torch.Tensor],
) -> dict:
    train_ids = fold_assignments.loc[fold_assignments["fold"] != fold, "sample_id"].tolist()
    test_ids = fold_assignments.loc[fold_assignments["fold"] == fold, "sample_id"].tolist()
    fit_ids, val_ids = make_inner_validation_split(train_ids, labels, args.seed + fold, args.val_size)

    set_seed(args.seed + fold)
    he_train = [he_bags_by_id[sid] for sid in train_ids]
    he_test = [he_bags_by_id[sid] for sid in test_ids]
    he_fit = [he_bags_by_id[sid] for sid in fit_ids]
    he_val = [he_bags_by_id[sid] for sid in val_ids]
    he_in_dim = int(he_train[0].shape[1])

    model = PathologyCox(
        he_in_dim=he_in_dim,
        he_emb_dim=args.he_emb_dim,
        he_aggregator=args.he_aggregator,
        he_attn_dim=args.he_attn_dim,
        he_dropout=args.he_dropout,
        head_hidden_dims=args.head_hidden_dims,
        head_dropout=args.head_dropout,
        head_activation=args.head_activation,
    ).to(device)
    fit_labels = pack_labels(labels, fit_ids, device)
    val_labels = pack_labels(labels, val_ids, device)
    train_labels = pack_labels(labels, train_ids, device)

    print(
        f"[FOLD {fold}] train_n={len(train_ids)} train_events={int(labels.loc[train_ids, 'Event'].sum())} "
        f"val_n={len(val_ids)} val_events={int(labels.loc[val_ids, 'Event'].sum())} "
        f"test_n={len(test_ids)} test_events={int(labels.loc[test_ids, 'Event'].sum())}"
    )
    model, history = train_model(model, he_fit, fit_labels, args, device, he_val=he_val, val_label_tensors=val_labels)

    fold_dir = args.out_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(fold_dir / "history.csv", index=False)
    train_ci, train_risk = evaluate_model(model, he_train, labels, train_ids, device)
    test_ci, test_risk = evaluate_model(model, he_test, labels, test_ids, device)
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
        "he_in_dim": he_in_dim,
    }
    with open(fold_dir / "summary.json", "w") as handle:
        json.dump(fold_summary, handle, indent=2)
    torch.save(
        {
            "model_state": model.state_dict(),
            "summary": fold_summary,
            "he_aggregator": args.he_aggregator,
            "he_emb_dim": args.he_emb_dim,
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

    sample_ids, labels, he_bags_by_id = load_master_dataset(args)
    fold_assignments = make_fold_assignments(sample_ids, labels, args)
    sample_ids = fold_assignments["sample_id"].astype(str).tolist()
    labels = labels.loc[sample_ids]
    he_bags_by_id = {sid: he_bags_by_id[sid] for sid in sample_ids}
    fold_assignments.to_csv(args.out_dir / "fold_assignments.csv", index=False)

    tile_counts = [int(he_bags_by_id[sid].shape[0]) for sid in sample_ids]
    he_in_dim = int(next(iter(he_bags_by_id.values())).shape[1])
    print(f"[DATA] labels: {args.labels_master}")
    print(f"[DATA] pathology index: {args.pathology_index}")
    print(f"[DATA] patients used: {len(sample_ids)}")
    print(f"[DATA] pathology dim: {he_in_dim}")
    print(f"[DATA] tile counts after cap: min={min(tile_counts)} median={float(np.median(tile_counts)):.1f} max={max(tile_counts)}")

    results = []
    for fold in sorted(fold_assignments["fold"].unique()):
        results.append(run_fold(int(fold), args, device, fold_assignments, labels, he_bags_by_id))

    results_df = pd.DataFrame(results)
    required_cols = ["fold", "train_n", "test_n", "train_events", "test_events", "c_index"]
    results_df[required_cols].to_csv(args.out_dir / "results_per_fold.csv", index=False)
    results_df.to_csv(args.out_dir / "results_per_fold_detailed.csv", index=False)

    c_indices = results_df["c_index"].astype(float).values
    summary = {
        "strategy": "pathology_unimodal_cox",
        "labels_master": str(args.labels_master),
        "pathology_index": str(args.pathology_index),
        "fold_assignments": str(args.fold_assignments) if args.fold_assignments else None,
        "n_splits": int(len(results_df)),
        "n_patients": len(sample_ids),
        "he_in_dim": he_in_dim,
        "he_aggregator": args.he_aggregator,
        "he_emb_dim": args.he_emb_dim,
        "he_attn_dim": args.he_attn_dim,
        "he_tile_cap": args.he_tile_cap,
        "tile_count_min": min(tile_counts),
        "tile_count_median": float(np.median(tile_counts)),
        "tile_count_max": max(tile_counts),
        "mean_c_index": float(np.mean(c_indices)),
        "std_c_index": float(np.std(c_indices, ddof=1)) if len(c_indices) > 1 else 0.0,
        "per_fold_results": results_df[required_cols].to_dict(orient="records"),
        "fold_c_indices": {str(int(row.fold)): float(row.c_index) for row in results_df.itertuples()},
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
