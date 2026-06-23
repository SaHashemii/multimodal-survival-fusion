#!/usr/bin/env python3
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
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch import nn

from rna_token_plans import (
    build_category_plan,
    build_pathway_plan,
    impute_rna_with_medians,
    load_biological_categories,
    load_gmt_pathways,
    load_rna_matrix,
    log_token_plan,
    select_top_cv_genes,
)
from survpgc_model import SurvPGCUnifiedCox


MODEL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = MODEL_DIR.parent
DEFAULT_OUT = DEFAULT_INPUT_DIR / "outputs" / "survpgc_unified"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified SurvPGC-style Cox model for scFoundation embeddings, RNA category, or RNA pathway tokens."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--labels-master", type=Path, default=None)
    parser.add_argument("--pathology-index", type=Path, default=None)
    parser.add_argument("--clinical-token-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)

    parser.add_argument("--omics-source", choices=["scfoundation", "category", "pathway"], required=True)
    parser.add_argument("--omics-token-dir", type=Path, default=None, help="Required for --omics-source scfoundation.")
    parser.add_argument("--rna-master", type=Path, default=None, help="Required for category/pathway unless input-dir contains all_samples_RNA_matrix.csv.")
    parser.add_argument("--biological-categories", type=Path, default=None, help="Required for --omics-source category.")
    parser.add_argument("--pathway-gmt", type=Path, nargs="*", default=None, help="Required for --omics-source pathway.")
    parser.add_argument("--rna-mode", choices=["all_genes", "top_cv"], default="all_genes")
    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--min-genes-per-token", type=int, default=10)
    parser.add_argument("--min-pathway-coverage", type=float, default=0.90)
    parser.add_argument("--rna-token-dim", type=int, default=256)
    parser.add_argument("--rna-hidden-dim", type=int, default=256)
    parser.add_argument("--rna-dropout", type=float, default=0.25)

    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--pathology-token-cap", type=int, default=4096)
    parser.add_argument("--pathology-in-dim", type=int, default=None, help="Optional fixed UNI/pathology feature dimension check.")
    parser.add_argument("--clinical-token-count", type=int, default=None, help="Optional fixed clinical token count check.")
    parser.add_argument("--clinical-in-dim", type=int, default=None, help="Optional fixed clinical embedding dimension check.")
    parser.add_argument("--omics-token-count", type=int, default=None, help="Optional fixed scFoundation token count check.")
    parser.add_argument("--omics-in-dim", type=int, default=None, help="Optional fixed scFoundation embedding dimension check.")
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--attention-dim-head", type=int, default=128)
    parser.add_argument("--fusion-dropout", type=float, default=0.30)
    parser.add_argument("--fusion-hidden-dims", type=int, nargs="*", default=[256, 64])

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-events-per-batch", type=int, default=3)
    parser.add_argument(
        "--training-style",
        choices=["rna_clinical_batch", "event_batch", "baseline_stream"],
        default="rna_clinical_batch",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.labels_master = args.labels_master or args.input_dir / "task3_combined_labels.csv"
    args.pathology_index = args.pathology_index or args.input_dir / "he_paths_all_splits_minimal.csv"
    args.rna_master = args.rna_master or args.input_dir / "all_samples_RNA_matrix.csv"
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.omics_source == "scfoundation" and args.omics_token_dir is None:
        raise ValueError("--omics-token-dir is required when --omics-source scfoundation.")
    if args.omics_source == "category" and args.biological_categories is None:
        raise ValueError("--biological-categories is required when --omics-source category.")
    if args.omics_source == "pathway" and not args.pathway_gmt:
        raise ValueError("--pathway-gmt is required when --omics-source pathway.")


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
    out = df.set_index("sample_id")[["Time", "Event"]].copy()
    out["Time"] = pd.to_numeric(out["Time"], errors="coerce")
    out["Event"] = out["Event"].apply(to_event)
    return out.dropna(subset=["Time", "Event"])


def load_pathology_index(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "source_path_sol" in df.columns and "feature_path" not in df.columns:
        df = df.rename(columns={"source_path_sol": "feature_path"})
    missing = {"sample_id", "feature_path"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required pathology index columns: {sorted(missing)}")
    df["sample_id"] = df["sample_id"].astype(str)
    original = df["sample_id"].copy()
    df["sample_id"] = df["sample_id"].str.replace(r"^2U_", "3U_", regex=True)
    df["sample_id"] = df["sample_id"].str.replace(r"^2B_", "3B_", regex=True)
    remapped = int((original != df["sample_id"]).sum())
    if remapped:
        print(f"[DATA] remapped {remapped} pathology sample IDs from 2U_/2B_ to 3U_/3B_")
    df["feature_path"] = df["feature_path"].astype(str)
    return df.drop_duplicates("sample_id").set_index("sample_id")


def load_feature_tensor(path: Path) -> torch.Tensor | None:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    except OSError as exc:
        print(f"[Feature] WARN: cannot load {path}: {exc}")
        return None
    feats = obj.get("feats") if isinstance(obj, dict) and "feats" in obj else obj
    feats = obj.get("embeddings") if isinstance(obj, dict) and "embeddings" in obj else feats
    if isinstance(feats, np.ndarray):
        feats = torch.from_numpy(feats)
    if not torch.is_tensor(feats):
        print(f"[Feature] WARN: unsupported object in {path}")
        return None
    feats = feats.float()
    if feats.ndim < 1 or feats.numel() == 0:
        print(f"[Feature] WARN: empty tensor in {path}: {tuple(feats.shape)}")
        return None
    if feats.ndim == 1:
        feats = feats.unsqueeze(0)
    return feats.contiguous()


def sample_and_pad_pathology(feats: torch.Tensor, sample_id: str, seed: int, token_cap: int) -> tuple[torch.Tensor, torch.Tensor]:
    if feats.ndim != 2 or feats.shape[0] == 0:
        raise ValueError(f"Pathology features for {sample_id} must be [tiles, dim], got {tuple(feats.shape)}")
    n_tiles = feats.shape[0]
    n_keep = min(n_tiles, token_cap)
    if n_tiles > token_cap:
        digest = hashlib.sha256(f"{sample_id}:{seed}".encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little") % (2**32))
        idx = rng.choice(n_tiles, size=token_cap, replace=False)
        idx.sort()
        feats = feats[idx]
    elif n_tiles < token_cap:
        pad = torch.zeros((token_cap - n_tiles, feats.shape[1]), dtype=feats.dtype)
        feats = torch.cat([feats, pad], dim=0)
    mask = torch.zeros(token_cap, dtype=torch.bool)
    mask[:n_keep] = True
    return feats.contiguous(), mask


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


def event_aware_batch_indices(events: torch.Tensor, batch_size: int, min_events: int = 3) -> list[torch.Tensor]:
    n = len(events)
    ev_idx = torch.where(events == 1)[0]
    no_idx = torch.where(events == 0)[0]
    ev_idx = ev_idx[torch.randperm(len(ev_idx), device=events.device)]
    no_idx = no_idx[torch.randperm(len(no_idx), device=events.device)]
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


def load_master_dataset(args: argparse.Namespace):
    labels = load_labels(args.labels_master)
    pathology = load_pathology_index(args.pathology_index)
    rna = load_rna_matrix(args.rna_master) if args.omics_source in {"category", "pathway"} else None

    label_ids = set(labels.index)
    pathology_ids = set(pathology.index)
    clinical_ids = {sid for sid in label_ids | pathology_ids if (args.clinical_token_dir / f"{sid}.pt").is_file()}
    if rna is not None:
        base_ids = label_ids & pathology_ids & clinical_ids & set(rna.index)
    else:
        omics_ids = {sid for sid in label_ids | pathology_ids if (args.omics_token_dir / f"{sid}.pt").is_file()}
        base_ids = label_ids & pathology_ids & clinical_ids & omics_ids
    common = sorted(base_ids)
    if len(common) < args.n_splits:
        raise ValueError(f"Too few patients with required modalities: {len(common)}")

    pathology_tokens: dict[str, torch.Tensor] = {}
    pathology_masks: dict[str, torch.Tensor] = {}
    clinical_tokens: dict[str, torch.Tensor] = {}
    scfoundation_tokens: dict[str, torch.Tensor] = {}
    pathology_shapes = set()
    clinical_shapes = set()
    omics_shapes = set()
    dropped = []

    for sid in common:
        feature_path = Path(pathology.loc[sid, "feature_path"])
        if not feature_path.is_absolute():
            feature_path = args.pathology_index.parent / feature_path
        path_feats = load_feature_tensor(feature_path)
        clinic = load_feature_tensor(args.clinical_token_dir / f"{sid}.pt")
        omics = load_feature_tensor(args.omics_token_dir / f"{sid}.pt") if args.omics_source == "scfoundation" else None
        if path_feats is None or clinic is None or (args.omics_source == "scfoundation" and omics is None):
            dropped.append(sid)
            continue
        if path_feats.ndim != 2 or clinic.ndim != 2 or (omics is not None and omics.ndim != 2):
            print(f"[DATA] WARN: dropping {sid}; expected all loaded modalities to be 2D tensors")
            dropped.append(sid)
            continue
        if args.pathology_in_dim is not None and path_feats.shape[1] != args.pathology_in_dim:
            print(f"[DATA] WARN: dropping {sid}; pathology dim {path_feats.shape[1]} != {args.pathology_in_dim}")
            dropped.append(sid)
            continue
        if args.clinical_token_count is not None and clinic.shape[0] != args.clinical_token_count:
            print(f"[DATA] WARN: dropping {sid}; clinical token count {clinic.shape[0]} != {args.clinical_token_count}")
            dropped.append(sid)
            continue
        if args.clinical_in_dim is not None and clinic.shape[1] != args.clinical_in_dim:
            print(f"[DATA] WARN: dropping {sid}; clinical dim {clinic.shape[1]} != {args.clinical_in_dim}")
            dropped.append(sid)
            continue
        if omics is not None:
            if args.omics_token_count is not None and omics.shape[0] != args.omics_token_count:
                print(f"[DATA] WARN: dropping {sid}; omics token count {omics.shape[0]} != {args.omics_token_count}")
                dropped.append(sid)
                continue
            if args.omics_in_dim is not None and omics.shape[1] != args.omics_in_dim:
                print(f"[DATA] WARN: dropping {sid}; omics dim {omics.shape[1]} != {args.omics_in_dim}")
                dropped.append(sid)
                continue

        path_tok, path_mask = sample_and_pad_pathology(path_feats, sid, args.seed, args.pathology_token_cap)
        pathology_tokens[sid] = path_tok
        pathology_masks[sid] = path_mask
        clinical_tokens[sid] = clinic
        pathology_shapes.add((path_tok.shape[0], path_tok.shape[1]))
        clinical_shapes.add(tuple(clinic.shape))
        if omics is not None:
            scfoundation_tokens[sid] = omics
            omics_shapes.add(tuple(omics.shape))

    sample_ids = sorted(set(common) - set(dropped))
    if len(sample_ids) < args.n_splits:
        raise ValueError(f"Too few loadable patients after validation: {len(sample_ids)}")
    if len(pathology_shapes) != 1:
        raise ValueError(f"Pathology tensor shapes differ after padding: {sorted(pathology_shapes)}")
    if len(clinical_shapes) != 1:
        raise ValueError(f"Clinical token shapes differ across retained patients: {sorted(clinical_shapes)}")
    if args.omics_source == "scfoundation" and len(omics_shapes) != 1:
        raise ValueError(f"scFoundation token shapes differ across retained patients: {sorted(omics_shapes)}")

    _, pathology_in_dim = next(iter(pathology_shapes))
    clinical_token_count, clinical_in_dim = next(iter(clinical_shapes))
    args.pathology_in_dim = int(pathology_in_dim)
    args.clinical_token_count = int(clinical_token_count)
    args.clinical_in_dim = int(clinical_in_dim)
    if args.omics_source == "scfoundation":
        omics_token_count, omics_in_dim = next(iter(omics_shapes))
        args.omics_token_count = int(omics_token_count)
        args.omics_in_dim = int(omics_in_dim)

    print("[DATA] retained patients:", len(sample_ids))
    print(f"[DATA] pathology token shape: [{args.pathology_token_cap}, {args.pathology_in_dim}]")
    print(f"[DATA] clinical token shape: [{args.clinical_token_count}, {args.clinical_in_dim}]")
    if args.omics_source == "scfoundation":
        print(f"[DATA] scFoundation token shape: [{args.omics_token_count}, {args.omics_in_dim}]")
    else:
        print(f"[DATA] RNA matrix shape: samples={len(rna.index)} genes={rna.shape[1]}")

    rna = rna.loc[sample_ids] if rna is not None else None
    return sample_ids, rna, labels.loc[sample_ids], pathology_tokens, pathology_masks, clinical_tokens, scfoundation_tokens


def make_fold_assignments(sample_ids: list[str], labels: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    y = labels.loc[sample_ids, "Event"].astype(int).values
    splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    fold_assignments = pd.DataFrame({"sample_id": sample_ids, "fold": -1})
    for fold, (_, test_idx) in enumerate(splitter.split(np.zeros(len(sample_ids)), y)):
        fold_assignments.loc[test_idx, "fold"] = fold
    print("[CV] Event distribution per held-out fold:")
    for fold in range(args.n_splits):
        fold_ids = fold_assignments.loc[fold_assignments["fold"] == fold, "sample_id"].tolist()
        events = labels.loc[fold_ids, "Event"].astype(int)
        print(f"  fold {fold}: n={len(fold_ids)} events={int(events.sum())} censored={int((events == 0).sum())}")
    return fold_assignments


def make_inner_validation_split(train_ids: list[str], labels: pd.DataFrame, seed: int, val_size: float) -> tuple[list[str], list[str]]:
    y = labels.loc[train_ids, "Event"].astype(int).values
    class_counts = np.bincount(y, minlength=2)
    if class_counts.min() < 2:
        raise ValueError(f"Cannot create stratified validation split; event counts are {class_counts.tolist()}.")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    fit_idx, val_idx = next(splitter.split(np.zeros(len(train_ids)), y))
    return [train_ids[i] for i in fit_idx], [train_ids[i] for i in val_idx]


def stack_by_id(data: dict[str, torch.Tensor], sample_ids: list[str], device: torch.device) -> torch.Tensor:
    return torch.stack([data[sid] for sid in sample_ids], dim=0).to(device)


def to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(array, dtype=torch.float32, device=device)


def tensor_pack(sample_ids, labels, omics_data, pathology_tokens, pathology_masks, clinical_tokens, device, omics_source: str):
    if omics_source == "scfoundation":
        omics_tensor = stack_by_id(omics_data, sample_ids, device)
    else:
        omics_tensor = to_tensor(omics_data.loc[sample_ids].values.astype(np.float32), device)
    return (
        stack_by_id(pathology_tokens, sample_ids, device),
        omics_tensor,
        stack_by_id(clinical_tokens, sample_ids, device),
        stack_by_id(pathology_masks, sample_ids, device),
        to_tensor(labels.loc[sample_ids, "Time"].values.astype(np.float32), device),
        to_tensor(labels.loc[sample_ids, "Event"].values.astype(np.float32), device),
    )


def build_model(args: argparse.Namespace, rna_gene_indices: list[list[int]] | None, device: torch.device) -> SurvPGCUnifiedCox:
    omics_in_dim = args.rna_token_dim if rna_gene_indices is not None else args.omics_in_dim
    omics_token_count = len(rna_gene_indices) if rna_gene_indices is not None else args.omics_token_count
    return SurvPGCUnifiedCox(
        pathology_in_dim=args.pathology_in_dim,
        clinical_in_dim=args.clinical_in_dim,
        clinical_token_count=args.clinical_token_count,
        omics_in_dim=omics_in_dim,
        omics_token_count=omics_token_count,
        projection_dim=args.projection_dim,
        attention_dim_head=args.attention_dim_head,
        fusion_hidden_dims=args.fusion_hidden_dims,
        fusion_dropout=args.fusion_dropout,
        rna_gene_indices=rna_gene_indices,
        rna_hidden_dim=args.rna_hidden_dim,
        rna_token_dim=args.rna_token_dim,
        rna_dropout=args.rna_dropout,
    ).to(device)


def train_model(model, train_tensors, args, device, val_tensors=None):
    path_tr, omics_tr, clinic_tr, mask_tr, time_tr, event_tr = train_tensors
    if val_tensors is not None:
        path_va, omics_va, clinic_va, mask_va, time_va, event_va = val_tensors
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = deepcopy(model.state_dict())
    best_loss = math.inf
    wait = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        if args.training_style == "baseline_stream":
            batches = [torch.arange(len(path_tr), device=device)]
        elif args.training_style == "event_batch":
            batches = event_aware_batch_indices(event_tr, args.batch_size, args.min_events_per_batch)
        else:
            perm = torch.randperm(len(path_tr), device=device)
            batches = [perm[start : start + args.batch_size] for start in range(0, len(path_tr), args.batch_size)]
        for idx in batches:
            opt.zero_grad()
            risk = model.forward_all(path_tr[idx], omics_tr[idx], clinic_tr[idx], mask_tr[idx])
            loss = cox_loss(risk, event_tr[idx], time_tr[idx])
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            train_risk_tensor = model.forward_all(path_tr, omics_tr, clinic_tr, mask_tr)
            train_full_loss = float(cox_loss(train_risk_tensor, event_tr, time_tr).detach().cpu())
            val_loss = math.nan
            val_ci = math.nan
            if val_tensors is not None:
                val_risk_tensor = model.forward_all(path_va, omics_va, clinic_va, mask_va)
                val_loss = float(cox_loss(val_risk_tensor, event_va, time_va).detach().cpu())
                val_ci = concordance_index(val_risk_tensor.detach().cpu().numpy(), time_va.cpu().numpy(), event_va.cpu().numpy())
        mean_loss = float(np.mean(losses)) if losses else math.nan
        train_ci = concordance_index(train_risk_tensor.detach().cpu().numpy(), time_tr.cpu().numpy(), event_tr.cpu().numpy())
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
    model.eval()
    with torch.no_grad():
        risk = model.forward_all(tensors[0], tensors[1], tensors[2], tensors[3]).detach().cpu().numpy()
    time = labels.loc[sample_ids, "Time"].values.astype(float)
    event = labels.loc[sample_ids, "Event"].values.astype(int)
    ci = concordance_index(risk, time, event)
    return ci, pd.DataFrame({"sample_id": sample_ids, "log_risk": risk, "Event": event, "Time": time})


def prepare_fold_omics(fold: int, args: argparse.Namespace, rna: pd.DataFrame | None, fit_ids, train_ids, val_ids, test_ids):
    if args.omics_source == "scfoundation":
        return None, None, None, None, None, None

    gene_names = rna.columns.astype(str).tolist()
    fit_raw = rna.loc[fit_ids, gene_names]
    medians = fit_raw.median(axis=0, skipna=True).fillna(0.0)
    fit_rna = impute_rna_with_medians(fit_raw, medians)
    train_rna = impute_rna_with_medians(rna.loc[train_ids, gene_names], medians)
    val_rna = impute_rna_with_medians(rna.loc[val_ids, gene_names], medians)
    test_rna = impute_rna_with_medians(rna.loc[test_ids, gene_names], medians)
    selected_genes = select_top_cv_genes(fit_rna, top_k=args.top_k) if args.rna_mode == "top_cv" else None

    if args.omics_source == "category":
        definitions = load_biological_categories(args.biological_categories)
        plan = build_category_plan(definitions, gene_names, selected_genes, args.min_genes_per_token, fold)
    else:
        definitions = load_gmt_pathways(args.pathway_gmt)
        plan = build_pathway_plan(
            definitions,
            gene_names,
            selected_genes,
            args.min_genes_per_token,
            args.min_pathway_coverage,
            fold,
        )
    if not plan.names:
        raise ValueError(f"Fold {fold} has no retained RNA tokens.")
    log_token_plan(plan, fold)
    return train_rna, fit_rna, val_rna, test_rna, selected_genes, plan


def write_fold_token_artifacts(fold_dir: Path, args: argparse.Namespace, selected_genes, plan, medians_source=None) -> None:
    if plan is None:
        return
    plan.stats.to_csv(fold_dir / "rna_token_stats.csv", index=False)
    pd.Series(plan.names).to_csv(fold_dir / "rna_token_names.txt", index=False, header=False)
    for name, genes in zip(plan.names, plan.gene_names):
        safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)
        pd.Series(genes).to_csv(fold_dir / f"genes_{safe_name}.txt", index=False, header=False)
    if selected_genes is not None:
        pd.Series(selected_genes).to_csv(fold_dir / "selected_top_cv_genes.txt", index=False, header=False)


def run_fold(
    fold: int,
    args: argparse.Namespace,
    device: torch.device,
    fold_assignments: pd.DataFrame,
    rna: pd.DataFrame | None,
    labels: pd.DataFrame,
    pathology_tokens,
    pathology_masks,
    clinical_tokens,
    scfoundation_tokens,
) -> dict:
    train_ids = fold_assignments.loc[fold_assignments["fold"] != fold, "sample_id"].tolist()
    test_ids = fold_assignments.loc[fold_assignments["fold"] == fold, "sample_id"].tolist()
    fit_ids, val_ids = make_inner_validation_split(train_ids, labels, args.seed + fold, args.val_size)

    train_omics = fit_omics = val_omics = test_omics = scfoundation_tokens
    selected_genes = None
    plan = None
    rna_medians = None
    if args.omics_source != "scfoundation":
        gene_names = rna.columns.astype(str).tolist()
        medians_series = rna.loc[fit_ids, gene_names].median(axis=0, skipna=True).fillna(0.0)
        rna_medians = {str(k): float(v) for k, v in medians_series.items()}
        train_omics, fit_omics, val_omics, test_omics, selected_genes, plan = prepare_fold_omics(
            fold, args, rna, fit_ids, train_ids, val_ids, test_ids
        )

    fold_dir = args.out_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    write_fold_token_artifacts(fold_dir, args, selected_genes, plan)
    if rna_medians is not None:
        with open(fold_dir / "rna_imputation_medians.json", "w") as handle:
            json.dump(rna_medians, handle, indent=2)

    set_seed(args.seed + fold)
    model = build_model(args, plan.gene_indices if plan is not None else None, device)
    train_tensors = tensor_pack(train_ids, labels, train_omics, pathology_tokens, pathology_masks, clinical_tokens, device, args.omics_source)
    fit_tensors = tensor_pack(fit_ids, labels, fit_omics, pathology_tokens, pathology_masks, clinical_tokens, device, args.omics_source)
    val_tensors = tensor_pack(val_ids, labels, val_omics, pathology_tokens, pathology_masks, clinical_tokens, device, args.omics_source)
    test_tensors = tensor_pack(test_ids, labels, test_omics, pathology_tokens, pathology_masks, clinical_tokens, device, args.omics_source)

    print(
        f"[FOLD {fold}] train_n={len(train_ids)} train_events={int(labels.loc[train_ids, 'Event'].sum())} "
        f"val_n={len(val_ids)} val_events={int(labels.loc[val_ids, 'Event'].sum())} "
        f"test_n={len(test_ids)} test_events={int(labels.loc[test_ids, 'Event'].sum())}"
    )
    model, history = train_model(model, fit_tensors, args, device, val_tensors=val_tensors)
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
        "omics_source": args.omics_source,
        "rna_mode": args.rna_mode if args.omics_source != "scfoundation" else None,
        "omics_tokens": args.omics_token_count if plan is None else len(plan.names),
        "token_names": None if plan is None else plan.names,
        "genes_per_token": None if plan is None else [len(g) for g in plan.gene_names],
    }
    with open(fold_dir / "summary.json", "w") as handle:
        json.dump(fold_summary, handle, indent=2)
    save_payload = {"model_state": model.state_dict(), "summary": fold_summary}
    if plan is not None:
        save_payload.update(
            {
                "rna_token_stats": plan.stats.to_dict(orient="records"),
                "rna_token_names": plan.names,
                "rna_token_gene_names": plan.gene_names,
            }
        )
    torch.save(save_payload, fold_dir / "model.pt")
    print(f"[FOLD {fold}] test_c_index={test_ci:.4f} train_n={len(train_ids)} test_n={len(test_ids)}")
    return fold_summary


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    sample_ids, rna, labels, pathology_tokens, pathology_masks, clinical_tokens, scfoundation_tokens = load_master_dataset(args)
    fold_assignments = make_fold_assignments(sample_ids, labels, args)
    fold_assignments.to_csv(args.out_dir / "fold_assignments.csv", index=False)

    results = []
    for fold in range(args.n_splits):
        results.append(
            run_fold(
                fold,
                args,
                device,
                fold_assignments,
                rna,
                labels,
                pathology_tokens,
                pathology_masks,
                clinical_tokens,
                scfoundation_tokens,
            )
        )

    results_df = pd.DataFrame(results)
    required_result_cols = ["fold", "train_n", "test_n", "train_events", "test_events", "c_index"]
    results_df[required_result_cols].to_csv(args.out_dir / "results_per_fold.csv", index=False)
    results_df.to_csv(args.out_dir / "results_per_fold_detailed.csv", index=False)

    token_stats_paths = sorted(args.out_dir.glob("fold_*/rna_token_stats.csv"))
    if token_stats_paths:
        pd.concat([pd.read_csv(path) for path in token_stats_paths], ignore_index=True).to_csv(
            args.out_dir / "rna_token_stats_all_folds.csv",
            index=False,
        )

    c_indices = results_df["c_index"].astype(float).values
    summary = {
        "strategy": "survpgc_unified_attention_fusion",
        "omics_source": args.omics_source,
        "rna_mode": args.rna_mode if args.omics_source != "scfoundation" else None,
        "top_k": args.top_k if args.omics_source != "scfoundation" and args.rna_mode == "top_cv" else None,
        "n_splits": args.n_splits,
        "n_patients": len(sample_ids),
        "pathology_token_shape": [args.pathology_token_cap, args.pathology_in_dim],
        "clinical_token_shape": [args.clinical_token_count, args.clinical_in_dim],
        "omics_token_shape": [args.omics_token_count, args.omics_in_dim] if args.omics_source == "scfoundation" else None,
        "projection_dim": args.projection_dim,
        "attention_dim_head": args.attention_dim_head,
        "fused_dim": args.attention_dim_head * 4,
        "pathology_mask_used": False,
        "mean_c_index": float(np.mean(c_indices)),
        "std_c_index": float(np.std(c_indices, ddof=1)) if len(c_indices) > 1 else 0.0,
        "per_fold_results": results_df[required_result_cols].to_dict(orient="records"),
        "fold_c_indices": {str(int(row.fold)): float(row.c_index) for row in results_df.itertuples()},
        "fusion_hidden_dims": args.fusion_hidden_dims,
        "training_style": args.training_style,
        "seed": args.seed,
        "labels_master": str(args.labels_master),
        "pathology_index": str(args.pathology_index),
        "clinical_token_dir": str(args.clinical_token_dir),
        "omics_token_dir": str(args.omics_token_dir) if args.omics_token_dir else None,
        "rna_master": str(args.rna_master) if args.omics_source != "scfoundation" else None,
        "biological_categories": str(args.biological_categories) if args.biological_categories else None,
        "pathway_gmt": [str(path) for path in args.pathway_gmt] if args.pathway_gmt else None,
    }
    with open(args.out_dir / "cross_validation_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
