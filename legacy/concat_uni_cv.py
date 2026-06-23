#!/usr/bin/env python3
"""
Variance-filter RNA + clinical + pathology Cox model with concat fusion.

Clinical branch can use either:
    clinical.csv -> tabular MLP -> clinical embedding
    clinical .pt text embeddings -> token projection -> clinical embedding

Fusion:
    concat(RNA embedding, clinical embedding, pathology embedding) -> Cox head
"""

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
from torch import nn
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from rna_extractors.variance_filter import VarianceFilterExtractor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "5_CV"
DEFAULT_OUT = DEFAULT_INPUT_DIR / "outputs" / "concat_uni_cv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train variance_filter RNA + clinical + pathology Cox model with stratified 5-fold CV.")
    parser.add_argument("--labels-master", type=Path, default=DEFAULT_INPUT_DIR / "task3_combined_labels.csv")
    parser.add_argument("--clinical-source", choices=["tabular", "embedding"], default="tabular")
    parser.add_argument("--clinical-master", type=Path, default=DEFAULT_INPUT_DIR / "clinical.csv")
    parser.add_argument(
        "--clinical-embedding-dir",
        type=Path,
        default=None,
        help="Folder containing one clinical embedding .pt file per patient.",
    )
    parser.add_argument("--rna-master", type=Path, default=DEFAULT_INPUT_DIR / "all_samples_RNA_matrix.csv")
    parser.add_argument("--pathology-index", type=Path, default=DEFAULT_INPUT_DIR / "he_paths_all_splits_minimal.csv")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--val-size", type=float, default=0.2)

    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--use-cv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rna-hidden-dims", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--rna-out-dim", type=int, default=256)
    parser.add_argument("--rna-dropout", type=float, default=0.25)
    parser.add_argument("--rna-activation", choices=["relu", "gelu", "silu", "selu"], default="selu")
    parser.add_argument("--rna-input-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clinical-hidden-dims", type=int, nargs="*", default=[256, 128])
    parser.add_argument("--clinical-emb-dim", type=int, default=128)
    parser.add_argument("--clinical-token-hidden-dim", type=int, default=256)
    parser.add_argument("--clinical-token-out-dim", type=int, default=128)
    parser.add_argument("--clinical-dropout", type=float, default=0.20)
    parser.add_argument("--clinical-activation", choices=["relu", "selu"], default="selu")

    parser.add_argument("--he-aggregator", choices=["gated", "mean", "mean+std"], default="gated")
    parser.add_argument("--he-attn-dim", type=int, default=128)
    parser.add_argument("--he-emb-dim", type=int, default=256)
    parser.add_argument("--he-dropout", type=float, default=0.20)
    parser.add_argument("--he-tile-cap", type=int, default=4096)

    parser.add_argument("--fusion-hidden-dims", type=int, nargs="*", default=[256, 64])
    parser.add_argument("--fusion-dropout", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-events-per-batch", type=int, default=3)
    parser.add_argument(
        "--training-style",
        choices=["rna_clinical_batch", "event_batch", "baseline_stream"],
        default="rna_clinical_batch",
        help=(
            "rna_clinical_batch matches run_variance_filter_clinical.py; "
            "baseline_stream matches the UNI baseline's full-cohort streaming loop."
        ),
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_rna(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    rna_t = df.T
    rna_t.index.name = "sample_id"
    return rna_t.apply(pd.to_numeric, errors="coerce")


def to_event(value) -> int:
    if isinstance(value, str):
        return 1 if value.strip().lower() in {"1", "yes", "y", "true", "t", "progression", "progressed"} else 0
    return int(value)


def load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "sample" in df.columns and "sample_id" not in df.columns:
        df = df.rename(columns={"sample": "sample_id"})
    if "case_id" in df.columns and "sample_id" not in df.columns:
        df = df.rename(columns={"case_id": "sample_id"})
    if "time" in df.columns and "Time" not in df.columns:
        df = df.rename(columns={"time": "Time"})
    if "Time_to_prog_or_FUend" in df.columns:
        df = df.rename(columns={"Time_to_prog_or_FUend": "Time"})
    if "time_to_prog_or_FUend" in df.columns:
        df = df.rename(columns={"time_to_prog_or_FUend": "Time"})
    if "time_to_HG_recur_or_FUend" in df.columns:
        df = df.rename(columns={"time_to_HG_recur_or_FUend": "Time"})
    if "duration" in df.columns and "Time" not in df.columns:
        df = df.rename(columns={"duration": "Time"})
    if "survival_time" in df.columns and "Time" not in df.columns:
        df = df.rename(columns={"survival_time": "Time"})
    if "progression" in df.columns and "Event" not in df.columns:
        df["Event"] = df["progression"].apply(to_event)
    if "event" in df.columns and "Event" not in df.columns:
        df = df.rename(columns={"event": "Event"})
    if "sample_id" not in df.columns:
        first = df.columns[0]
        df = df.rename(columns={first: "sample_id"})
    missing = {"sample_id", "Time", "Event"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required label columns: {sorted(missing)}")
    df["sample_id"] = df["sample_id"].astype(str)
    return df.set_index("sample_id")[["Time", "Event"]].copy()


def load_clinical(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "case_id" in df.columns and "sample_id" not in df.columns:
        df = df.rename(columns={"case_id": "sample_id"})
    if "sample_id" not in df.columns:
        raise ValueError(f"{path} must contain sample_id or case_id.")
    df["sample_id"] = df["sample_id"].astype(str)
    return df


def load_clinical_embedding_file(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    emb = obj["embeddings"] if isinstance(obj, dict) and "embeddings" in obj else obj
    if isinstance(emb, np.ndarray):
        emb = torch.from_numpy(emb)
    if not torch.is_tensor(emb):
        raise TypeError(f"Unsupported clinical embedding object in {path}")
    emb = emb.float().contiguous()
    if emb.ndim == 1:
        emb = emb.unsqueeze(0)
    if emb.ndim != 2:
        raise ValueError(f"Expected [tokens, dim] or [dim] clinical embedding in {path}, got {tuple(emb.shape)}")
    return emb


def load_clinical_embeddings(embedding_dir: Path) -> dict[str, torch.Tensor]:
    files = sorted(embedding_dir.glob("*.pt"))
    if not files:
        raise ValueError(f"No .pt files found in clinical embedding dir: {embedding_dir}")
    embeddings = {}
    dims = set()
    token_counts = []
    for path in files:
        emb = load_clinical_embedding_file(path)
        embeddings[path.stem] = emb
        dims.add(int(emb.shape[1]))
        token_counts.append(int(emb.shape[0]))
    if len(dims) != 1:
        raise ValueError(f"Clinical embedding dimensions differ across files: {sorted(dims)}")
    print(
        "[DATA] clinical embeddings: "
        f"patients={len(embeddings)} dim={next(iter(dims))} "
        f"token_count_min={min(token_counts)} token_count_max={max(token_counts)}"
    )
    return embeddings


def stack_clinical_embeddings(
    embeddings: dict[str, torch.Tensor],
    sample_ids: list[str],
    max_tokens: int,
    embedding_dim: int,
) -> np.ndarray:
    out = torch.zeros((len(sample_ids), max_tokens, embedding_dim), dtype=torch.float32)
    for i, sid in enumerate(sample_ids):
        emb = embeddings[sid]
        out[i, : emb.shape[0], :] = emb
    return out.numpy().astype(np.float32)


def infer_numeric(col: pd.Series) -> bool:
    coerced = pd.to_numeric(col.replace({-1: np.nan, "-1": np.nan}), errors="coerce")
    return np.isfinite(coerced).mean() >= 0.8


def preprocess_clinical(train: pd.DataFrame, test: pd.DataFrame, val: pd.DataFrame | None = None):
    xtr = train.drop(columns=["sample_id"]).copy()
    xte = test.drop(columns=["sample_id"]).copy()
    xva = val.drop(columns=["sample_id"]).copy() if val is not None else None
    frames = [xtr, xte] + ([xva] if xva is not None else [])
    for df in frames:
        for col in df.columns:
            df[col] = df[col].replace({-1: np.nan, "-1": np.nan})

    numeric_cols, cat_cols = [], []
    for col in xtr.columns:
        (numeric_cols if infer_numeric(xtr[col]) else cat_cols).append(col)

    num_medians, num_means, num_stds = {}, {}, {}
    num_outputs = []
    for df, is_train in [(xtr, True), (xte, False)] + ([(xva, False)] if xva is not None else []):
        out = pd.DataFrame(index=df.index)
        for col in numeric_cols:
            values = pd.to_numeric(df[col], errors="coerce")
            if is_train:
                num_medians[col] = float(values.median())
            out[col] = values.fillna(num_medians[col])
        num_outputs.append(out)
    tr_num, te_num = num_outputs[:2]
    va_num = num_outputs[2] if xva is not None else None
    for col in numeric_cols:
        num_means[col] = float(tr_num[col].mean())
        std = float(tr_num[col].std())
        num_stds[col] = std if std > 1e-8 else 1.0
        tr_num[col] = (tr_num[col] - num_means[col]) / num_stds[col]
        te_num[col] = (te_num[col] - num_means[col]) / num_stds[col]
        if va_num is not None:
            va_num[col] = (va_num[col] - num_means[col]) / num_stds[col]

    if cat_cols:
        tr_cat = pd.get_dummies(xtr[cat_cols].fillna("Missing").astype("string"))
        te_cat = pd.get_dummies(xte[cat_cols].fillna("Missing").astype("string")).reindex(columns=tr_cat.columns, fill_value=0)
        va_cat = (
            pd.get_dummies(xva[cat_cols].fillna("Missing").astype("string")).reindex(columns=tr_cat.columns, fill_value=0)
            if xva is not None
            else None
        )
    else:
        tr_cat = pd.DataFrame(index=xtr.index)
        te_cat = pd.DataFrame(index=xte.index)
        va_cat = pd.DataFrame(index=xva.index) if xva is not None else None

    tr = pd.concat([tr_num, tr_cat], axis=1)
    te = pd.concat([te_num, te_cat], axis=1)
    va = pd.concat([va_num, va_cat], axis=1) if xva is not None else None
    keep = tr.columns[np.var(tr.values.astype(float), axis=0) > 0.0].tolist()
    tr = tr[keep]
    te = te.reindex(columns=keep, fill_value=0)
    if va is not None:
        va = va.reindex(columns=keep, fill_value=0)
    meta = {
        "numeric_cols": numeric_cols,
        "categorical_cols": cat_cols,
        "kept_features": keep,
        "numeric_medians": num_medians,
        "numeric_means": num_means,
        "numeric_stds": num_stds,
    }
    if va is not None:
        return tr.values.astype(np.float32), te.values.astype(np.float32), va.values.astype(np.float32), keep, meta
    return tr.values.astype(np.float32), te.values.astype(np.float32), keep, meta


def fit_clinical_preprocessor(fit: pd.DataFrame):
    xfit = fit.drop(columns=["sample_id"]).copy()
    for col in xfit.columns:
        xfit[col] = xfit[col].replace({-1: np.nan, "-1": np.nan})

    numeric_cols, cat_cols = [], []
    for col in xfit.columns:
        (numeric_cols if infer_numeric(xfit[col]) else cat_cols).append(col)

    num_medians, num_means, num_stds = {}, {}, {}
    fit_num = pd.DataFrame(index=xfit.index)
    for col in numeric_cols:
        values = pd.to_numeric(xfit[col], errors="coerce")
        num_medians[col] = float(values.median())
        fit_num[col] = values.fillna(num_medians[col])

    for col in numeric_cols:
        num_means[col] = float(fit_num[col].mean())
        std = float(fit_num[col].std())
        num_stds[col] = std if std > 1e-8 else 1.0
        fit_num[col] = (fit_num[col] - num_means[col]) / num_stds[col]

    if cat_cols:
        fit_cat = pd.get_dummies(xfit[cat_cols].fillna("Missing").astype("string"))
    else:
        fit_cat = pd.DataFrame(index=xfit.index)

    fit_processed = pd.concat([fit_num, fit_cat], axis=1)
    keep = fit_processed.columns[np.var(fit_processed.values.astype(float), axis=0) > 0.0].tolist()

    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": cat_cols,
        "kept_features": keep,
        "numeric_medians": num_medians,
        "numeric_means": num_means,
        "numeric_stds": num_stds,
        "categorical_dummy_cols": fit_cat.columns.tolist(),
    }


def transform_clinical(df: pd.DataFrame, meta: dict) -> np.ndarray:
    x = df.drop(columns=["sample_id"]).copy()
    for col in x.columns:
        x[col] = x[col].replace({-1: np.nan, "-1": np.nan})

    out_num = pd.DataFrame(index=x.index)
    for col in meta["numeric_cols"]:
        values = pd.to_numeric(x[col], errors="coerce")
        values = values.fillna(meta["numeric_medians"][col])
        out_num[col] = (values - meta["numeric_means"][col]) / meta["numeric_stds"][col]

    cat_cols = meta["categorical_cols"]
    if cat_cols:
        out_cat = pd.get_dummies(x[cat_cols].fillna("Missing").astype("string"))
        out_cat = out_cat.reindex(columns=meta["categorical_dummy_cols"], fill_value=0)
    else:
        out_cat = pd.DataFrame(index=x.index)

    processed = pd.concat([out_num, out_cat], axis=1)
    processed = processed.reindex(columns=meta["kept_features"], fill_value=0)
    return processed.values.astype(np.float32)


def cox_loss(risk: torch.Tensor, event: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(time, descending=True)
    r = risk[order]
    e = event[order]
    log_cumsum = torch.logcumsumexp(r, dim=0)
    observed = e == 1
    if observed.sum() == 0:
        return torch.zeros((), device=risk.device, requires_grad=True)
    return -((r[observed] - log_cumsum[observed]).sum() / observed.sum())


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
                extra = no_idx[: n_no_per_batch - len(no_chunk)]
                no_chunk = torch.cat([no_chunk, extra])
            no_ptr = (no_ptr + n_no_per_batch) % len(no_idx)
        else:
            no_chunk = torch.tensor([], dtype=torch.long, device=events.device)
        batch = torch.cat([ev_chunk, no_chunk])
        if len(batch) > 1:
            batches.append(batch[torch.randperm(len(batch), device=events.device)])
        ev_ptr += n_ev_per_batch
    return batches


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


def bootstrap_ci(risk: np.ndarray, time: np.ndarray, event: np.ndarray, seed: int, n_boot: int = 500):
    point = concordance_index(risk, time, event)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(risk), len(risk))
        if len(np.unique(event[idx])) < 2:
            continue
        vals.append(concordance_index(risk[idx], time[idx], event[idx]))
    if not vals:
        return point, math.nan, math.nan
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


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


class ClinicalEmbeddingEncoder(nn.Module):
    def __init__(
        self,
        token_count: int,
        in_dim: int,
        token_hidden_dim: int,
        token_out_dim: int,
        emb_dim: int,
        dropout: float,
        activation: str,
    ):
        super().__init__()
        drop_cls = nn.AlphaDropout if activation == "selu" else nn.Dropout
        act_cls = nn.SELU if activation == "selu" else nn.ReLU
        self.token_net = nn.Sequential(
            nn.Linear(in_dim, token_hidden_dim),
            nn.LayerNorm(token_hidden_dim),
            act_cls(),
            drop_cls(dropout),
            nn.Linear(token_hidden_dim, token_out_dim),
            nn.LayerNorm(token_out_dim),
            act_cls(),
            drop_cls(dropout),
        )
        self.fused_net = nn.Sequential(
            nn.Linear(token_count * token_out_dim, emb_dim),
            act_cls(),
            drop_cls(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"ClinicalEmbeddingEncoder expects [batch, tokens, dim], got {tuple(x.shape)}")
        mask = (x.abs().sum(dim=-1, keepdim=True) > 0).float()
        token_emb = self.token_net(x) * mask
        return self.fused_net(token_emb.flatten(start_dim=1))


class ClinicalEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], emb_dim: int, dropout: float, activation: str):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        drop_cls = nn.AlphaDropout if activation == "selu" else nn.Dropout
        act_cls = nn.SELU if activation == "selu" else nn.ReLU
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev, hidden_dim), nn.LayerNorm(hidden_dim), act_cls(), drop_cls(dropout)])
            prev = hidden_dim
        layers.extend([nn.Linear(prev, emb_dim), act_cls(), drop_cls(dropout)])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNACLinicalPathologyCox(nn.Module):
    def __init__(
        self,
        rna_extractor: VarianceFilterExtractor,
        clinical_source: str,
        clinical_token_count: int | None,
        clinical_dim: int,
        he_in_dim: int,
        he_emb_dim: int,
        he_aggregator: str,
        he_attn_dim: int,
        he_dropout: float,
        clinical_hidden_dims: list[int],
        clinical_token_hidden_dim: int,
        clinical_token_out_dim: int,
        clinical_emb_dim: int,
        clinical_dropout: float,
        clinical_activation: str,
        fusion_hidden_dims: list[int],
        fusion_dropout: float,
    ):
        super().__init__()
        self.rna_extractor = rna_extractor
        if clinical_source == "embedding":
            if clinical_token_count is None:
                raise ValueError("clinical_token_count is required for embedding clinical source.")
            self.clinical_encoder = ClinicalEmbeddingEncoder(
                token_count=clinical_token_count,
                in_dim=clinical_dim,
                token_hidden_dim=clinical_token_hidden_dim,
                token_out_dim=clinical_token_out_dim,
                emb_dim=clinical_emb_dim,
                dropout=clinical_dropout,
                activation=clinical_activation,
            )
        elif clinical_source == "tabular":
            self.clinical_encoder = ClinicalEncoder(
                clinical_dim,
                hidden_dims=clinical_hidden_dims,
                emb_dim=clinical_emb_dim,
                dropout=clinical_dropout,
                activation=clinical_activation,
            )
        else:
            raise ValueError(f"Unknown clinical_source: {clinical_source}")
        self.he_encoder = HEEncoder(he_in_dim, he_emb_dim, he_aggregator, he_attn_dim, he_dropout)
        fusion_in = rna_extractor.output_dim + clinical_emb_dim + he_emb_dim
        layers: list[nn.Module] = [nn.LayerNorm(fusion_in), nn.Dropout(fusion_dropout)]
        prev = fusion_in
        for hidden_dim in fusion_hidden_dims:
            layers.extend([nn.Linear(prev, hidden_dim), nn.LayerNorm(hidden_dim), nn.SELU(), nn.AlphaDropout(fusion_dropout)])
            prev = hidden_dim
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)

    def forward_sample(self, rna: torch.Tensor, clinical: torch.Tensor, he_bag: torch.Tensor):
        he_emb, weights = self.he_encoder(he_bag)
        z = torch.cat([self.rna_extractor(rna.unsqueeze(0)).squeeze(0), self.clinical_encoder(clinical), he_emb], dim=0)
        return self.head(z).squeeze(), weights

    def forward_all(self, rna: torch.Tensor, clinical: torch.Tensor, he_bags: list[torch.Tensor], device: torch.device):
        rna_emb = self.rna_extractor(rna)
        clin_emb = self.clinical_encoder(clinical)
        risks = []
        for i, bag in enumerate(he_bags):
            he_emb, _ = self.he_encoder(bag.to(device))
            fused = torch.cat([rna_emb[i], clin_emb[i], he_emb], dim=0)
            risks.append(self.head(fused).squeeze())
        return torch.stack(risks)


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def train_model(model, train_tensors, he_tr, args, device, val_tensors=None, he_val=None):
    rna_tr, clin_tr, time_tr, event_tr = train_tensors
    if val_tensors is not None:
        rna_va, clin_va, time_va, event_va = val_tensors
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = deepcopy(model.state_dict())
    best_loss = math.inf
    wait = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        if args.training_style == "baseline_stream":
            opt.zero_grad()
            risk = model.forward_all(rna_tr, clin_tr, he_tr, device)
            loss = cox_loss(risk, event_tr, time_tr)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        elif args.training_style == "event_batch":
            batches = event_aware_batch_indices(event_tr, args.batch_size, args.min_events_per_batch)
            for idx in batches:
                batch_ids = idx.detach().cpu().tolist()
                batch_bags = [he_tr[i] for i in batch_ids]
                opt.zero_grad()
                risk = model.forward_all(rna_tr[idx], clin_tr[idx], batch_bags, device)
                loss = cox_loss(risk, event_tr[idx], time_tr[idx])
                loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                losses.append(float(loss.detach().cpu()))
        else:
            perm = torch.randperm(len(rna_tr), device=device)
            for start in range(0, len(rna_tr), args.batch_size):
                idx = perm[start : start + args.batch_size]
                batch_ids = idx.detach().cpu().tolist()
                batch_bags = [he_tr[i] for i in batch_ids]
                opt.zero_grad()
                risk = model.forward_all(rna_tr[idx], clin_tr[idx], batch_bags, device)
                loss = cox_loss(risk, event_tr[idx], time_tr[idx])
                loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            train_risk_tensor = model.forward_all(rna_tr, clin_tr, he_tr, device)
            train_risk = train_risk_tensor.detach().cpu().numpy()
            train_full_loss = float(cox_loss(train_risk_tensor, event_tr, time_tr).detach().cpu())
            val_loss = math.nan
            val_ci = math.nan
            if val_tensors is not None and he_val is not None:
                val_risk_tensor = model.forward_all(rna_va, clin_va, he_val, device)
                val_loss = float(cox_loss(val_risk_tensor, event_va, time_va).detach().cpu())
                val_ci = concordance_index(
                    val_risk_tensor.detach().cpu().numpy(),
                    time_va.cpu().numpy(),
                    event_va.cpu().numpy(),
                )
        mean_loss = float(np.mean(losses)) if losses else math.nan
        train_ci = concordance_index(train_risk, time_tr.cpu().numpy(), event_tr.cpu().numpy())
        monitor_loss = val_loss if val_tensors is not None and he_val is not None else mean_loss
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


def load_master_dataset(
    args: argparse.Namespace,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame | dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    rna = load_rna(args.rna_master)
    labels = load_labels(args.labels_master)
    if args.clinical_source == "embedding":
        if args.clinical_embedding_dir is None:
            raise ValueError("--clinical-embedding-dir is required when --clinical-source embedding.")
        clinical_data = load_clinical_embeddings(args.clinical_embedding_dir)
        clinical_ids = set(clinical_data)
        clinical_label = "Clinical embeddings"
        clinical_common_label = "clinical embeddings"
    else:
        clinical_data = load_clinical(args.clinical_master)
        clinical_ids = set(clinical_data["sample_id"].astype(str))
        clinical_label = "Clinical tabular"
        clinical_common_label = "clinical"
    pathology = load_pathology_index(args.pathology_index)

    rna_ids = set(rna.index)
    label_ids = set(labels.index)
    pathology_ids = set(pathology.index)
    print("[DATA] modality patient counts:")
    print(f"  RNA: {len(rna_ids)}")
    print(f"  {clinical_label}: {len(clinical_ids)}")
    print(f"  Labels: {len(label_ids)}")
    print(f"  Pathology index: {len(pathology_ids)}")
    print(
        f"[DATA] common patients in RNA/genomics + {clinical_common_label} + labels + pathology index: "
        f"{len(rna_ids & label_ids & clinical_ids & pathology_ids)}"
    )

    common = sorted(rna_ids & label_ids & clinical_ids & pathology_ids)
    if len(common) < args.n_splits:
        raise ValueError(f"Too few patients with RNA, labels, {clinical_common_label}, and pathology index rows: {len(common)}")

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
    print(f"[DATA] retained {len(sample_ids)} patients with all modalities; dropped {len(missing_features)} missing/invalid pathology bags")
    if args.clinical_source == "embedding":
        clinical_data = {sid: clinical_data[sid] for sid in sample_ids}
    else:
        clinical_data = clinical_data[clinical_data["sample_id"].isin(sample_ids)].copy()
    return sample_ids, rna.loc[sample_ids], labels.loc[sample_ids], clinical_data, he_bags


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
        n_events = int(events.sum())
        n_censored = int((events == 0).sum())
        ratio = n_events / max(len(fold_ids), 1)
        print(f"  fold {fold}: n={len(fold_ids)} events={n_events} censored={n_censored} event_ratio={ratio:.3f}")
    return fold_assignments


def impute_rna_from_train(
    train: pd.DataFrame,
    test: pd.DataFrame,
    val: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, float]]:
    medians = train.median(axis=0, skipna=True).fillna(0.0)
    x_train = train.fillna(medians).values.astype(np.float32)
    x_test = test.fillna(medians).values.astype(np.float32)
    x_val = val.fillna(medians).values.astype(np.float32) if val is not None else None
    return x_train, x_test, x_val, {str(k): float(v) for k, v in medians.items()}


def impute_rna_with_medians(data: pd.DataFrame, medians: pd.Series) -> np.ndarray:
    return data.fillna(medians).values.astype(np.float32)


def build_rna_extractor(args: argparse.Namespace, gene_names: list[str], xrna_train: np.ndarray) -> VarianceFilterExtractor:
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
    extractor.fit(torch.tensor(xrna_train, dtype=torch.float32))
    return extractor


def build_model(
    args: argparse.Namespace,
    rna_extractor: VarianceFilterExtractor,
    clinical_token_count: int | None,
    clinical_dim: int,
    he_in_dim: int,
    device: torch.device,
):
    return RNACLinicalPathologyCox(
        rna_extractor=rna_extractor,
        clinical_source=args.clinical_source,
        clinical_token_count=clinical_token_count,
        clinical_dim=clinical_dim,
        he_in_dim=he_in_dim,
        he_emb_dim=args.he_emb_dim,
        he_aggregator=args.he_aggregator,
        he_attn_dim=args.he_attn_dim,
        he_dropout=args.he_dropout,
        clinical_hidden_dims=args.clinical_hidden_dims,
        clinical_token_hidden_dim=args.clinical_token_hidden_dim,
        clinical_token_out_dim=args.clinical_token_out_dim,
        clinical_emb_dim=args.clinical_emb_dim,
        clinical_dropout=args.clinical_dropout,
        clinical_activation=args.clinical_activation,
        fusion_hidden_dims=args.fusion_hidden_dims,
        fusion_dropout=args.fusion_dropout,
    ).to(device)


def evaluate_model(model, tensors, he_bags: list[torch.Tensor], labels: pd.DataFrame, sample_ids: list[str], device: torch.device) -> tuple[float, pd.DataFrame]:
    model.eval()
    with torch.no_grad():
        risk = model.forward_all(tensors[0], tensors[1], he_bags, device).detach().cpu().numpy()
    time = labels.loc[sample_ids, "Time"].values.astype(float)
    event = labels.loc[sample_ids, "Event"].values.astype(int)
    ci = concordance_index(risk, time, event)
    risk_table = pd.DataFrame({"sample_id": sample_ids, "log_risk": risk, "Event": event, "Time": time})
    return ci, risk_table


def make_inner_validation_split(train_ids: list[str], labels: pd.DataFrame, seed: int, val_size: float) -> tuple[list[str], list[str]]:
    y = labels.loc[train_ids, "Event"].astype(int).values
    class_counts = np.bincount(y, minlength=2)
    if class_counts.min() < 2:
        raise ValueError(
            "Cannot create a stratified early-stopping validation split: "
            f"event counts in outer training fold are {class_counts.tolist()}."
        )
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    fit_idx, val_idx = next(splitter.split(np.zeros(len(train_ids)), y))
    fit_ids = [train_ids[i] for i in fit_idx]
    val_ids = [train_ids[i] for i in val_idx]
    return fit_ids, val_ids


def run_fold(
    fold: int,
    args: argparse.Namespace,
    device: torch.device,
    fold_assignments: pd.DataFrame,
    rna: pd.DataFrame,
    labels: pd.DataFrame,
    clinical_data: pd.DataFrame | dict[str, torch.Tensor],
    he_bags_by_id: dict[str, torch.Tensor],
) -> dict:
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

    clinical_token_count = None
    clinical_token_counts = None
    clinical_metadata = {}
    if args.clinical_source == "embedding":
        clinical_embeddings = clinical_data
        clinical_token_count = max(int(emb.shape[0]) for emb in clinical_embeddings.values())
        clinical_dim = int(next(iter(clinical_embeddings.values())).shape[1])
        xclin_train = stack_clinical_embeddings(clinical_embeddings, train_ids, clinical_token_count, clinical_dim)
        xclin_fit = stack_clinical_embeddings(clinical_embeddings, fit_ids, clinical_token_count, clinical_dim)
        xclin_test = stack_clinical_embeddings(clinical_embeddings, test_ids, clinical_token_count, clinical_dim)
        xclin_val = stack_clinical_embeddings(clinical_embeddings, val_ids, clinical_token_count, clinical_dim)
        clinical_token_counts = {sid: int(clinical_embeddings[sid].shape[0]) for sid in train_ids + test_ids}
        clinical_metadata = {
            "clinical_embedding_dir": str(args.clinical_embedding_dir),
            "clinical_token_count_max": clinical_token_count,
            "clinical_embedding_dim": clinical_dim,
            "clinical_token_hidden_dim": args.clinical_token_hidden_dim,
            "clinical_token_out_dim": args.clinical_token_out_dim,
            "clinical_emb_dim": args.clinical_emb_dim,
            "clinical_token_counts": clinical_token_counts,
        }
    else:
        clinical = clinical_data
        clinical_index = clinical.set_index("sample_id")
        clin_train = clinical_index.loc[train_ids].reset_index()
        clin_fit = clinical_index.loc[fit_ids].reset_index()
        clin_test = clinical_index.loc[test_ids].reset_index()
        clin_val = clinical_index.loc[val_ids].reset_index()
        clin_meta = fit_clinical_preprocessor(clin_fit)
        xclin_train = transform_clinical(clin_train, clin_meta)
        xclin_fit = transform_clinical(clin_fit, clin_meta)
        xclin_test = transform_clinical(clin_test, clin_meta)
        xclin_val = transform_clinical(clin_val, clin_meta)
        clinical_dim = xclin_train.shape[1]
        clinical_metadata = {
            **clin_meta,
            "clinical_master": str(args.clinical_master),
            "clinical_feature_count": len(clin_meta["kept_features"]),
            "clinical_emb_dim": args.clinical_emb_dim,
        }

    set_seed(args.seed + fold)
    rna_extractor = build_rna_extractor(args, gene_names, xrna_fit)
    selected_genes = [str(gene) for gene in rna_extractor.get_selected_genes()]
    he_train = [he_bags_by_id[sid] for sid in train_ids]
    he_test = [he_bags_by_id[sid] for sid in test_ids]
    he_fit = [he_bags_by_id[sid] for sid in fit_ids]
    he_val = [he_bags_by_id[sid] for sid in val_ids]
    he_in_dim = int(he_train[0].shape[1])

    model = build_model(args, rna_extractor, clinical_token_count, clinical_dim, he_in_dim, device)
    train_tensors = (
        to_tensor(xrna_train, device),
        to_tensor(xclin_train, device),
        to_tensor(labels.loc[train_ids, "Time"].values.astype(np.float32), device),
        to_tensor(labels.loc[train_ids, "Event"].values.astype(np.float32), device),
    )
    fit_tensors = (
        to_tensor(xrna_fit, device),
        to_tensor(xclin_fit, device),
        to_tensor(labels.loc[fit_ids, "Time"].values.astype(np.float32), device),
        to_tensor(labels.loc[fit_ids, "Event"].values.astype(np.float32), device),
    )
    val_tensors = (
        to_tensor(xrna_val, device),
        to_tensor(xclin_val, device),
        to_tensor(labels.loc[val_ids, "Time"].values.astype(np.float32), device),
        to_tensor(labels.loc[val_ids, "Event"].values.astype(np.float32), device),
    )
    test_tensors = (
        to_tensor(xrna_test, device),
        to_tensor(xclin_test, device),
        to_tensor(labels.loc[test_ids, "Time"].values.astype(np.float32), device),
        to_tensor(labels.loc[test_ids, "Event"].values.astype(np.float32), device),
    )

    print(
        f"[FOLD {fold}] train_n={len(train_ids)} train_events={int(labels.loc[train_ids, 'Event'].sum())} "
        f"val_n={len(val_ids)} val_events={int(labels.loc[val_ids, 'Event'].sum())} "
        f"test_n={len(test_ids)} test_events={int(labels.loc[test_ids, 'Event'].sum())}"
    )
    model, history = train_model(model, fit_tensors, he_fit, args, device, val_tensors=val_tensors, he_val=he_val)
    fold_dir = args.out_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(fold_dir / "history.csv", index=False)

    train_ci, train_risk = evaluate_model(model, train_tensors, he_train, labels, train_ids, device)
    test_ci, test_risk = evaluate_model(model, test_tensors, he_test, labels, test_ids, device)
    train_risk.to_csv(fold_dir / "train_risk_scores.csv", index=False)
    test_risk.to_csv(fold_dir / "test_risk_scores.csv", index=False)
    pd.Series(selected_genes).to_csv(args.out_dir / f"selected_genes_fold_{fold}.txt", index=False, header=False)
    if args.clinical_source == "embedding":
        with open(fold_dir / "clinical_embedding_metadata.json", "w") as handle:
            json.dump(clinical_metadata, handle, indent=2)
    else:
        pd.Series(clinical_metadata["kept_features"]).to_csv(fold_dir / "clinical_features_used.txt", index=False, header=False)
        with open(fold_dir / "clinical_preprocessing.json", "w") as handle:
            json.dump(clinical_metadata, handle, indent=2)
    with open(fold_dir / "rna_imputation_medians.json", "w") as handle:
        json.dump(rna_medians, handle, indent=2)

    train_events = int(labels.loc[train_ids, "Event"].sum())
    test_events = int(labels.loc[test_ids, "Event"].sum())
    fold_summary = {
        "fold": fold,
        "c_index": test_ci,
        "train_c_index": train_ci,
        "train_n": len(train_ids),
        "test_n": len(test_ids),
        "train_events": train_events,
        "test_events": test_events,
        "early_stop_fit_n": len(fit_ids),
        "early_stop_val_n": len(val_ids),
        "early_stop_fit_events": int(labels.loc[fit_ids, "Event"].sum()),
        "early_stop_val_events": int(labels.loc[val_ids, "Event"].sum()),
        "selected_genes": selected_genes,
        "selected_gene_count": len(selected_genes),
        "clinical_source": args.clinical_source,
    }
    if args.clinical_source == "embedding":
        fold_summary.update(
            {
                "clinical_embedding_shape_padded": [clinical_token_count, clinical_dim],
                "clinical_token_count_min": min(clinical_token_counts.values()),
                "clinical_token_count_max": max(clinical_token_counts.values()),
            }
        )
    else:
        fold_summary["clinical_feature_count"] = clinical_metadata["clinical_feature_count"]
    with open(fold_dir / "summary.json", "w") as handle:
        json.dump(fold_summary, handle, indent=2)
    checkpoint = {
        "model_state": model.state_dict(),
        "summary": fold_summary,
        "selected_genes": selected_genes,
        "clinical_source": args.clinical_source,
    }
    if args.clinical_source == "embedding":
        checkpoint["clinical_embedding_metadata"] = {
            "clinical_embedding_dir": str(args.clinical_embedding_dir),
            "clinical_embedding_shape_padded": [clinical_token_count, clinical_dim],
            "clinical_token_hidden_dim": args.clinical_token_hidden_dim,
            "clinical_token_out_dim": args.clinical_token_out_dim,
            "clinical_emb_dim": args.clinical_emb_dim,
        }
    else:
        checkpoint["clinical_preprocessing"] = clinical_metadata
    torch.save(checkpoint, fold_dir / "model.pt")
    print(
        f"[FOLD {fold}] test_c_index={test_ci:.4f} train_n={len(train_ids)} "
        f"test_n={len(test_ids)} train_events={train_events} test_events={test_events}"
    )
    return fold_summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    sample_ids, rna, labels, clinical_data, he_bags_by_id = load_master_dataset(args)
    fold_assignments = make_fold_assignments(sample_ids, labels, args)
    fold_assignments.to_csv(args.out_dir / "fold_assignments.csv", index=False)

    results = []
    for fold in range(args.n_splits):
        results.append(run_fold(fold, args, device, fold_assignments, rna, labels, clinical_data, he_bags_by_id))

    results_df = pd.DataFrame(results)
    results_df.pop("selected_genes")
    results_df["selected_genes_file"] = [f"selected_genes_fold_{fold}.txt" for fold in results_df["fold"]]
    required_result_cols = ["fold", "train_n", "test_n", "train_events", "test_events", "c_index"]
    results_df[required_result_cols].to_csv(args.out_dir / "results_per_fold.csv", index=False)
    results_df.to_csv(args.out_dir / "results_per_fold_detailed.csv", index=False)

    c_indices = results_df["c_index"].astype(float).values
    per_fold_results = results_df[required_result_cols].to_dict(orient="records")
    summary = {
        "strategy": "variance_filter_clinical_pathology" if args.clinical_source == "tabular" else "variance_filter_clinical_embedding_pathology",
        "fusion": "concat",
        "clinical_source": args.clinical_source,
        "n_splits": args.n_splits,
        "stratification": "progression/Event",
        "shuffle": True,
        "random_state": 42,
        "n_patients": len(sample_ids),
        "input_genes": int(rna.shape[1]),
        "top_k": args.top_k,
        "use_cv": args.use_cv,
        "mean_c_index": float(np.mean(c_indices)),
        "std_c_index": float(np.std(c_indices, ddof=1)) if len(c_indices) > 1 else 0.0,
        "per_fold_results": per_fold_results,
        "fold_c_indices": {str(int(row.fold)): float(row.c_index) for row in results_df.itertuples()},
        "selected_gene_counts": {str(int(row.fold)): int(row.selected_gene_count) for row in results_df.itertuples()},
        "rna_hidden_dims": args.rna_hidden_dims,
        "rna_out_dim": args.rna_out_dim,
        "clinical_emb_dim": args.clinical_emb_dim,
        "he_aggregator": args.he_aggregator,
        "he_emb_dim": args.he_emb_dim,
        "fusion_hidden_dims": args.fusion_hidden_dims,
        "training_style": args.training_style,
        "seed": args.seed,
    }
    if args.clinical_source == "embedding":
        summary.update(
            {
                "clinical_embedding_dir": str(args.clinical_embedding_dir),
                "clinical_token_hidden_dim": args.clinical_token_hidden_dim,
                "clinical_token_out_dim": args.clinical_token_out_dim,
                "clinical_embedding_shapes_padded": {
                    str(int(row.fold)): list(row.clinical_embedding_shape_padded) for row in results_df.itertuples()
                },
                "clinical_token_count_min": int(results_df["clinical_token_count_min"].min()),
                "clinical_token_count_max": int(results_df["clinical_token_count_max"].max()),
            }
        )
    else:
        summary.update(
            {
                "clinical_master": str(args.clinical_master),
                "clinical_hidden_dims": args.clinical_hidden_dims,
                "clinical_feature_counts": {str(int(row.fold)): int(row.clinical_feature_count) for row in results_df.itertuples()},
            }
        )
    with open(args.out_dir / "cross_validation_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
