"""Trainer for SurvPGC-style token models."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from mm_survival.training.cross_validation import event_aware_batch_indices
from mm_survival.training.losses import cox_ph_loss
from mm_survival.training.metrics import concordance_index


SurvPGCTensors = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def _make_batches(
    n_samples: int,
    events: torch.Tensor,
    training_style: str,
    batch_size: int,
    min_events_per_batch: int,
    device: torch.device,
) -> list[torch.Tensor]:
    if training_style in {"full_batch", "baseline_stream"}:
        return [torch.arange(n_samples, device=device)]
    if training_style == "event_batch":
        return event_aware_batch_indices(events, batch_size, min_events_per_batch)
    if training_style in {"random_batch", "rna_clinical_batch"}:
        perm = torch.randperm(n_samples, device=device)
        return [perm[start : start + batch_size] for start in range(0, n_samples, batch_size)]
    raise ValueError(f"Unknown training_style: {training_style}")


def train_survpgc(
    model: nn.Module,
    train_tensors: SurvPGCTensors,
    *,
    device: torch.device,
    val_tensors: SurvPGCTensors | None = None,
    epochs: int = 300,
    patience: int = 40,
    batch_size: int = 64,
    min_events_per_batch: int = 3,
    training_style: str = "full_batch",
    lr: float = 2e-4,
    weight_decay: float = 1e-5,
    grad_clip: float = 5.0,
) -> tuple[nn.Module, pd.DataFrame]:
    """Train a SurvPGC-style token Cox model."""
    pathology_train, omics_train, clinical_train, mask_train, time_train, event_train = train_tensors
    if val_tensors is not None:
        pathology_val, omics_val, clinical_val, mask_val, time_val, event_val = val_tensors

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = deepcopy(model.state_dict())
    best_loss = math.inf
    wait = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        batches = _make_batches(
            n_samples=len(pathology_train),
            events=event_train,
            training_style=training_style,
            batch_size=batch_size,
            min_events_per_batch=min_events_per_batch,
            device=device,
        )
        for idx in batches:
            optimizer.zero_grad()
            risk = model.forward_all(
                pathology_train[idx],
                omics_train[idx],
                clinical_train[idx],
                mask_train[idx],
            )
            loss = cox_ph_loss(risk, event_train[idx], time_train[idx])
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            train_risk_tensor = model.forward_all(pathology_train, omics_train, clinical_train, mask_train)
            train_risk = train_risk_tensor.detach().cpu().numpy()
            train_full_loss = float(cox_ph_loss(train_risk_tensor, event_train, time_train).detach().cpu())
            val_loss = math.nan
            val_ci = math.nan
            if val_tensors is not None:
                val_risk_tensor = model.forward_all(pathology_val, omics_val, clinical_val, mask_val)
                val_loss = float(cox_ph_loss(val_risk_tensor, event_val, time_val).detach().cpu())
                val_ci = concordance_index(
                    val_risk_tensor.detach().cpu().numpy(),
                    time_val.detach().cpu().numpy(),
                    event_val.detach().cpu().numpy(),
                )

        mean_loss = float(np.mean(losses)) if losses else math.nan
        train_ci = concordance_index(
            train_risk,
            time_train.detach().cpu().numpy(),
            event_train.detach().cpu().numpy(),
        )
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
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def evaluate_survpgc(
    model: nn.Module,
    tensors: SurvPGCTensors,
    sample_ids: list[str],
) -> tuple[float, pd.DataFrame]:
    """Evaluate a SurvPGC-style token Cox model."""
    pathology, omics, clinical, mask, time, event = tensors
    model.eval()
    with torch.no_grad():
        risk = model.forward_all(pathology, omics, clinical, mask).detach().cpu().numpy()
    time_np = time.detach().cpu().numpy().astype(float)
    event_np = event.detach().cpu().numpy().astype(int)
    c_index = concordance_index(risk, time_np, event_np)
    risk_table = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "log_risk": risk,
            "Event": event_np,
            "Time": time_np,
        }
    )
    return c_index, risk_table
