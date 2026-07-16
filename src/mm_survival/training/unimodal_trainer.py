"""
Trainers for unimodal Cox models
================================

Shared training and evaluation loops for RNA-only, clinical-only, and
pathology-only Cox baselines.

Pipeline
--------
  build batches → forward model → Cox partial likelihood loss
  validation loss/C-index → early stopping → restore best checkpoint

Design rationale
----------------
* Tensor-based unimodal models cover RNA and clinical inputs.
* Pathology models use a separate loop because each patient can have a
  variable-length tile bag instead of one fixed-size tensor row.
* Unimodal risk-score tables are saved in the same format as multimodal outputs,
  enabling direct comparison and late-fusion experiments.
"""

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


TensorTensors = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _make_batches(
    n_samples: int,
    events: torch.Tensor,
    training_style: str,
    batch_size: int,
    min_events_per_batch: int,
    device: torch.device,
) -> list[torch.Tensor]:

    # Match the batching options used by multimodal trainers so unimodal and
    # multimodal baselines can be compared under the same optimization style.
    if training_style in {"full_batch", "baseline_stream"}:
        return [torch.arange(n_samples, device=device)]
    if training_style == "event_batch":
        return event_aware_batch_indices(events, batch_size, min_events_per_batch)
    if training_style in {"random_batch", "rna_clinical_batch"}:
        perm = torch.randperm(n_samples, device=device)
        return [perm[start : start + batch_size] for start in range(0, n_samples, batch_size)]
    raise ValueError(f"Unknown training_style: {training_style}")


def train_tensor_unimodal(
    model: nn.Module,
    train_tensors: TensorTensors,
    *,
    device: torch.device,
    val_tensors: TensorTensors | None = None,
    epochs: int = 300,
    patience: int = 40,
    batch_size: int = 64,
    min_events_per_batch: int = 3,
    training_style: str = "full_batch",
    lr: float = 2e-4,
    weight_decay: float = 1e-5,
    grad_clip: float = 5.0,
) -> tuple[nn.Module, pd.DataFrame]:
    """Train RNA-only or clinical-only Cox models."""
    x_train, time_train, event_train = train_tensors
    if val_tensors is not None:
        x_val, time_val, event_val = val_tensors

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = deepcopy(model.state_dict())
    best_loss = math.inf
    wait = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        batches = _make_batches(len(x_train), event_train, training_style, batch_size, min_events_per_batch, device)
        for idx in batches:

            # RNA and clinical unimodal inputs are fixed-size tensors, so a batch
            # is selected by normal tensor indexing.
            optimizer.zero_grad()
            risk = model.forward_all(x_train[idx])
            loss = cox_ph_loss(risk, event_train[idx], time_train[idx])
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            train_risk_tensor = model.forward_all(x_train)
            train_full_loss = float(cox_ph_loss(train_risk_tensor, event_train, time_train).detach().cpu())
            val_loss = math.nan
            val_ci = math.nan
            if val_tensors is not None:
                val_risk_tensor = model.forward_all(x_val)
                val_loss = float(cox_ph_loss(val_risk_tensor, event_val, time_val).detach().cpu())
                val_ci = concordance_index(
                    val_risk_tensor.detach().cpu().numpy(),
                    time_val.detach().cpu().numpy(),
                    event_val.detach().cpu().numpy(),
                )

        mean_loss = float(np.mean(losses)) if losses else math.nan
        train_ci = concordance_index(
            train_risk_tensor.detach().cpu().numpy(),
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
        # Standard unimodal runs select the checkpoint with lowest validation
        # loss; if no validation split is provided, train batch loss is used.
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


def train_pathology_unimodal(
    model: nn.Module,
    pathology_train: list[torch.Tensor],
    time_train: torch.Tensor,
    event_train: torch.Tensor,
    *,
    device: torch.device,
    pathology_val: list[torch.Tensor] | None = None,
    time_val: torch.Tensor | None = None,
    event_val: torch.Tensor | None = None,
    epochs: int = 300,
    patience: int = 40,
    batch_size: int = 64,
    min_events_per_batch: int = 3,
    training_style: str = "full_batch",
    lr: float = 2e-4,
    weight_decay: float = 1e-5,
    grad_clip: float = 5.0,
) -> tuple[nn.Module, pd.DataFrame]:
    """Train pathology-only Cox models."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = deepcopy(model.state_dict())
    best_loss = math.inf
    wait = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        batches = _make_batches(len(pathology_train), event_train, training_style, batch_size, min_events_per_batch, device)
        for idx in batches:

            # Pathology bags are Python lists of variable-length tensors, so the
            # batch is gathered manually instead of by tensor indexing.
            batch_bags = [pathology_train[i] for i in idx.detach().cpu().tolist()]
            optimizer.zero_grad()
            risk = model.forward_all(batch_bags, device=device)
            loss = cox_ph_loss(risk, event_train[idx], time_train[idx])
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            train_risk_tensor = model.forward_all(pathology_train, device=device)
            train_full_loss = float(cox_ph_loss(train_risk_tensor, event_train, time_train).detach().cpu())
            val_loss = math.nan
            val_ci = math.nan
            if pathology_val is not None and time_val is not None and event_val is not None:
                val_risk_tensor = model.forward_all(pathology_val, device=device)
                val_loss = float(cox_ph_loss(val_risk_tensor, event_val, time_val).detach().cpu())
                val_ci = concordance_index(
                    val_risk_tensor.detach().cpu().numpy(),
                    time_val.detach().cpu().numpy(),
                    event_val.detach().cpu().numpy(),
                )

        mean_loss = float(np.mean(losses)) if losses else math.nan
        train_ci = concordance_index(
            train_risk_tensor.detach().cpu().numpy(),
            time_train.detach().cpu().numpy(),
            event_train.detach().cpu().numpy(),
        )
        monitor_loss = val_loss if pathology_val is not None else mean_loss
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
        # Keep the best pathology checkpoint by validation loss, matching the
        # tensor unimodal trainer.
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


def evaluate_tensor_unimodal(
    model: nn.Module,
    tensors: TensorTensors,
    sample_ids: list[str],
) -> tuple[float, pd.DataFrame]:
    """Evaluate RNA-only or clinical-only Cox models."""
    x, time, event = tensors
    model.eval()
    with torch.no_grad():
        risk = model.forward_all(x).detach().cpu().numpy()
    time_np = time.detach().cpu().numpy().astype(float)
    event_np = event.detach().cpu().numpy().astype(int)
    c_index = concordance_index(risk, time_np, event_np)

    # Risk tables use a shared schema across unimodal and multimodal models so
    # downstream summaries, KM plots, and late fusion can reuse them.
    return c_index, pd.DataFrame({"sample_id": sample_ids, "log_risk": risk, "Event": event_np, "Time": time_np})


def evaluate_pathology_unimodal(
    model: nn.Module,
    pathology_bags: list[torch.Tensor],
    time: torch.Tensor,
    event: torch.Tensor,
    sample_ids: list[str],
    device: torch.device,
) -> tuple[float, pd.DataFrame]:
    """Evaluate pathology-only Cox models."""
    model.eval()
    with torch.no_grad():
        risk = model.forward_all(pathology_bags, device=device).detach().cpu().numpy()
    time_np = time.detach().cpu().numpy().astype(float)
    event_np = event.detach().cpu().numpy().astype(int)
    c_index = concordance_index(risk, time_np, event_np)
    return c_index, pd.DataFrame({"sample_id": sample_ids, "log_risk": risk, "Event": event_np, "Time": time_np})
