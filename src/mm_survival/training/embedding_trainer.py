"""
Trainer for embedding-based multimodal Cox models
=================================================

Used by concat, gated, and low-rank fusion models after RNA, clinical, and
pathology inputs have already been converted to embeddings.

Training pipeline
-----------------
  1. Build event-aware or random mini-batches from the fit split.
  2. Optionally apply deterministic RNA dropout to training patients.
  3. Optimize the Cox partial likelihood.
  4. Evaluate validation data with complete modalities.
  5. If robust_missing_rna=True, also evaluate validation data with all RNA
     removed and use the average validation C-index for model selection.

Design rationale
----------------
* Training RNA dropout teaches the model to tolerate missing RNA.
* Validation/test dropout is not random; the same patients are evaluated in
  complete and missing-RNA settings for deterministic robustness estimates.
* The trainer accepts an rna_mask but leaves fusion-specific interpretation to
  the model, so concat, gated, and low-rank can handle missing RNA differently.
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
from mm_survival.training.rna_missing import apply_rna_mask, missing_rna_mask_like


def _batch_pathology(pathology_bags: list[torch.Tensor], indices: torch.Tensor) -> list[torch.Tensor]:
    return [pathology_bags[i] for i in indices.detach().cpu().tolist()]


def _make_batches(
    n_samples: int,
    events: torch.Tensor,
    training_style: str,
    batch_size: int,
    min_events_per_batch: int,
    device: torch.device,
) -> list[torch.Tensor]:

    # The original experiments used multiple training styles. Keeping this
    # routing here lets configs reproduce full-batch, random-batch, and
    # event-aware Cox optimization without changing model code.
    if training_style in {"full_batch", "baseline_stream"}:
        return [torch.arange(n_samples, device=device)]
    if training_style == "event_batch":
        return event_aware_batch_indices(events, batch_size, min_events_per_batch)
    if training_style in {"random_batch", "rna_clinical_batch"}:
        perm = torch.randperm(n_samples, device=device)
        return [perm[start : start + batch_size] for start in range(0, n_samples, batch_size)]
    raise ValueError(f"Unknown training_style: {training_style}")


def _forward_all(
    model: nn.Module,
    rna: torch.Tensor,
    clinical: torch.Tensor,
    pathology_bags: list[torch.Tensor],
    device: torch.device,
    rna_mask: torch.Tensor | None = None,
) -> torch.Tensor:

    # Older models do not require an RNA mask, while missing-RNA-aware fusion
    # modules use it to zero modality contributions in a model-specific way.
    if rna_mask is None:
        return model.forward_all(rna, clinical, pathology_bags, device)
    return model.forward_all(rna, clinical, pathology_bags, device, rna_mask=rna_mask)


def train_embedding_multimodal(
    model: nn.Module,
    train_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    pathology_train: list[torch.Tensor],
    *,
    device: torch.device,
    val_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    pathology_val: list[torch.Tensor] | None = None,
    epochs: int = 300,
    patience: int = 40,
    batch_size: int = 64,
    min_events_per_batch: int = 3,
    training_style: str = "full_batch",
    lr: float = 2e-4,
    weight_decay: float = 1e-5,
    grad_clip: float = 5.0,
    rna_train_mask: torch.Tensor | None = None,
    robust_missing_rna: bool = False,
) -> tuple[nn.Module, pd.DataFrame]:
    """Train an embedding-based multimodal Cox model."""
    rna_train, clinical_train, time_train, event_train = train_tensors
    if val_tensors is not None:
        rna_val, clinical_val, time_val, event_val = val_tensors

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = deepcopy(model.state_dict())
    best_loss = math.inf
    best_score = -math.inf
    wait = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        batches = _make_batches(
            n_samples=len(rna_train),
            events=event_train,
            training_style=training_style,
            batch_size=batch_size,
            min_events_per_batch=min_events_per_batch,
            device=device,
        )
        for idx in batches:
            batch_bags = _batch_pathology(pathology_train, idx)
            batch_rna_mask = rna_train_mask[idx] if rna_train_mask is not None else None

            # RNA dropout is applied only to training batches. The same mask is
            # also passed to the model so gated/low-rank modules can suppress
            # RNA-specific outputs, not just zero the raw RNA input.
            batch_rna = apply_rna_mask(rna_train[idx], batch_rna_mask)
            optimizer.zero_grad()
            risk = _forward_all(model, batch_rna, clinical_train[idx], batch_bags, device, rna_mask=batch_rna_mask)
            loss = cox_ph_loss(risk, event_train[idx], time_train[idx])
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():

            # Training metrics are computed under the same observed/missing RNA
            # mask used during optimization, so the reported train loss reflects
            # the actual training condition.
            train_eval_rna = apply_rna_mask(rna_train, rna_train_mask)
            train_risk_tensor = _forward_all(model, train_eval_rna, clinical_train, pathology_train, device, rna_mask=rna_train_mask)
            train_risk = train_risk_tensor.detach().cpu().numpy()
            train_full_loss = float(cox_ph_loss(train_risk_tensor, event_train, time_train).detach().cpu())
            val_loss = math.nan
            val_ci = math.nan
            val_complete_loss = math.nan
            val_complete_ci = math.nan
            val_missing_loss = math.nan
            val_missing_ci = math.nan
            val_avg_ci = math.nan
            if val_tensors is not None and pathology_val is not None:

                # Complete-modality validation: RNA, clinical, and pathology are
                # all available for the validation patients.
                val_risk_tensor = model.forward_all(rna_val, clinical_val, pathology_val, device)
                val_loss = float(cox_ph_loss(val_risk_tensor, event_val, time_val).detach().cpu())
                val_ci = concordance_index(
                    val_risk_tensor.detach().cpu().numpy(),
                    time_val.detach().cpu().numpy(),
                    event_val.detach().cpu().numpy(),
                )
                val_complete_loss = val_loss
                val_complete_ci = val_ci
                if robust_missing_rna:

                    # Missing-RNA validation is deterministic: every validation
                    # patient is evaluated again with RNA set to zero.
                    missing_val_mask = missing_rna_mask_like(rna_val)
                    missing_val_rna = apply_rna_mask(rna_val, missing_val_mask)
                    missing_val_risk = _forward_all(
                        model,
                        missing_val_rna,
                        clinical_val,
                        pathology_val,
                        device,
                        rna_mask=missing_val_mask,
                    )
                    val_missing_loss = float(cox_ph_loss(missing_val_risk, event_val, time_val).detach().cpu())
                    val_missing_ci = concordance_index(
                        missing_val_risk.detach().cpu().numpy(),
                        time_val.detach().cpu().numpy(),
                        event_val.detach().cpu().numpy(),
                    )
                    val_avg_ci = float((val_complete_ci + val_missing_ci) / 2.0)

        mean_loss = float(np.mean(losses)) if losses else math.nan
        train_ci = concordance_index(
            train_risk,
            time_train.detach().cpu().numpy(),
            event_train.detach().cpu().numpy(),
        )
        monitor_loss = val_loss if val_tensors is not None and pathology_val is not None else mean_loss
        monitor_score = val_avg_ci if robust_missing_rna and not math.isnan(val_avg_ci) else math.nan
        history.append(
            {
                "epoch": epoch,
                "train_loss": mean_loss,
                "train_full_loss": train_full_loss,
                "train_ci": train_ci,
                "val_loss": val_loss,
                "val_ci": val_ci,
                "val_complete_loss": val_complete_loss,
                "val_complete_ci": val_complete_ci,
                "val_missing_rna_loss": val_missing_loss,
                "val_missing_rna_ci": val_missing_ci,
                "val_avg_ci": val_avg_ci,
                "monitor_loss": monitor_loss,
                "monitor_score": monitor_score,
            }
        )

        # Standard experiments select the lowest validation loss. Robust
        # missing-RNA experiments select the best average of complete and
        # missing-RNA validation C-index.
        improved = monitor_score > best_score if robust_missing_rna else monitor_loss < best_loss
        if improved:
            best_loss = monitor_loss
            if robust_missing_rna:
                best_score = monitor_score
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def evaluate_embedding_multimodal(
    model: nn.Module,
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    pathology_bags: list[torch.Tensor],
    sample_ids: list[str],
    device: torch.device,
    rna_mask: torch.Tensor | None = None,
) -> tuple[float, pd.DataFrame]:
    """Evaluate an embedding-based multimodal Cox model."""
    rna, clinical, time, event = tensors
    model.eval()
    with torch.no_grad():

        # Passing rna_mask=None gives complete-modality evaluation; passing an
        # all-zero mask gives the missing-RNA test setting.
        eval_rna = apply_rna_mask(rna, rna_mask)
        risk = _forward_all(model, eval_rna, clinical, pathology_bags, device, rna_mask=rna_mask).detach().cpu().numpy()
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
