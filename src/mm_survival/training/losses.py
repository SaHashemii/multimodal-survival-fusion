"""Loss functions for survival prediction."""

from __future__ import annotations

import torch


def cox_ph_loss(risk: torch.Tensor, event: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """Negative Cox partial log-likelihood.

    ``risk`` is interpreted as log-risk, where larger values indicate higher
    hazard. ``event`` should be 1 for observed events and 0 for censored cases.
    """
    order = torch.argsort(time, descending=True)
    ordered_risk = risk[order]
    ordered_event = event[order]
    log_cumsum = torch.logcumsumexp(ordered_risk, dim=0)
    observed = ordered_event == 1
    if observed.sum() == 0:
        return torch.zeros((), device=risk.device, requires_grad=True)
    return -((ordered_risk[observed] - log_cumsum[observed]).sum() / observed.sum())
