"""
Loss functions for survival prediction
======================================

Implements the negative Cox proportional hazards partial log-likelihood used to
train every survival model in this repository.

Risk convention
---------------
  higher model output = higher predicted relative risk

The Cox loss compares each observed event patient against the risk set of
patients who were still at risk at that event time. Censored patients contribute
to risk sets but do not create event terms themselves.
"""

from __future__ import annotations

import torch


def cox_ph_loss(risk: torch.Tensor, event: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """Negative Cox partial log-likelihood.

    ``risk`` is interpreted as log-risk, where larger values indicate higher
    hazard. ``event`` should be 1 for observed events and 0 for censored cases.
    """
    # Sort from longest to shortest time so the cumulative log-sum-exp at each
    # row represents the risk set for that event time.
    order = torch.argsort(time, descending=True)
    ordered_risk = risk[order]
    ordered_event = event[order]
    log_cumsum = torch.logcumsumexp(ordered_risk, dim=0)
    observed = ordered_event == 1
    if observed.sum() == 0:

        # A Cox batch with no observed events has no partial-likelihood terms.
        return torch.zeros((), device=risk.device, requires_grad=True)
    return -((ordered_risk[observed] - log_cumsum[observed]).sum() / observed.sum())
