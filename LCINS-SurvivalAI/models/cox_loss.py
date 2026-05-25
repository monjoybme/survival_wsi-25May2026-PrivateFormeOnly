"""
models/cox_loss.py
------------------
Negative log partial likelihood (Cox PH loss) for survival analysis.

Loss formula (Breslow approximation for ties):

    L = -∑_{i ∈ U} [ R_i  −  log ∑_{j ∈ Ω_i} exp(R_j) ]

where:
    U    = set of uncensored (event=1) instances in the mini-batch
    Ω_i  = risk set for i: all j in the batch with survival_time_j ≥ survival_time_i
    R_i  = predicted log-hazard (risk score) for instance i

The loss is averaged over uncensored instances for numerical stability across
batches of different sizes.

Reference:
    Cox, D.R. (1972). Regression models and life tables.
    Journal of the Royal Statistical Society, Series B.
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Functional Cox Loss
# ──────────────────────────────────────────────────────────────────────────────

def cox_partial_likelihood_loss(
    risk_scores:    torch.Tensor,
    survival_times: torch.Tensor,
    events:         torch.Tensor,
    reduction:      str = "mean",
    eps:            float = 1e-7,
) -> torch.Tensor:
    """
    Compute the negative Cox partial log-likelihood for a mini-batch.

    Args:
        risk_scores:    (N,) float — model output log-hazards.
        survival_times: (N,) float — patient survival times (days/months).
        events:         (N,) float — event indicators (1 = death, 0 = censored).
        reduction:      "mean" | "sum" — how to reduce over uncensored cases.
        eps:            Small constant for numerical stability.

    Returns:
        Scalar loss tensor (supports autograd).
    """
    n = risk_scores.size(0)
    if n == 0:
        return torch.tensor(0.0, device=risk_scores.device, requires_grad=True)

    # ── 1. Sort by survival time ascending ───────────────────────────────────
    # After ascending sort, for sample at position i the risk set
    # Ω_i = {j : t_j ≥ t_i} corresponds to all positions j = i, i+1, ..., N-1
    # (because t[j] ≥ t[i] for j ≥ i).

    order           = torch.argsort(survival_times, descending=False)
    risk_sorted     = risk_scores[order]         # (N,)
    events_sorted   = events[order]              # (N,)
    # survival times are now non-decreasing — we do not need them further

    # ── 2. Numerically stable cumulative sum from the right ──────────────────
    # log(∑_{j=i}^{N-1} exp(R_j)) computed via log-sum-exp trick
    # Max-shift: use the global max for stability
    max_risk        = risk_sorted.max().detach()
    exp_risk        = torch.exp(risk_sorted - max_risk)           # (N,)

    # Right-to-left (reverse) cumulative sum
    # rev_cumsum[i] = ∑_{j=i}^{N-1} exp(R_j - max_R)
    rev_cumsum      = torch.flip(
        torch.cumsum(torch.flip(exp_risk, dims=[0]), dim=0),
        dims=[0],
    )                                                              # (N,)

    log_risk_set    = torch.log(rev_cumsum + eps) + max_risk      # (N,)

    # ── 3. Per-sample partial likelihood contribution ─────────────────────────
    log_pl          = risk_sorted - log_risk_set                  # (N,)

    # ── 4. Sum / mean over uncensored cases ──────────────────────────────────
    uncensored_mask = events_sorted.bool()
    n_events        = uncensored_mask.sum()

    if n_events == 0:
        # No events in this batch — return zero loss, keep graph for safety
        return torch.tensor(0.0, device=risk_scores.device, dtype=risk_scores.dtype)

    if reduction == "mean":
        loss = -log_pl[uncensored_mask].mean()
    elif reduction == "sum":
        loss = -log_pl[uncensored_mask].sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction!r}")

    return loss


# ──────────────────────────────────────────────────────────────────────────────
# nn.Module Wrapper
# ──────────────────────────────────────────────────────────────────────────────

class CoxLoss(nn.Module):
    """
    nn.Module wrapper around `cox_partial_likelihood_loss` for use inside
    the training loop.

    Args:
        reduction: "mean" | "sum"
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        risk_scores:    torch.Tensor,
        survival_times: torch.Tensor,
        events:         torch.Tensor,
    ) -> torch.Tensor:
        return cox_partial_likelihood_loss(
            risk_scores, survival_times, events, self.reduction
        )

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}"


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    n = 32
    risk   = torch.randn(n, requires_grad=True)
    times  = torch.rand(n) * 100
    events = (torch.rand(n) > 0.3).float()

    loss_fn = CoxLoss(reduction="mean")
    loss    = loss_fn(risk, times, events)
    print(f"Cox loss (random inputs, n={n}): {loss.item():.4f}")
    loss.backward()
    print(f"Gradient norm: {risk.grad.norm().item():.4f}")
