"""Eliminability as a measurable operator.

Two distinct operators, because they answer different questions:

1. FROZEN-WEIGHT eliminability (eliminability / eliminability_trajectory):
       E_i = L(S \\ i; theta*) - L(S; theta*)
   recomputed on the SAME converged weights theta*. This is a cheap marginal
   sensitivity: how much of the residual budget term i currently occupies. It is
   the right object for trajectory plots E_i(theta_t) and for stability bands,
   but at low loss it is dominated by which terms still carry residual, not by
   structural necessity. It can be near zero or negative for a term that is in
   fact essential, simply because the converged solution already satisfies it.

2. RETRAIN eliminability (retrain_eliminability):
       E_i^retrain = L*(S \\ i) - L*(S)
   where each L* is the loss reached by RETRAINING from scratch on the reduced
   set, then evaluated on the FULL set S. This is the structural-necessity
   object the model-reduction story needs: a large positive value means a model
   trained without term i cannot reproduce the full physics, so i is not
   eliminable. This is more expensive (one retrain per term) but is what
   pruning decisions should rest on.

Normalized variants divide by the base loss for cross-problem comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .residual import PINN


@dataclass
class EliminabilityResult:
    raw: Dict[str, float]          # E_i
    normalized: Dict[str, float]   # Etilde_i
    base_loss: float               # L(S)

    def ranking(self, normalized: bool = True) -> List[tuple]:
        """Terms sorted descending by importance (least eliminable first)."""
        src = self.normalized if normalized else self.raw
        return sorted(src.items(), key=lambda kv: kv[1], reverse=True)

    def eliminable(self, eps: float, normalized: bool = True) -> List[str]:
        """Names whose eliminability falls below threshold eps."""
        src = self.normalized if normalized else self.raw
        return [n for n, v in src.items() if v < eps]


def _loss_value(model: PINN, batch: dict, active: Optional[set]) -> float:
    """Scalar loss with autograd available for PDE derivatives but no parameter
    graph retained. We detach the final scalar."""
    val = model.total_loss(batch, active=active)
    return float(val.detach().cpu())


def eliminability(
    model: PINN,
    batch: dict,
    grouped: bool = False,
    eps_floor: float = 1e-30,
) -> EliminabilityResult:
    """Compute E_i and Etilde_i over the current weights.

    If grouped, the removed set for an operator is all terms sharing that group,
    giving operator-level eliminability (advective, pressure, viscous, ...)."""
    all_names = set(model.spec.names())
    base = _loss_value(model, batch, active=all_names)

    raw: Dict[str, float] = {}
    if grouped:
        for g in model.spec.groups():
            members = {t.name for t in model.spec.terms if t.group == g}
            without = _loss_value(model, batch, active=all_names - members)
            raw[g] = without - base
    else:
        for name in model.spec.names():
            without = _loss_value(model, batch, active=all_names - {name})
            raw[name] = without - base

    denom = max(abs(base), eps_floor)
    norm = {k: v / denom for k, v in raw.items()}
    return EliminabilityResult(raw=raw, normalized=norm, base_loss=base)


def eliminability_trajectory(
    model: PINN,
    batch: dict,
    snapshots: List[Dict[str, torch.Tensor]],
    grouped: bool = False,
) -> List[EliminabilityResult]:
    """Replay saved state_dicts to build E_i(theta_t) along training.

    snapshots is a list of cloned state_dicts captured during the run. We restore
    each, measure eliminability, then leave the model on the last snapshot."""
    results: List[EliminabilityResult] = []
    for sd in snapshots:
        model.load_state_dict(sd)
        results.append(eliminability(model, batch, grouped=grouped))
    return results


def retrain_eliminability(
    build_model,
    batch_fn,
    train_fn,
    steps: int = 3000,
    lr: float = 1e-3,
    repeats: int = 1,
    grouped: bool = False,
    eval_batches: int = 4,
    eps_floor: float = 1e-30,
) -> EliminabilityResult:
    """Structural eliminability via retraining.

    For the full set and for each removed term (or group), train a fresh model
    on the reduced active set, then score it on the FULL set S. The gap is how
    much capability is lost by never seeing that constraint. build_model returns
    a fresh PINN; train_fn(model, batch_fn, steps, lr, active) trains in place.

    `repeats` averages over initializations; `eval_batches` averages the final
    full-set loss over several batches to reduce sampling noise."""
    template = build_model()
    all_names = set(template.spec.names())

    def trained_full_loss(active: Optional[set]) -> float:
        vals = []
        for _ in range(repeats):
            m = build_model()
            train_fn(m, batch_fn, steps=steps, lr=lr, active=active)
            # score on the FULL set
            batch_vals = []
            for _ in range(eval_batches):
                batch_vals.append(_loss_value(m, batch_fn(), active=all_names))
            vals.append(sum(batch_vals) / len(batch_vals))
        return sum(vals) / len(vals)

    base = trained_full_loss(all_names)
    raw: Dict[str, float] = {}
    if grouped:
        for g in template.spec.groups():
            members = {t.name for t in template.spec.terms if t.group == g}
            raw[g] = trained_full_loss(all_names - members) - base
    else:
        for name in template.spec.names():
            raw[name] = trained_full_loss(all_names - {name}) - base

    denom = max(abs(base), eps_floor)
    norm = {k: v / denom for k, v in raw.items()}
    return EliminabilityResult(raw=raw, normalized=norm, base_loss=base)
