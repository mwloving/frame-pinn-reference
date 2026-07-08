"""Ordering defect.

    Lambda_O = || A(B(x)) - B(A(x)) ||

where A optimizes the data loss only and B the physics (PDE + boundary) loss
only. A large defect on the solution fields, residuals, or the eliminability
profile means the two constraint families do not commute as optimization
operators: path dependence in training is structural, not merely a different
local minimum.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, List

import torch

from .eliminability import eliminability
from .residual import PINN
from .trainer import train


@dataclass
class OrderingDefect:
    field_defect: float          # ||u_AB - u_BA|| on a probe grid
    residual_defect: float       # difference in PDE residual norm
    elim_l1: float               # L1 distance between eliminability profiles
    rank_changed: bool           # did the eliminability ranking reorder?
    ranking_AB: List[str]
    ranking_BA: List[str]


def _phase_train(
    model: PINN,
    batch_fn: Callable[[], dict],
    schedule: List[tuple],
    lr: float,
) -> None:
    """schedule is a list of (active_set, n_steps). Runs phases in order, with a
    fresh optimizer per phase so the phases are cleanly isolated."""
    for active, k in schedule:
        train(model, batch_fn, steps=k, lr=lr, active=active, log_every=max(1, k // 5))


def ordering_defect(
    build_model: Callable[[], PINN],
    batch_fn: Callable[[], dict],
    data_terms: set,
    physics_terms: set,
    probe: torch.Tensor,
    residual_term: str,
    k: int = 1500,
    lr: float = 1e-3,
    seed: int = 0,
) -> OrderingDefect:
    """Protocol 1 (A then B) versus Protocol 2 (B then A), measured three ways.

    A is the data-only phase, B is the physics-only phase. Both protocols start
    from the same initialization so the only difference is the order."""
    torch.manual_seed(seed)
    m_ab = build_model()
    init = copy.deepcopy(m_ab.state_dict())
    m_ba = build_model()
    m_ba.load_state_dict(copy.deepcopy(init))

    _phase_train(m_ab, batch_fn, [(data_terms, k), (physics_terms, k)], lr)
    _phase_train(m_ba, batch_fn, [(physics_terms, k), (data_terms, k)], lr)

    with torch.no_grad():
        u_ab = m_ab.net(probe)
        u_ba = m_ba.net(probe)
        field_defect = float(torch.linalg.vector_norm(u_ab - u_ba))

    b = batch_fn()
    names = [t.name for t in m_ab.spec.terms]
    idx = names.index(residual_term)
    r_ab = float(torch.mean(m_ab.spec.terms[idx].fn(m_ab.net, b) ** 2).detach())
    r_ba = float(torch.mean(m_ba.spec.terms[idx].fn(m_ba.net, b) ** 2).detach())
    residual_defect = abs(r_ab - r_ba)

    e_ab = eliminability(m_ab, b)
    e_ba = eliminability(m_ba, b)
    elim_l1 = sum(abs(e_ab.normalized[n] - e_ba.normalized[n]) for n in e_ab.normalized)
    rank_ab = [n for n, _ in e_ab.ranking()]
    rank_ba = [n for n, _ in e_ba.ranking()]

    return OrderingDefect(
        field_defect=field_defect,
        residual_defect=residual_defect,
        elim_l1=float(elim_l1),
        rank_changed=(rank_ab != rank_ba),
        ranking_AB=rank_ab,
        ranking_BA=rank_ba,
    )
