"""Training loop with two features the diagnostics need: periodic weight
snapshots (for eliminability trajectories) and per-step instability signals
(for stability eliminability)."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from .residual import PINN


@dataclass
class TrainLog:
    total_loss: List[float] = field(default_factory=list)
    term_loss: Dict[str, List[float]] = field(default_factory=dict)
    grad_norm: List[float] = field(default_factory=list)
    snapshots: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    snapshot_steps: List[int] = field(default_factory=list)


def train(
    model: PINN,
    batch_fn: Callable[[], dict],
    steps: int = 5000,
    lr: float = 1e-3,
    active: Optional[set] = None,
    snapshot_every: Optional[int] = None,
    log_every: int = 50,
    optimizer_cls=torch.optim.Adam,
) -> TrainLog:
    """Train on the active set. batch_fn returns a fresh batch each call so
    collocation resampling is possible. active=None means the full set S."""
    opt = optimizer_cls(model.parameters(), lr=lr)
    log = TrainLog(term_loss={n: [] for n in model.spec.names()})

    for step in range(steps):
        batch = batch_fn()
        opt.zero_grad()
        ld = model.loss_dict(batch)
        names = set(ld.keys()) if active is None else set(active)
        total = torch.zeros((), device=next(model.parameters()).device)
        for n, v in ld.items():
            if n in names:
                total = total + model.weights[n] * v
        total.backward()

        gnorm = torch.sqrt(
            sum((p.grad.detach() ** 2).sum() for p in model.parameters() if p.grad is not None)
        )
        opt.step()

        if step % log_every == 0:
            log.total_loss.append(float(total.detach()))
            log.grad_norm.append(float(gnorm.detach()))
            for n, v in ld.items():
                log.term_loss[n].append(float(v.detach()))

        if snapshot_every and step % snapshot_every == 0:
            log.snapshots.append(copy.deepcopy(model.state_dict()))
            log.snapshot_steps.append(step)

    log.snapshots.append(copy.deepcopy(model.state_dict()))
    log.snapshot_steps.append(steps)
    return log


def stability_sigma(log: TrainLog, window: int = 20, spike_factor: float = 3.0) -> Dict[str, float]:
    """Candidate sigma metrics over the tail of a run.

    - loss_var: variance of total loss over the final window.
    - spike_rate: fraction of logged steps whose grad norm exceeds spike_factor
      times the running median (a proxy for stiffness-driven blowups)."""
    tl = torch.tensor(log.total_loss[-window:]) if log.total_loss else torch.tensor([0.0])
    gn = torch.tensor(log.grad_norm) if log.grad_norm else torch.tensor([0.0])
    med = torch.median(gn) if gn.numel() else torch.tensor(1.0)
    spikes = (gn > spike_factor * med).float().mean() if gn.numel() else torch.tensor(0.0)
    return {
        "loss_var": float(tl.var(unbiased=False)),
        "spike_rate": float(spikes),
    }
