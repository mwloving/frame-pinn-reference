"""Stability eliminability.

    E_i^stab = sigma(S \\ i) - sigma(S)

A positive value means removing term i reduces training instability, so i is a
stiffness contributor; a negative value means removing i increases instability,
so i is stabilizing. sigma is any scalar instability metric from
``frame_pinn.trainer.stability_sigma`` (for example loss variance or the rate of
gradient-norm spikes).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from .residual import PINN
from .trainer import stability_sigma, train


def stability_eliminability(
    build_model: Callable[[], PINN],
    batch_fn: Callable[[], dict],
    sigma_key: str = "loss_var",
    steps: int = 3000,
    lr: float = 1e-3,
    repeats: int = 3,
) -> Dict[str, float]:
    """For each term, train the full set versus train-without and compare sigma.

    Averaged over ``repeats`` seeds because instability metrics are themselves
    noisy. ``build_model`` returns a fresh, freshly initialized PINN each call."""
    names = build_model().spec.names()

    def avg_sigma(active: Optional[set]) -> float:
        vals = []
        for _ in range(repeats):
            m = build_model()
            log = train(m, batch_fn, steps=steps, lr=lr, active=active, log_every=20)
            vals.append(stability_sigma(log)[sigma_key])
        return float(sum(vals) / len(vals))

    full = avg_sigma(None)
    out: Dict[str, float] = {}
    all_names = set(names)
    for n in names:
        out[n] = avg_sigma(all_names - {n}) - full
    return out
