"""MLP backbone and a PINN wrapper whose physics residual is decomposed into
named, individually toggleable terms. Term toggling is what makes eliminability
measurable: we recompute the loss with a term's set membership removed."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Plain tanh MLP. Tanh is the standard PINN activation because it gives
    smooth, nonzero higher derivatives, which the autograd-based PDE residuals
    require."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int = 64,
        depth: int = 4,
        activation: Callable[[], nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(in_dim, width), activation()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), activation()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TermContribution:
    """A single named residual term plus the loss group it belongs to.

    A term computes a per-point residual tensor. Grouping lets us report
    eliminability both per primitive term and per grouped operator (advective,
    pressure, viscous) as the methodology asks."""

    name: str
    group: str
    fn: Callable[..., torch.Tensor]


@dataclass
class ResidualSpec:
    """Declarative description of a PINN loss as a set S of terms.

    Each entry is a TermContribution. Eliminability removes element i from S and
    recomputes the loss, so the loss assembler must respect an 'active set'."""

    terms: List[TermContribution] = field(default_factory=list)

    def names(self) -> List[str]:
        return [t.name for t in self.terms]

    def groups(self) -> List[str]:
        seen: List[str] = []
        for t in self.terms:
            if t.group not in seen:
                seen.append(t.group)
        return seen

    def add(self, name: str, group: str, fn: Callable[..., torch.Tensor]) -> "ResidualSpec":
        self.terms.append(TermContribution(name=name, group=group, fn=fn))
        return self


class PINN(nn.Module):
    """Wraps an MLP and a ResidualSpec.

    The PDE problem provides a closure that, given the network and a batch of
    collocation/boundary/data inputs, returns a dict mapping each term name to
    its per-point residual tensor. This class turns that dict into a scalar loss
    while honoring an active-set mask used for eliminability."""

    def __init__(
        self,
        net: MLP,
        spec: ResidualSpec,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.net = net
        self.spec = spec
        self.weights = {n: 1.0 for n in spec.names()}
        if weights:
            self.weights.update(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def loss_dict(self, batch: dict) -> Dict[str, torch.Tensor]:
        """Return mean-squared scalar loss per term, unweighted."""
        out: Dict[str, torch.Tensor] = {}
        for term in self.spec.terms:
            r = term.fn(self.net, batch)
            out[term.name] = torch.mean(r**2)
        return out

    def total_loss(
        self,
        batch: dict,
        active: Optional[Iterable[str]] = None,
    ) -> torch.Tensor:
        """Weighted sum over the active set. If active is None, all terms are in
        S. Passing active = S \\ {i} realizes the eliminability removal."""
        ld = self.loss_dict(batch)
        if active is None:
            active = set(ld.keys())
        else:
            active = set(active)
        total = torch.zeros((), device=next(self.parameters()).device)
        for name, val in ld.items():
            if name in active:
                total = total + self.weights[name] * val
        return total
