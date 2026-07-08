"""1D Poisson control problem: u_xx = f on x in [-1, 1] with u(-1) = u(1) = 0.

The forcing is chosen so the exact solution is u(x) = sin(pi x), giving
f(x) = -pi^2 sin(pi x). This benchmark exists to demonstrate the eliminability
sign convention discussed in the paper. Because the problem has a unique solution
only when the boundary condition is present, retraining without the boundary term
produces a model that cannot reproduce the solution, so the boundary term carries
a large positive retrain eliminability. It is the clearest worked check that
"least eliminable" lines up with "physically necessary."
"""

from __future__ import annotations

import math

import torch

from frame_pinn.utils import grad
from frame_pinn.residual import MLP, PINN, ResidualSpec


def _r_pde(net, batch):
    X = batch["pde"].clone().requires_grad_(True)
    u = net(X)
    u_x = grad(u, X)
    u_xx = grad(u_x, X)
    f = -(math.pi**2) * torch.sin(math.pi * X)
    return u_xx - f


def _r_bc(net, batch):
    return net(batch["bc"])


def build_spec() -> ResidualSpec:
    spec = ResidualSpec()
    spec.add("pde", "pde", _r_pde)
    spec.add("bc", "data", _r_bc)
    return spec


def build_model(width=40, depth=4, weights=None) -> PINN:
    net = MLP(in_dim=1, out_dim=1, width=width, depth=depth)
    w = {"pde": 1.0, "bc": 20.0}
    if weights:
        w.update(weights)
    return PINN(net, build_spec(), weights=w)


def batch_fn(n_pde=400, device="cpu"):
    def _f():
        x = torch.rand(n_pde, 1, device=device) * 2 - 1
        bc = torch.tensor([[-1.0], [1.0]], device=device)
        return {"pde": x, "bc": bc}

    return _f


def probe_grid(n=80, device="cpu"):
    return torch.linspace(-1, 1, n, device=device).reshape(-1, 1)


def exact(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(math.pi * x)
