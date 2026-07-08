"""Allen-Cahn: u_t - d u_xx + 5(u^3 - u) = 0, x in [-1,1], t in [0,1].
IC: u(0,x) = x^2 cos(pi x). Periodic-ish Dirichlet BC: u(t,+-1) = -1.

Decomposed terms: time (u_t), diffusion (-d u_xx), reaction cubic (5 u^3),
reaction linear (-5 u). Splitting the reaction into cubic and linear pieces
lets eliminability probe which nonlinearity the learned solution leans on."""

from __future__ import annotations

import math

import torch

from frame_pinn.utils import column, grad
from frame_pinn.residual import MLP, PINN, ResidualSpec

D = 1e-4


def _r_time(net, batch):
    X = batch["pde"].clone().requires_grad_(True)
    u = net(X)
    return column(grad(u, X), 1)


def _r_diff(net, batch):
    X = batch["pde"].clone().requires_grad_(True)
    u = net(X)
    u_x = column(grad(u, X), 0)
    u_xx = column(grad(u_x, X), 0)
    return -D * u_xx


def _r_cubic(net, batch):
    X = batch["pde"]
    u = net(X)
    return 5.0 * u**3


def _r_linear(net, batch):
    X = batch["pde"]
    u = net(X)
    return -5.0 * u


def _r_ic(net, batch):
    X = batch["ic"]
    u = net(X)
    x = column(X, 0)
    return u - (x**2 * torch.cos(math.pi * x))


def _r_bc(net, batch):
    X = batch["bc"]
    return net(X) - (-1.0)


def build_spec() -> ResidualSpec:
    spec = ResidualSpec()
    spec.add("u_t", "time", _r_time)
    spec.add("diff", "viscous", _r_diff)
    spec.add("react_cubic", "reaction", _r_cubic)
    spec.add("react_lin", "reaction", _r_linear)
    spec.add("ic", "data", _r_ic)
    spec.add("bc", "data", _r_bc)
    return spec


def build_model(width=64, depth=5, weights=None) -> PINN:
    net = MLP(in_dim=2, out_dim=1, width=width, depth=depth)
    w = {"u_t": 1.0, "diff": 1.0, "react_cubic": 1.0, "react_lin": 1.0, "ic": 20.0, "bc": 20.0}
    if weights:
        w.update(weights)
    return PINN(net, build_spec(), weights=w)


def batch_fn(n_pde=2000, n_ic=200, n_bc=200, device="cpu"):
    def _f():
        x = torch.rand(n_pde, 1, device=device) * 2 - 1
        t = torch.rand(n_pde, 1, device=device)
        pde = torch.cat([x, t], dim=1)
        xic = torch.rand(n_ic, 1, device=device) * 2 - 1
        ic = torch.cat([xic, torch.zeros_like(xic)], dim=1)
        tbc = torch.rand(n_bc, 1, device=device)
        sign = (torch.randint(0, 2, (n_bc, 1), device=device).float() * 2 - 1)
        bc = torch.cat([sign, tbc], dim=1)
        return {"pde": pde, "ic": ic, "bc": bc}

    return _f


def probe_grid(n=40, device="cpu"):
    xs = torch.linspace(-1, 1, n)
    ts = torch.linspace(0, 1, n)
    gx, gt = torch.meshgrid(xs, ts, indexing="ij")
    return torch.stack([gx.reshape(-1), gt.reshape(-1)], dim=1).to(device)
