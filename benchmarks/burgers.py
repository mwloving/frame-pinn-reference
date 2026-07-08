"""1D viscous Burgers: u_t + u u_x - nu u_xx = 0 on x in [-1,1], t in [0,1].
IC: u(0,x) = -sin(pi x). BC: u(t,+-1) = 0.

The PDE residual is decomposed into three primitive terms so each is
individually eliminable: time derivative (u_t), advection (u u_x), diffusion
(nu u_xx). Plus separate IC and BC data terms."""

from __future__ import annotations

import math

import torch

from frame_pinn.utils import column, grad
from frame_pinn.residual import MLP, PINN, ResidualSpec

NU = 0.01 / math.pi


def _xt(net, batch, key):
    X = batch[key].clone().requires_grad_(True)
    u = net(X)
    return X, u


def _r_time(net, batch):
    X, u = _xt(net, batch, "pde")
    g = grad(u, X)
    u_t = column(g, 1)
    # store nothing; each term recomputes independently for clean eliminability
    return u_t


def _r_adv(net, batch):
    X, u = _xt(net, batch, "pde")
    g = grad(u, X)
    u_x = column(g, 0)
    return u * u_x


def _r_diff(net, batch):
    X, u = _xt(net, batch, "pde")
    g = grad(u, X)
    u_x = column(g, 0)
    u_xx = column(grad(u_x, X), 0)
    return -NU * u_xx


def _r_ic(net, batch):
    X = batch["ic"]
    u = net(X)
    target = -torch.sin(math.pi * column(X, 0))
    return u - target


def _r_bc(net, batch):
    X = batch["bc"]
    return net(X)


def build_spec() -> ResidualSpec:
    spec = ResidualSpec()
    spec.add("u_t", "time", _r_time)
    spec.add("adv", "advective", _r_adv)
    spec.add("diff", "viscous", _r_diff)
    spec.add("ic", "data", _r_ic)
    spec.add("bc", "data", _r_bc)
    return spec


def build_model(width=40, depth=5, weights=None) -> PINN:
    net = MLP(in_dim=2, out_dim=1, width=width, depth=depth)
    w = {"u_t": 1.0, "adv": 1.0, "diff": 1.0, "ic": 10.0, "bc": 10.0}
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
