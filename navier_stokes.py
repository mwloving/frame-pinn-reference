"""Steady 2D lid-driven cavity, velocity-pressure form on [0,1]^2.

Momentum residuals (x and y):
  advective:  u u_x + v u_y      |  u v_x + v v_y
  pressure:   p_x                 |  p_y
  viscous:   -nu (u_xx + u_yy)    | -nu (v_xx + v_yy)
Continuity: u_x + v_y = 0.

Terms are grouped (advective / pressure / viscous / continuity / data) so
eliminability can be reported per grouped operator, which is the natural object
for Navier-Stokes and is expected to correlate with Reynolds number."""

from __future__ import annotations

import torch

from frame_pinn.utils import column, grad
from frame_pinn.residual import MLP, PINN, ResidualSpec

NU = 0.01  # Re ~ 1/nu for unit lid velocity and unit cavity


def _fields(net, X):
    """net outputs (u, v, p)."""
    out = net(X)
    return out[:, 0:1], out[:, 1:2], out[:, 2:3]


def _derivs(net, batch, key="pde"):
    X = batch[key].clone().requires_grad_(True)
    u, v, p = _fields(net, X)
    gu, gv, gp = grad(u, X), grad(v, X), grad(p, X)
    u_x, u_y = column(gu, 0), column(gu, 1)
    v_x, v_y = column(gv, 0), column(gv, 1)
    p_x, p_y = column(gp, 0), column(gp, 1)
    u_xx = column(grad(u_x, X), 0)
    u_yy = column(grad(u_y, X), 1)
    v_xx = column(grad(v_x, X), 0)
    v_yy = column(grad(v_y, X), 1)
    return dict(u=u, v=v, p=p, u_x=u_x, u_y=u_y, v_x=v_x, v_y=v_y,
                p_x=p_x, p_y=p_y, u_xx=u_xx, u_yy=u_yy, v_xx=v_xx, v_yy=v_yy)


def _adv_x(net, b):
    d = _derivs(net, b)
    return d["u"] * d["u_x"] + d["v"] * d["u_y"]


def _adv_y(net, b):
    d = _derivs(net, b)
    return d["u"] * d["v_x"] + d["v"] * d["v_y"]


def _press_x(net, b):
    return _derivs(net, b)["p_x"]


def _press_y(net, b):
    return _derivs(net, b)["p_y"]


def _visc_x(net, b):
    d = _derivs(net, b)
    return -NU * (d["u_xx"] + d["u_yy"])


def _visc_y(net, b):
    d = _derivs(net, b)
    return -NU * (d["v_xx"] + d["v_yy"])


def _cont(net, b):
    d = _derivs(net, b)
    return d["u_x"] + d["v_y"]


def _bc(net, b):
    """Lid (top, y=1): u=1, v=0. Other walls: u=v=0. Provided as target tensors
    in the batch."""
    X = b["bc"]
    u, v, _ = _fields(net, X)
    tu, tv = b["bc_u"], b["bc_v"]
    return torch.cat([u - tu, v - tv], dim=0)


def build_spec() -> ResidualSpec:
    spec = ResidualSpec()
    spec.add("adv_x", "advective", _adv_x)
    spec.add("adv_y", "advective", _adv_y)
    spec.add("press_x", "pressure", _press_x)
    spec.add("press_y", "pressure", _press_y)
    spec.add("visc_x", "viscous", _visc_x)
    spec.add("visc_y", "viscous", _visc_y)
    spec.add("cont", "continuity", _cont)
    spec.add("bc", "data", _bc)
    return spec


def build_model(width=64, depth=6, weights=None) -> PINN:
    net = MLP(in_dim=2, out_dim=3, width=width, depth=depth)
    w = {n: 1.0 for n in build_spec().names()}
    w["bc"] = 10.0
    if weights:
        w.update(weights)
    return PINN(net, build_spec(), weights=w)


def batch_fn(n_pde=3000, n_bc=400, device="cpu"):
    def _f():
        xy = torch.rand(n_pde, 2, device=device)
        # boundary points sampled on the four walls
        side = torch.randint(0, 4, (n_bc,), device=device)
        s = torch.rand(n_bc, 1, device=device)
        pts = torch.zeros(n_bc, 2, device=device)
        tu = torch.zeros(n_bc, 1, device=device)
        tv = torch.zeros(n_bc, 1, device=device)
        # 0 bottom,1 top(lid),2 left,3 right
        bottom = side == 0
        top = side == 1
        left = side == 2
        right = side == 3
        pts[bottom.squeeze() if bottom.dim() > 1 else bottom] = torch.cat(
            [s[bottom], torch.zeros_like(s[bottom])], dim=1)
        pts[top] = torch.cat([s[top], torch.ones_like(s[top])], dim=1)
        pts[left] = torch.cat([torch.zeros_like(s[left]), s[left]], dim=1)
        pts[right] = torch.cat([torch.ones_like(s[right]), s[right]], dim=1)
        tu[top] = 1.0
        return {"pde": xy, "bc": pts, "bc_u": tu, "bc_v": tv}

    return _f


def probe_grid(n=30, device="cpu"):
    xs = torch.linspace(0, 1, n)
    ys = torch.linspace(0, 1, n)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1).to(device)
