"""Install verification. Tiny budget, a few steps, no convergence expected.

Confirms that imports resolve, the autograd residuals assemble, the trainer runs,
and every diagnostic returns without error on all benchmarks. Runs in a few
seconds on a single CPU core.

    python examples/smoke_fast.py
"""

import sys
import time

import torch

from frame_pinn import (
    train,
    eliminability,
    retrain_eliminability,
    ordering_defect,
    stability_sigma,
)
from benchmarks import burgers, allen_cahn, navier_stokes, poisson_control

torch.manual_seed(0)

CFG = [
    (burgers, "Burgers", dict(n_pde=64, n_ic=16, n_bc=16),
     {"ic", "bc"}, {"u_t", "adv", "diff"}, "adv", False),
    (allen_cahn, "Allen-Cahn", dict(n_pde=64, n_ic=16, n_bc=16),
     {"ic", "bc"}, {"u_t", "diff", "react_cubic", "react_lin"}, "react_cubic", False),
    (navier_stokes, "Navier-Stokes", dict(n_pde=48, n_bc=16),
     {"bc"}, {"adv_x", "adv_y", "press_x", "press_y", "visc_x", "visc_y", "cont"},
     "cont", True),
    (poisson_control, "Poisson control", dict(n_pde=64),
     {"bc"}, {"pde"}, "pde", False),
]

STEPS = 3
ORDER_K = 2


def check(mod, name, batch_kw, data_terms, physics_terms, residual_term, grouped):
    t0 = time.time()
    bf = mod.batch_fn(**batch_kw)
    build = lambda: mod.build_model(width=16, depth=2)

    model = build()
    log = train(model, bf, steps=STEPS, lr=1e-3, snapshot_every=1, log_every=1)
    assert len(log.total_loss) >= 1

    er = eliminability(model, bf(), grouped=grouped)
    assert er.raw
    rer = retrain_eliminability(build, bf, train, steps=STEPS, lr=1e-3,
                                repeats=1, eval_batches=1, grouped=grouped)
    assert rer.raw

    od = ordering_defect(build, bf, data_terms, physics_terms,
                         mod.probe_grid(n=6), residual_term, k=ORDER_K, lr=1e-3)
    assert od.field_defect >= 0.0

    sig = stability_sigma(log)
    assert "loss_var" in sig

    print(f"  {name:16s} ok  ({time.time() - t0:4.1f}s)")


def main():
    print("install verification (tiny budget, no convergence):")
    for mod, name, bkw, dt, pt, rt, grouped in CFG:
        check(mod, name, bkw, dt, pt, rt, grouped)
    print("ALL OK")


if __name__ == "__main__":
    main()
