"""Tests that every benchmark constructs and produces residuals."""

import pytest
import torch

from frame_pinn import train
from benchmarks import burgers, allen_cahn, navier_stokes, poisson_control

ALL = [burgers, allen_cahn, navier_stokes, poisson_control]


@pytest.mark.parametrize("mod", ALL)
def test_build_model_and_spec(mod):
    m = mod.build_model(width=16, depth=2)
    names = m.spec.names()
    assert len(names) >= 2
    # weights cover every term
    assert set(m.weights) >= set(names)


@pytest.mark.parametrize("mod", ALL)
def test_loss_dict_is_finite(mod):
    torch.manual_seed(0)
    m = mod.build_model(width=16, depth=2)
    bf = mod.batch_fn() if mod is not navier_stokes else mod.batch_fn(n_pde=48, n_bc=16)
    ld = m.loss_dict(bf())
    assert set(ld) == set(m.spec.names())
    for v in ld.values():
        assert torch.isfinite(v)


@pytest.mark.parametrize("mod", ALL)
def test_short_training_runs(mod):
    torch.manual_seed(0)
    m = mod.build_model(width=16, depth=2)
    bf = mod.batch_fn() if mod is not navier_stokes else mod.batch_fn(n_pde=48, n_bc=16)
    log = train(m, bf, steps=3, lr=1e-3, log_every=1)
    assert len(log.total_loss) >= 1
    assert all(torch.isfinite(torch.tensor(x)) for x in log.total_loss)
