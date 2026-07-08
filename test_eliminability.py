"""Tests for the eliminability operators."""

import torch

from frame_pinn import (
    eliminability,
    retrain_eliminability,
    train,
    EliminabilityResult,
)
from benchmarks import poisson_control


def test_frozen_eliminability_returns_all_terms():
    torch.manual_seed(0)
    m = poisson_control.build_model(width=16, depth=2)
    bf = poisson_control.batch_fn(n_pde=64)
    train(m, bf, steps=20, lr=1e-3, log_every=5)
    res = eliminability(m, bf())
    assert isinstance(res, EliminabilityResult)
    assert set(res.raw) == set(m.spec.names())
    assert set(res.normalized) == set(m.spec.names())


def test_ranking_is_sorted_descending():
    torch.manual_seed(0)
    m = poisson_control.build_model(width=16, depth=2)
    bf = poisson_control.batch_fn(n_pde=64)
    train(m, bf, steps=20, lr=1e-3, log_every=5)
    ranking = eliminability(m, bf()).ranking()
    values = [v for _, v in ranking]
    assert values == sorted(values, reverse=True)


def test_retrain_eliminability_marks_boundary_necessary():
    # On Poisson with a known unique solution, removing the boundary term and
    # retraining should be clearly non-eliminable (large positive score). This
    # is the sign-convention check from the paper.
    torch.manual_seed(0)
    build = lambda: poisson_control.build_model(width=24, depth=3)
    bf = poisson_control.batch_fn(n_pde=200)
    res = retrain_eliminability(build, bf, train, steps=400, lr=2e-3, repeats=1)
    # boundary should not be the most eliminable term
    most_eliminable = res.ranking()[-1][0]
    assert most_eliminable != "bc"
    assert res.raw["bc"] > 0.0


def test_grouped_eliminability_uses_group_names():
    torch.manual_seed(0)
    m = poisson_control.build_model(width=16, depth=2)
    bf = poisson_control.batch_fn(n_pde=64)
    train(m, bf, steps=10, lr=1e-3, log_every=5)
    res = eliminability(m, bf(), grouped=True)
    assert set(res.raw) == set(m.spec.groups())
