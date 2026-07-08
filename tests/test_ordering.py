"""Tests for the ordering defect operator."""

import torch

from frame_pinn import ordering_defect, OrderingDefect
from benchmarks import poisson_control


def test_ordering_defect_returns_nonnegative_field_defect():
    torch.manual_seed(0)
    build = lambda: poisson_control.build_model(width=16, depth=2)
    bf = poisson_control.batch_fn(n_pde=64)
    od = ordering_defect(
        build, bf,
        data_terms={"bc"},
        physics_terms={"pde"},
        probe=poisson_control.probe_grid(n=8),
        residual_term="pde",
        k=5, lr=1e-3, seed=0,
    )
    assert isinstance(od, OrderingDefect)
    assert od.field_defect >= 0.0
    assert od.residual_defect >= 0.0


def test_ordering_defect_rankings_have_all_terms():
    torch.manual_seed(0)
    build = lambda: poisson_control.build_model(width=16, depth=2)
    bf = poisson_control.batch_fn(n_pde=64)
    od = ordering_defect(
        build, bf,
        data_terms={"bc"},
        physics_terms={"pde"},
        probe=poisson_control.probe_grid(n=8),
        residual_term="pde",
        k=5, lr=1e-3, seed=0,
    )
    assert set(od.ranking_AB) == {"pde", "bc"}
    assert set(od.ranking_BA) == {"pde", "bc"}
