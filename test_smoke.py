"""Smoke tests: package imports and public API surface."""

import frame_pinn


def test_version_present():
    assert isinstance(frame_pinn.__version__, str)
    assert frame_pinn.__version__.count(".") >= 1


def test_public_api_exported():
    expected = {
        "PINN", "ResidualSpec", "TermContribution", "MLP",
        "eliminability", "retrain_eliminability", "eliminability_trajectory",
        "ordering_defect", "OrderingDefect",
        "stability_eliminability", "stability_sigma",
        "train", "TrainLog", "EliminabilityResult",
    }
    assert expected.issubset(set(frame_pinn.__all__))
    for name in expected:
        assert hasattr(frame_pinn, name)


def test_residual_spec_build():
    spec = frame_pinn.ResidualSpec()
    spec.add("a", "g1", lambda net, b: net)
    spec.add("b", "g2", lambda net, b: net)
    assert spec.names() == ["a", "b"]
    assert spec.groups() == ["g1", "g2"]
