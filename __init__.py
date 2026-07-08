"""FRAME-PINN: structural diagnostics for physics-informed neural networks.

A reference implementation of eliminability (frozen and retrain), an ordering
defect, and a stability diagnostic, operating on a PINN loss that is expressed as
an explicit set of named, individually addressable residual terms.

This package accompanies the SoftwareX article "FRAME-PINN: A Software Framework
for Structural Diagnostics in Physics-Informed Neural Networks." It is a research
reference implementation, not a full research platform.
"""

from .residual import MLP, PINN, ResidualSpec, TermContribution
from .eliminability import (
    EliminabilityResult,
    eliminability,
    eliminability_trajectory,
    retrain_eliminability,
)
from .trainer import TrainLog, train, stability_sigma
from .ordering import OrderingDefect, ordering_defect
from .stability import stability_eliminability

__version__ = "1.0.0"

__all__ = [
    "MLP",
    "PINN",
    "ResidualSpec",
    "TermContribution",
    "EliminabilityResult",
    "eliminability",
    "eliminability_trajectory",
    "retrain_eliminability",
    "TrainLog",
    "train",
    "stability_sigma",
    "OrderingDefect",
    "ordering_defect",
    "stability_eliminability",
    "__version__",
]
