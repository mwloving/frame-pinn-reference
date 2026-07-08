"""Shared helpers.

Autograd utilities for building PDE residuals. Inputs carry ``requires_grad`` so
that spatial and temporal derivatives of the network output can be taken for the
physics residual terms.
"""

from __future__ import annotations

import torch


def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """d outputs / d inputs, same shape as inputs, with the graph retained so
    higher derivatives can be taken."""
    g = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]
    return g


def column(x: torch.Tensor, j: int) -> torch.Tensor:
    """Return column j of a 2D tensor as a column vector."""
    return x[:, j : j + 1]
