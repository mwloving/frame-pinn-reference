# Architecture

FRAME-PINN is organized around a single principle: the unit of analysis is the
individual residual term, not the network or the loss as a whole.

## Pipeline

A PINN problem is expressed as a decomposed residual specification, which feeds
the eliminability operators, the ordering and stability diagnostics, and a
reproducible validation layer.

![architecture](../figures/architecture.png)

## Core abstraction: `ResidualSpec`

A `ResidualSpec` describes a PINN loss as a list of named `TermContribution`
objects. Each term carries:

- a **name** (for example `adv`, `diff`, `bc`),
- a **group** label (for example `advective`, `viscous`, `data`), and
- a **closure** that returns a per-point residual tensor given the network and a
  batch.

Representing terms as closures rather than precomputed tensors is the key design
choice. A closure recomputes its residual on demand from the current network
state, so a term definition stays valid after every weight update and across the
repeated retraining the diagnostics require. This is what lets any term be added
to or removed from the loss without rebuilding the model.

## `PINN` and the active set

The `PINN` wrapper assembles the terms into a scalar loss while honoring an
*active set*. Removing a term from the active set realizes the reduced loss
`L(S \ i)` directly, which is the mechanism underlying eliminability. Terms can
also be removed by group, which yields operator-level diagnostics for systems
such as the incompressible Navier-Stokes equations.

## Modules

| Module | Responsibility |
|--------|----------------|
| `residual.py` | `ResidualSpec`, `TermContribution`, `PINN`, `MLP` |
| `eliminability.py` | frozen, trajectory, and retrain eliminability |
| `ordering.py` | ordering defect |
| `stability.py` | stability eliminability |
| `trainer.py` | training loop, per-term logging, weight snapshots, sigma metrics |
| `utils.py` | autograd helpers for building PDE residuals |

The diagnostic logic is independent of any particular PDE. A user attaches the
diagnostics to a new model simply by supplying a `ResidualSpec`.
