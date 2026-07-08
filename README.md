# FRAME-PINN Reference Implementation

This repository accompanies the SoftwareX article *"FRAME-PINN: A Software
Framework for Structural Diagnostics in Physics-Informed Neural Networks."* It
provides a lightweight, reproducible implementation of the structural diagnostics
described in the paper, exercised on canonical PINN benchmarks. It is intended as
a research reference implementation rather than a full research platform.

## What is FRAME-PINN?

A Python toolkit for structural diagnostics in physics-informed neural networks
(PINNs). A PINN minimizes a composite loss assembled from several residual terms,
but the standard pipeline reports only aggregate quantities such as the total
loss. FRAME-PINN treats that loss as an explicit set of named, individually
addressable terms and measures how each one contributes to the trained solution.

It provides three diagnostics:

- **Eliminability** — how much a term contributes to the trained solution, in two
  variants. *Frozen* eliminability is the marginal contribution at fixed weights;
  *retrain* eliminability is structural necessity, measured by retraining without
  the term and scoring on the full set.
- **Ordering defect** — how much the schedule of data versus physics training
  changes the result, quantifying the failure of the two constraint families to
  commute as optimization operators.
- **Stability eliminability** — whether a term contributes to or relieves training
  instability.

FRAME-PINN is diagnostic instrumentation rather than a PDE solver. It attaches to
any PINN whose loss can be written as a sum of named terms.

## Installation

```bash
pip install -e .
```

Installation requires Python (>= 3.9) and PyTorch. To run the test suite,
`pip install -e ".[dev]"`.

## Verify it works

```bash
python examples/smoke_fast.py
```

This runs every diagnostic on all four benchmarks at a tiny budget in a few
seconds on a single CPU core, and prints `ALL OK` on success.

## A five-minute demo

```bash
python examples/burgers_demo.py
```

produces output of the form:

```
final training loss: 3.9e-01

Retrain eliminability (normalized, least eliminable first)
  ic       +11.457
  u_t       +7.981
  bc        +4.385
  adv       +1.861
  diff      -0.007

Ordering defect = 19.8   (ranking changed under reordering: True)
```

The initial condition and time-derivative terms are least eliminable, while the
diffusion term is eliminable at the benchmark's low viscosity. The ordering defect
is nonzero and reorders the ranking, indicating that the data and physics
constraints do not commute. Similar demos exist for Allen-Cahn and Navier-Stokes.

## Repository layout

```
src/frame_pinn/     the reusable library
  residual.py         ResidualSpec, TermContribution, PINN, MLP
  eliminability.py    frozen, retrain, trajectory operators
  ordering.py         ordering defect
  stability.py        stability eliminability
  trainer.py          snapshotting trainer + sigma metrics
  utils.py            autograd helpers
benchmarks/         Burgers, Allen-Cahn, Navier-Stokes, Poisson control
examples/           five-minute demos + install verification
experiments/        cross-seed validation harness, results, report
tests/              pytest suite
docs/               architecture, API, and theory notes
figures/            figures used in the paper
paper/              the SoftwareX manuscript source
```

## Reproducing the validation study

The `experiments/` directory contains the harness, the per-seed results in
`seed_results/`, and the machine-readable `validation_report.json` together with a
written `validation_report.md`. The reported rankings and statistics can be
inspected directly from the stored output. To recompute a single seed:

```bash
python experiments/run_one.py burgers 0
```

## Scope

This is the public reference implementation. It contains only the structural
diagnostics and the canonical benchmarks described in the paper. It is not the
authors' full internal research platform.

## License

MIT. See [LICENSE](LICENSE).
## Citation

If you use this software, please cite both the software and the accompanying
article; see [CITATION.cff](CITATION.cff).
