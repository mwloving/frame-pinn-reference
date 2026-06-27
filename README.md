# FRAME-PINN Reference

Toy-model reference implementation for applying the FRAME Eliminability diagnostic to Physics-Informed Neural Networks (PINNs).

This repository is intended as a lightweight, reproducible demonstration of how eliminability can be used to evaluate the structural importance of PDE terms, constraints, and residual components in PINN-style models.

## Purpose

The goal is not to provide a production PINN solver. The goal is to demonstrate a governance diagnostic:

* identify which terms are structurally important,
* test whether term rankings are stable across seeds,
* detect ordering defects,
* compare behavior across PDE examples,
* produce reproducible JSON/JSONL diagnostic outputs.

## Included Examples

Planned or included toy examples:

* Burgers equation
* Allen-Cahn equation
* 2D Navier-Stokes toy cavity setup
* fast smoke test

## Repository Status

Initial public reference implementation.

## License

MIT
