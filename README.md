# FRAME-PINN Reference

**FRAME-PINN Reference** is a lightweight, reproducible reference implementation demonstrating the **FRAME Eliminability diagnostic** for Physics-Informed Neural Networks (PINNs). Rather than introducing a new PINN architecture, the repository provides diagnostic tools for evaluating the structural importance of PDE terms, boundary conditions, initial conditions, and residual components within representative PINN benchmark problems.

## Quick Start

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/mwloving/frame-pinn-reference.git
cd frame-pinn-reference
pip install -r requirements.txt
## Purpose

This repository accompanies the FRAME-PINN methodology and is intended as a reproducible research reference. It demonstrates how eliminability-based diagnostics can be used to:

* identify structurally important PDE terms and constraints,
* measure ranking stability across independent training seeds,
* detect ordering defects in learned physics,
* compare structural behavior across multiple PDE benchmarks,
* generate reproducible diagnostic reports and validation artifacts.

The emphasis is on **structural diagnostics**, not maximizing numerical accuracy or providing a production PINN framework.

## Included Examples

The reference implementation includes lightweight benchmark demonstrations for:

* Burgers equation
* Allen–Cahn equation
* 2D Navier–Stokes cavity flow
* Fast smoke test

These examples illustrate how the FRAME Eliminability diagnostic can be applied consistently across multiple classes of physics-informed learning problems.

## Repository Status

**FRAME-PINN Reference v1.0.0**

This repository provides the reference implementation accompanying the FRAME-PINN publication and is intended for reproducible evaluation of eliminability, ordering defects, and structural diagnostics in representative PINN benchmark problems.

## License

MIT
