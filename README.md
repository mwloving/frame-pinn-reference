# FRAME-PINN Reference

**FRAME-PINN Reference** is a lightweight, reproducible reference implementation demonstrating the **FRAME Eliminability diagnostic** for Physics-Informed Neural Networks (PINNs). Rather than introducing a new PINN architecture, the repository provides diagnostic tools for evaluating the structural importance of PDE terms, boundary conditions, initial conditions, and residual components within representative PINN benchmark problems.

---

## Quick Start

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/mwloving/frame-pinn-reference.git
cd frame-pinn-reference
pip install -r requirements.txt
```

---

## Purpose

This repository accompanies the FRAME-PINN methodology and is intended as a reproducible research reference. It demonstrates how eliminability-based diagnostics can be used to:

- Identify structurally important PDE terms and constraints.
- Measure ranking stability across independent training seeds.
- Detect ordering defects in learned physics.
- Compare structural behavior across multiple PDE benchmarks.
- Generate reproducible diagnostic reports and validation artifacts.

The emphasis is on **structural diagnostics**, not maximizing numerical accuracy or providing a production PINN framework.

---

## What This Repository Demonstrates

FRAME-PINN provides a reproducible framework for evaluating the structural behavior of Physics-Informed Neural Networks using the FRAME Eliminability diagnostic. Rather than focusing solely on prediction accuracy, the repository illustrates how eliminability, ordering stability, and structural diagnostics can be used to analyze the contribution of individual PDE terms and constraints across representative benchmark problems.

---

## Included Examples

The reference implementation includes lightweight benchmark demonstrations for:

- Burgers equation
- Allen–Cahn equation
- 2D Navier–Stokes cavity flow
- Fast smoke test

These examples illustrate how the FRAME Eliminability diagnostic can be applied consistently across multiple classes of physics-informed learning problems.

---

## Repository Structure

```text
README.md
requirements.txt
LICENSE
CITATION.cff

eliminability.py
ordering.py
residual.py
stability.py
trainer.py
utils.py

burgers.py
allen_cahn.py
navier_stokes.py
poisson_control.py

burgers_demo.py
allen_cahn_demo.py
navier_stokes_demo.py
smoke_fast.py

aggregate.py
validation.py

test_benchmarks.py
test_eliminability.py
test_ordering.py
test_smoke.py

paper.md
paper.bib

architecture.md
api.md
theory.md
```

---

## Repository Status

**FRAME-PINN Reference v1.0.0**

This repository provides the reference implementation accompanying the FRAME-PINN methodology and is intended for reproducible evaluation of eliminability, ordering defects, and structural diagnostics in representative PINN benchmark problems.

---

## Citation

If you use this repository in research, please cite the accompanying publication and the repository's `CITATION.cff` file.

Related publications:

- **FRAME-PINN** (SoftwareX submission / preprint)
- **Evaluation Dependence in Quantum Measurement: A Minimal FRAME Formalism and Eliminability Discriminant**, *International Journal of Quantum Foundations*

---

## License

Released under the **MIT License**. See the `LICENSE` file for details.
