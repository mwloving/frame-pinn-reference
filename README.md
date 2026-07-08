# FRAME-PINN Reference

**FRAME-PINN Reference** is a lightweight, reproducible software framework for evaluating the structural behavior of Physics-Informed Neural Networks (PINNs) using the **FRAME Eliminability diagnostic**. Rather than introducing a new PINN architecture, it provides reproducible diagnostic tools for evaluating the structural importance of PDE terms, boundary conditions, initial conditions, and residual components across representative benchmark problems.

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

This repository accompanies the FRAME-PINN methodology and serves as a reproducible research reference implementation. It demonstrates how eliminability-based diagnostics can be used to:

- Identify structurally important PDE terms and constraints.
- Measure ranking stability across independent training seeds.
- Detect ordering defects in learned physics.
- Compare structural behavior across multiple benchmark PDEs.
- Generate reproducible diagnostic reports and validation artifacts.

The emphasis is on **structural diagnostics**, reproducibility, and scientific interpretation rather than maximizing numerical accuracy or providing a production PINN framework.

---

## Key Capabilities

FRAME-PINN provides reproducible structural diagnostics for representative Physics-Informed Neural Network benchmark problems. The reference implementation demonstrates how to:

- Evaluate the eliminability of PDE terms and constraints.
- Measure cross-seed ranking stability.
- Detect ordering defects during training.
- Compare structural behavior across multiple benchmark equations.
- Produce reproducible validation reports suitable for scientific comparison.

The repository is intended to illustrate how structural diagnostics can complement conventional PINN evaluation metrics by providing insight into the relative importance and interaction of governing physics terms.

---

## Included Examples

The reference implementation includes lightweight benchmark demonstrations for:

- Burgers equation
- Allen–Cahn equation
- 2D Navier–Stokes cavity flow
- Poisson control example
- Fast smoke test

These examples demonstrate application of the FRAME Eliminability diagnostic across representative classes of physics-informed learning problems.

---

## Repository Structure

```text
README.md
requirements.txt
LICENSE
CITATION.cff

Core diagnostics
├── eliminability.py
├── ordering.py
├── residual.py
├── stability.py
├── trainer.py
└── utils.py

Benchmark problems
├── burgers.py
├── allen_cahn.py
├── navier_stokes.py
└── poisson_control.py

Example applications
├── burgers_demo.py
├── allen_cahn_demo.py
├── navier_stokes_demo.py
└── smoke_fast.py

Validation
├── aggregate.py
├── validation.py
├── validation_report.md
├── validation_report.json
└── validation_summary.png

Tests
├── test_benchmarks.py
├── test_eliminability.py
├── test_ordering.py
└── test_smoke.py

Documentation
├── api.md
├── architecture.md
├── theory.md
├── architecture.png
├── eliminability_example.png
├── ordering_defect.png
└── cross_seed_rankings.png

Paper
├── paper.md
└── paper.bib
```

---

## Repository Status

**FRAME-PINN Reference v1.0.0**

This repository provides the reference implementation accompanying the FRAME-PINN methodology and is intended for reproducible evaluation of eliminability, ordering defects, and structural diagnostics in representative Physics-Informed Neural Network benchmark problems.

The implementation is designed as a lightweight research reference suitable for validation, experimentation, and independent reproduction of the accompanying methodology.

---

## Citation

If this repository contributes to your research, please cite the accompanying publication and the repository's `CITATION.cff` file.

### Related Publications

- **FRAME-PINN: A Software Framework for Structural Diagnostics in Physics-Informed Neural Networks** *(SoftwareX submission / preprint)*
- **Evaluation Dependence in Quantum Measurement: A Minimal FRAME Formalism and Eliminability Discriminant**, *International Journal of Quantum Foundations*.
- **Transport Constrained Emergent Geometry: An Operational Transport Mechanism for Effective Geometry from Relational Structure**, *International Journal of Quantum Foundations*.

---

## License

Released under the **MIT License**. See the `LICENSE` file for details.
