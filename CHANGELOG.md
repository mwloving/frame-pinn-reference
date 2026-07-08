# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
semantic versioning.

## [1.0.0] - 2026-06-01

Initial public release accompanying the SoftwareX article *"FRAME-PINN: A
Software Framework for Structural Diagnostics in Physics-Informed Neural
Networks."*

### Added

- `ResidualSpec` and `TermContribution`: a PINN loss expressed as an explicit set
  of named, individually addressable residual terms.
- Frozen eliminability, retrain eliminability, normalized and grouped variants,
  and an eliminability trajectory operator.
- Ordering defect over solution fields, residuals, and the eliminability profile.
- Stability eliminability and a snapshotting trainer with per-term logging.
- Four benchmarks: viscous Burgers, Allen-Cahn, reduced lid-driven cavity
  Navier-Stokes, and a Poisson control problem demonstrating the sign convention.
- Five-minute example demos and an install-verification script.
- Cross-seed validation harness with machine-readable JSON output and a written
  validation report.
- Test suite covering imports, eliminability, the ordering defect, and benchmark
  construction.
