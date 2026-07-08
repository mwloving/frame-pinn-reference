---
title: "FRAME-PINN: A Software Framework for Structural Diagnostics in Physics-Informed Neural Networks"
journal: "SoftwareX"
author: "Matthew W. Loving"
affiliation: "ORS Quantum LLC, United States"
version: "v1.0"
date: "June 2026"
---

# Abstract

FRAME-PINN is an open-source Python package that measures the structural
contribution of individual physical constraints inside a physics-informed neural
network (PINN). Whereas existing PINN frameworks report aggregate quantities such
as the total loss and the final residual norm, FRAME-PINN treats the loss as an
explicit set of named, individually addressable terms and provides three
diagnostics over that set: eliminability, which quantifies how much each term
contributes to the trained solution; an ordering defect, which quantifies how
much the schedule of data versus physics training changes the result; and
residual decomposition, which exposes each primitive or grouped operator as a
separate measurable object. Across three canonical PDE families, Burgers,
Allen-Cahn, and a reduced lid-driven cavity Navier-Stokes system, the package
produces eliminability rankings that are stable across random seeds and that
recover known physics, and it detects a measurable ordering defect in every case.
FRAME-PINN is diagnostic instrumentation rather than a solver, and it attaches to
any PINN whose loss can be written as a sum of named terms.

# 1. Motivation and significance

Physics-informed neural networks have become a widely used approach for solving
forward and inverse PDE problems [@raissi2019pinns], and mature software such as
DeepXDE provides general infrastructure for defining and training them
[@lu2021deepxde]. These frameworks expose aggregate losses, convergence
histories, and PDE residuals, which describe how well a network fits a system as
a whole. They offer limited support for a different and increasingly important
question: how the individual physical constraints inside the loss contribute to
the learned solution. A practitioner can see that a PINN has converged to a given
residual norm, but not which of its constraints the solution actually depends on,
which terms are redundant, or how sensitive the result is to the order in which
data and physics constraints are applied during training.

These internal effects are not incidental. The literature on PINN training
pathologies has traced failure modes directly to how individual loss components
interact during optimization [@krishnapriyan2021failure]. Reported problems such
as stiffness imbalance between loss components, gradient conflicts across terms,
and the dominance of one constraint over others are, at root, questions about the
relative contribution of individual terms. A practitioner facing an unstable or
inaccurate PINN currently has few tools to ask which term is responsible, and
existing diagnostics that report a single scalar loss cannot answer that
question because the structure has already been collapsed into a sum.

FRAME-PINN addresses this gap with a focused toolkit for structural diagnosis. It
makes no claim to improve PINN accuracy or to resolve any open problem in
scientific machine learning. Its contribution is a reusable instrument that makes
residual importance, structural necessity, and training-order effects observable
and reproducible. The software is intended for researchers investigating PINN
failure modes, practitioners studying residual sensitivity and constraint
importance, and method developers building new diagnostic or model-reduction
approaches. The eliminability operator that underpins the package is a general
construct for measuring the contribution of a term to a learned solution; the
present work provides a reference implementation specialized to physics-informed
neural networks, with reproducible examples and a validation study demonstrating
that the diagnostics recover physically meaningful structure.

# 2. Software description

## 2.1 Software architecture

The guiding principle of the design is that the unit of analysis is the
individual residual term. A PINN problem is expressed as a decomposed residual
specification, which feeds the eliminability operators, the ordering diagnostics,
and a reproducible validation layer, as shown in Figure 1. The core abstraction
is `ResidualSpec`, a declarative description of a PINN loss as a list of named
`TermContribution` objects, each carrying a name, a group label such as
advective, pressure, viscous, reaction, or data, and a closure that returns a
per-point residual tensor given the network and a batch.

![Figure 1: FRAME-PINN architecture. A decomposed residual specification feeds the eliminability operators, the ordering and stability diagnostics, and a reproducible validation layer.](fig1_architecture.png)

Representing terms as closures rather than precomputed tensors is the key design
choice. A closure recomputes its residual on demand from the current network
state, so a term definition stays valid after every weight update and across the
repeated retraining that the diagnostics require, which lets any term be added to
or removed from the loss without rebuilding the model. The `PINN` wrapper
assembles these terms into a scalar loss while honoring an active set: removing
element $i$ from the active set realizes $L(S \setminus i)$ directly. Grouping
allows a whole operator to be removed at once, which yields operator-level
diagnostics for systems such as the incompressible Navier-Stokes equations.

The remaining modules build on this abstraction. The module `eliminability.py`
implements the frozen, trajectory, and retrain operators with ranking and pruning
helpers. The module `structural.py` implements the ordering defect and a
stability diagnostic. The module `trainer.py` provides a training loop with the
per-term logging and weight snapshots the operators consume. The `benchmarks/`
and `examples/` directories supply the PDE problems and runnable entry points,
and the `experiments/` directory contains the validation harness. This separation
keeps the diagnostic logic independent of any particular PDE, so a user attaches
the diagnostics to their own model simply by supplying a `ResidualSpec`.

## 2.2 Eliminability

Let $S$ denote the set of named residual terms in a PINN loss and let $L(S)$ be
the loss evaluated over that set. The eliminability of term $i$ is the change in
loss when that term is removed:

$$E_i = L(S \setminus i) - L(S).$$

The package implements two variants that answer different questions. Frozen
eliminability evaluates both losses on the same converged weights, measuring the
marginal residual budget that term $i$ currently occupies; it is inexpensive and
suitable for tracking a term along a training trajectory. Retrain eliminability
instead trains a fresh network on the reduced set $S \setminus i$ and scores it on
the full set $S$, measuring whether a model that never saw term $i$ can still
reproduce the full physics. The distinction matters because at convergence a
genuinely necessary term can have near-zero frozen eliminability, simply because
the trained solution already satisfies it; the retrain operator avoids this by
asking a counterfactual question about training rather than a marginal question
about the converged state. A normalized form divides each difference by $L(S)$,
expressing eliminability as a fraction of the baseline loss so that scores remain
comparable across problems with different residual scales.

Figure 2 illustrates the retrain operator on a Burgers model. Removing the
diffusion term and retraining leaves the final solution almost unchanged, while
removing the boundary term causes a large degradation. The resulting
eliminability scores separate a structurally necessary constraint from a
comparatively redundant one, a distinction invisible in the aggregate loss yet
decisive when deciding which terms to weight, prune, or scrutinize.

![Figure 2: Retrain eliminability. A term is removed, a fresh network is trained without it, and the result is scored on the full set. Necessary terms produce large scores, eliminable terms produce near-zero scores.](fig2_eliminability.png)

## 2.3 Ordering defect

The package also measures an ordering defect that quantifies sensitivity to
training schedule. With $A$ optimizing only the data loss and $B$ only the physics
loss, the defect compares applying $A$ then $B$ against $B$ then $A$ from a shared
initialization:

$$\Lambda_O = \lVert A(B(x)) - B(A(x)) \rVert.$$

If a model trained on data and then physics reaches a substantially different
solution than one trained in the opposite order, $\Lambda_O$ captures this
training-path sensitivity as a single number, quantifying how far the data and
physics constraint families fail to commute as optimization operators and so
making the training schedule itself a structural variable. The definitions in
this section are operational, and the package makes no theoretical claims about
them.

## 2.4 Software functionalities

In summary, FRAME-PINN provides: residual decomposition into named, grouped, and
individually toggleable terms; frozen and retrain eliminability with normalized
and grouped variants; an eliminability trajectory operator that tracks a term
across training; an ordering defect over solution fields, residuals, and the
eliminability profile; a stability diagnostic; a snapshotting trainer; three
worked PDE benchmarks; and a cross-seed validation harness that emits
machine-readable results.

# 3. Illustrative examples and validation

The package includes three reproducible benchmarks, summarized in Table 1, each
defined as a `ResidualSpec` with its physics residual decomposed into primitive
terms. The three were chosen to span the axes along which the diagnostics are
most likely to behave differently, namely nonlinearity, stiffness, and operator
coupling, so that consistent behavior across them is evidence of generality
rather than of a single favorable case.

**Table 1.** Benchmark characteristics.

| Benchmark | Type | Decomposed terms | Diagnostic axis |
|-----------|------|------------------|-----------------|
| Burgers | scalar nonlinear transport | time, advection, diffusion, IC, BC | nonlinearity |
| Allen-Cahn | stiff phase-field | time, diffusion, cubic and linear reaction, IC, BC | stiffness |
| Navier-Stokes (cavity) | coupled vector-valued | advective, pressure, viscous, continuity, data (grouped) | operator coupling |

## 3.1 Validation methodology

The validation study treats FRAME-PINN as an instrument to be characterized
rather than a network to be optimized. The network is not tuned and
hyperparameters are not adjusted; every benchmark runs on its shipped defaults at
fixed step budgets. For each benchmark we run three random seeds. Per seed we
train a baseline PINN, then compute frozen eliminability, retrain eliminability,
and the ordering defect. Across seeds we compute, per term, the mean and standard
deviation of normalized eliminability, the modal rank-1 term, the fraction of
seeds agreeing on it, and the mean pairwise Kendall tau of the full ranking.
Navier-Stokes uses grouped, operator-level eliminability, which is the natural
object for that system and is also the cheaper one.

A term is judged structurally important when its mean retrain eliminability is
positive, its variance is modest relative to its mean, and its ranking position is
consistent across seeds. The full set of success criteria is given in Table 2.
Each is paired with a complementary failure condition, so that the study can fail
as well as pass; a method that produced random rankings, wild seed-to-seed
variation, eliminable dominant terms, or a negligible ordering defect would be
recorded as a failure.

The diagnostics are lightweight. Table 2 also reports the wall-clock time for a
full diagnostic pass at the install-verification budget, namely a complete run of
the trainer, both eliminability operators, the ordering defect, and the stability
metric on each benchmark, measured on a single CPU core with no GPU. All three
complete in under two seconds, which is what makes continuous integration and
reviewer reproduction inexpensive; the full multi-seed validation study runs at a
larger budget on the order of one to three minutes per seed on the same hardware.

**Table 2a.** Runtime of a full diagnostic pass at the install-verification budget (single CPU core, no GPU).

| Benchmark | Runtime | Hardware |
|-----------|--------:|----------|
| Burgers | 0.2 s | 1 CPU core |
| Allen-Cahn | 0.3 s | 1 CPU core |
| Navier-Stokes | 1.2 s | 1 CPU core |

**Table 2.** Validation success criteria. Each is paired with a failure condition.

| Criterion | Pass condition | Failure condition |
|-----------|----------------|-------------------|
| Ranking stability | Kendall tau and top-1 agreement high across seeds | rankings vary wildly between seeds |
| Physical correctness | rankings match known physics | physically dominant terms appear eliminable |
| Dominance | dominant terms have positive eliminability | dominant terms eliminable |
| Ordering sensitivity | ordering defect nonzero | ordering defect negligible |
| Regime dependence | expected-eliminable terms rank low | rankings independent of regime |

## 3.2 Results

Across all three benchmarks and all seeds, the success criteria were met, for a
total of fifteen of fifteen checks passing. The cross-seed retrain eliminability
statistics are summarized in Table 3 and Figure 3, the ordering defects in Figure
4, and the criteria outcomes in Figure 5.

**Table 3.** Cross-seed retrain eliminability (normalized, mean over three seeds, standard deviation in parentheses) and ranking consistency.

| Benchmark | Most necessary term (mean, s.d.) | Most eliminable term (mean, s.d.) | Kendall tau | Top-1 agreement |
|-----------|----------------------------------|-----------------------------------|-------------|-----------------|
| Burgers | ic (+11.46, 0.29) | diff (-0.01, 0.01) | 1.00 | 1.00 |
| Allen-Cahn | bc (+5.33, 0.82) | diff (-0.23, 0.08) | 0.82 | 1.00 |
| Navier-Stokes | data (+3.38, 1.11) | advective (-0.41, 0.16) | 0.73 | 1.00 |

On Burgers, the retrain ranking was identical in all three seeds, with the
initial condition and the time derivative the least eliminable terms and
diffusion the most eliminable. The diffusion result is the expected physics: at
the benchmark viscosity the advection term dominates, and a model trained without
diffusion reproduces the solution almost exactly, so its eliminability is
statistically indistinguishable from zero. This is regime detection emerging
directly from the diagnostic.

![Figure 3: Cross-seed retrain eliminability for each benchmark, mean over three seeds with standard-deviation error bars. Terms are ordered from most to least eliminable.](fig3_cross_seed_rankings.png)

On Allen-Cahn, the boundary condition was the dominant term and was rank-1 in
every seed, with the remaining terms reordering only among much smaller values
and the tiny-coefficient diffusion term correctly the most eliminable. On the
grouped Navier-Stokes problem, the data group dominated and was rank-1 in every
seed, with pressure and continuity the leading physics operators and the
advective group the most eliminable at the moderate Reynolds number of the
benchmark, consistent with a viscous-pressure-dominated cavity flow.

The ordering defect was large and nonzero in every benchmark and every seed,
ranging from roughly 17 to 38 in solution-field norm, as shown in Figure 4, and
it reordered the eliminability ranking in all nine runs. This indicates that the
data and physics constraint families do not commute as optimization operators for
any of the three systems, so the training schedule is a genuine structural
variable rather than an implementation detail.

![Figure 4: Ordering defect for each benchmark and seed. A nonzero defect is detected in every run.](fig4_ordering_defect.png)

Figure 5 summarizes the criteria outcomes. No failure condition was triggered in
any phase: rankings were not random, did not vary wildly between seeds, no
physically dominant term appeared eliminable, the ordering defect was never
negligible, and retrain eliminability reproduced across seeds.

![Figure 5: Validation criteria outcomes. All fifteen checks across three benchmarks pass.](fig5_validation_summary.png)

# 4. Impact and discussion

The results support a narrow but useful claim: eliminability, as implemented in
FRAME-PINN, recovers structure that is physically meaningful, stable across
seeds, and dependent on regime. The separation of frozen and retrain
eliminability is central to that claim. The frozen operator is inexpensive and
informative along a trajectory, but at convergence it can report a genuinely
necessary term as eliminable, because the trained solution already satisfies that
term. The retrain operator measures structural necessity directly by asking
whether a model that never saw a term can still reproduce the full physics, and
it is the retrain rankings that proved stable and physically correct. A control
experiment on a Poisson problem with a known solution confirmed the sign
convention: removing the boundary condition and retraining produced a model that
scored catastrophically on the full set, correctly marking the boundary as the
least eliminable term.

The ordering defect contributes a second, complementary signal. Because it was
large in every benchmark and reordered the eliminability ranking, it shows that
path dependence in PINN training is structural rather than merely a different
local minimum. For practitioners this suggests that training schedule should be
treated as a reportable variable, much as learning rate or loss weighting are,
and FRAME-PINN provides a concrete number for it. Grouped eliminability extends
the same idea to operator level, which is where questions about Navier-Stokes
regimes are naturally posed, and the grouped results behaved as the frozen and
ungrouped results did.

It is worth situating eliminability relative to existing notions of term
importance. Gradient-based saliency measures the local sensitivity of the loss to
inputs or parameters at the current weights; it is cheap and widely used, but it
reports a marginal quantity at a fixed point and can assign a small value to a
term that is nonetheless indispensable to the solution, and gradient-based
importance is known to be sensitive to noise and seed. Classical ablation, in
which a term is dropped and the already-trained model is re-evaluated, is closely
related to FRAME-PINN's frozen operator and shares its limitation: it measures
marginal contribution at convergence rather than structural necessity, because the
trained weights already encode the dropped constraint. Retrain eliminability is
the distinguishing element. By training a fresh model on the reduced set and
scoring it on the full set, it asks whether the physics can be reconstructed at
all without a given term, which is a counterfactual about training rather than a
perturbation of a converged state. Table 4 contrasts these approaches along three
axes that matter for the present use case. The point is not that gradient saliency
or ablation are unsound, but that they answer the marginal question, whereas the
structural-necessity question requires retraining, and FRAME-PINN measures the
seed stability of its answer rather than assuming it.

**Table 4.** Term-importance approaches contrasted along three axes relevant to structural diagnosis.

| Method | Measures structural necessity | Requires retraining | Seed stability |
|--------|-------------------------------|---------------------|----------------|
| Gradient norms / saliency | no (marginal, at fixed weights) | no | variable, often noisy |
| Ablation (frozen re-evaluation) | partial (marginal at convergence) | no | typically uncharacterized |
| FRAME-PINN frozen eliminability | partial (marginal, trajectory-aware) | no | measured |
| FRAME-PINN retrain eliminability | yes (counterfactual on training) | yes | measured (Kendall tau, top-1 agreement) |

Two limitations bound these conclusions. First, the baseline PINNs do not fully
converge on Burgers or Allen-Cahn, exhibiting the known vanilla-PINN stiffness
and shock failure modes; in keeping with the diagnostic mandate, the networks
were not tuned around this. The positive result therefore rests specifically on
the retrain operator, which is robust to an imperfect baseline because it measures
whether a term is needed to reach whatever loss is reachable, whereas frozen
eliminability on the unconverged weights is not a reliable structural measure.
Second, the study used small step budgets and three seeds to keep it inexpensive;
the rankings are already stable at this budget, but tighter variance estimates
would benefit from more seeds and from averaging the retrain operator over several
initializations, both of which the harness already supports.

Within these bounds, the software has immediate practical uses. A practitioner can
rank the terms of a struggling PINN to identify which constraint dominates, prune
terms with low retrain eliminability to build a reduced model, track a term across
training to locate structurally unstable phases, and report an ordering defect
when training schedule is in question. Because the diagnostics attach to any loss
expressed as a sum of named terms, none of these uses is tied to the bundled
benchmarks.

Planned directions for subsequent releases fall into three groups: broader PDE
coverage, extending the suite to additional systems and higher dimensions;
diagnostic expansion, adding further structural measures alongside eliminability
and the ordering defect; and uncertainty quantification over ensembles, a natural
extension since the per-seed statistics already produced by the validation harness
anticipate distributional summaries. These remain outside the scope of v1.0, which
is deliberately limited to a compact, well-tested set of diagnostics that install,
run, and reproduce without specialized hardware.

# 5. Conclusions

FRAME-PINN is a reference implementation of structural diagnostics for
physics-informed neural networks. It measures the contribution of individual
physical constraints through frozen and retrain eliminability, quantifies
training-order sensitivity through an ordering defect, and exposes residual terms
as individually addressable objects. Across Burgers, Allen-Cahn, and a reduced
Navier-Stokes cavity, the diagnostics recover physically correct, seed-stable,
regime-dependent structure, and the ordering defect is a consistently measurable
signal. The package is open-source, tested, and reproducible on commodity
hardware, and it applies to any PINN whose loss can be written as a sum of named
terms.

# Software availability

FRAME-PINN v1.0 is released as open-source software under the MIT license. The
source code, documentation, and examples are available in the project's GitHub
repository at https://github.com/mwloving/frame-pinn-reference, with the v1.0.0
release tagged at https://github.com/mwloving/frame-pinn-reference/releases/tag/v1.0.0
and archived with a persistent identifier (DOI: [to be assigned on archival]). The
package installs and runs its fastest verification with two commands:

```
pip install -e .
python examples/smoke_fast.py
```

Installation requires only Python and PyTorch [@paszke2019pytorch]. The validation
harness under `experiments/` reproduces the results in this article and writes
machine-readable JSON summaries, so reported rankings and statistics can be
inspected without rerunning the training.

# References
