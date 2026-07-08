# Theory notes

This is a brief operational summary. For motivation, validation, and discussion,
see the accompanying SoftwareX article in [`paper/paper.md`](../paper/paper.md).

## Eliminability

Let `S` be the set of named residual terms in a PINN loss and `L(S)` the loss over
that set. The eliminability of term `i` is

```
E_i = L(S \ i) - L(S).
```

Two variants answer different questions:

- **Frozen eliminability** evaluates both losses on the same converged weights. It
  is a marginal measure: how much of the residual budget term `i` currently
  occupies. It is cheap and useful along a training trajectory, but at convergence
  it can report a genuinely necessary term as eliminable, because the trained
  solution already satisfies that term.
- **Retrain eliminability** trains a fresh network on the reduced set `S \ i` and
  scores it on the full set `S`. It measures structural necessity: whether a model
  that never saw term `i` can still reproduce the full physics. This is the
  operator the validation study relies on.

A normalized form divides by `L(S)` so that scores are comparable across problems
with different residual scales. Terms may also be removed by group, giving
operator-level eliminability.

## Ordering defect

With `A` optimizing only the data loss and `B` only the physics loss, the ordering
defect compares the two training orders from a shared initialization:

```
Lambda_O = || A(B(x)) - B(A(x)) ||.
```

A large value means the two constraint families do not commute as optimization
operators, so the training schedule is a structural variable.

## Sign convention

The Poisson control benchmark (`benchmarks/poisson_control.py`) has a unique
solution only when the boundary condition is present. Removing the boundary term
and retraining therefore yields a model that cannot reproduce the solution, so the
boundary carries a large positive retrain eliminability. This is the clearest
worked check that "least eliminable" aligns with "physically necessary."

These definitions are operational. The software makes no theoretical claims about
them beyond what the validation study demonstrates empirically.
