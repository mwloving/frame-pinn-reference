# API reference

All public symbols are importable from the top-level `frame_pinn` package.

## Building a problem

### `ResidualSpec`

A set of named residual terms.

- `add(name, group, fn) -> ResidualSpec` — append a term; `fn(net, batch)` returns
  a per-point residual tensor.
- `names() -> list[str]` — term names in order.
- `groups() -> list[str]` — distinct group labels in order.

### `TermContribution`

Dataclass holding a term's `name`, `group`, and residual `fn`.

### `PINN`

Wraps an `MLP` and a `ResidualSpec`.

- `loss_dict(batch) -> dict[str, Tensor]` — mean-squared loss per term.
- `total_loss(batch, active=None) -> Tensor` — weighted sum over the active set;
  passing `active = S \ {i}` realizes the eliminability removal.

### `MLP`

A plain tanh multilayer perceptron with Xavier initialization.

## Training

### `train(model, batch_fn, steps, lr, active=None, snapshot_every=None, log_every=50) -> TrainLog`

Trains on the active set (the full set when `active=None`). `batch_fn()` returns a
fresh batch each call, which permits collocation resampling. Returns a `TrainLog`
with per-term losses, gradient norms, and optional weight snapshots.

### `stability_sigma(log, window=20, spike_factor=3.0) -> dict`

Instability metrics over the tail of a run: `loss_var` and `spike_rate`.

## Eliminability

### `eliminability(model, batch, grouped=False) -> EliminabilityResult`

Frozen eliminability `E_i = L(S \ i) - L(S)` on the current weights. With
`grouped=True`, removes whole operator groups.

### `retrain_eliminability(build_model, batch_fn, train_fn, steps, lr, repeats=1, grouped=False, eval_batches=4) -> EliminabilityResult`

Retrain eliminability: train a fresh model on the reduced set and score it on the
full set. `build_model()` returns a fresh `PINN`.

### `eliminability_trajectory(model, batch, snapshots, grouped=False) -> list[EliminabilityResult]`

Replays saved state dicts to build `E_i(theta_t)` along training.

### `EliminabilityResult`

- `raw`, `normalized` — per-term dictionaries.
- `base_loss` — `L(S)`.
- `ranking(normalized=True) -> list[(name, value)]` — least eliminable first.
- `eliminable(eps, normalized=True) -> list[str]` — terms below threshold.

## Ordering defect

### `ordering_defect(build_model, batch_fn, data_terms, physics_terms, probe, residual_term, k=1500, lr=1e-3, seed=0) -> OrderingDefect`

Compares data-then-physics against physics-then-data from a shared
initialization. Returns an `OrderingDefect` with `field_defect`,
`residual_defect`, `elim_l1`, `rank_changed`, and the two rankings.

## Stability

### `stability_eliminability(build_model, batch_fn, sigma_key="loss_var", steps=3000, lr=1e-3, repeats=3) -> dict`

For each term, compares an instability metric for the full run against runs with
the term removed. Positive means the term contributes to instability.
