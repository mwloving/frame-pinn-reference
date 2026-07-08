"""Five-minute Navier-Stokes (lid-driven cavity) demo.

Trains a small PINN on the reduced steady cavity problem and prints the grouped,
operator-level retrain eliminability ranking and the ordering defect. Grouped
eliminability is the natural object for this system: it ranks whole operators
(advective, pressure, viscous, continuity, data) rather than individual scalar
residuals.

    python examples/navier_stokes_demo.py
"""

import torch

from frame_pinn import train, retrain_eliminability, ordering_defect
from benchmarks import navier_stokes as ns


def main(seed: int = 0, steps: int = 250):
    torch.manual_seed(seed)
    bf = ns.batch_fn(n_pde=400, n_bc=120)

    model = ns.build_model()
    log = train(model, bf, steps=steps, lr=2e-3, log_every=50)
    print(f"final training loss: {log.total_loss[-1]:.3e}\n")

    er = retrain_eliminability(ns.build_model, bf, train,
                               steps=steps, lr=2e-3, repeats=1, grouped=True)
    print("Grouped retrain eliminability (normalized, least eliminable first)")
    for name, val in er.ranking():
        print(f"  {name:12s} {val:+8.3f}")

    od = ordering_defect(
        ns.build_model, bf,
        data_terms={"bc"},
        physics_terms={"adv_x", "adv_y", "press_x", "press_y",
                       "visc_x", "visc_y", "cont"},
        probe=ns.probe_grid(),
        residual_term="cont", k=120, lr=2e-3, seed=seed)
    print(f"\nOrdering defect = {od.field_defect:.1f}"
          f"   (ranking changed under reordering: {od.rank_changed})")


if __name__ == "__main__":
    main()
