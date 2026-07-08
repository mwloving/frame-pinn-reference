"""Five-minute Burgers demo.

Trains a small PINN on the viscous Burgers equation, then prints the retrain
eliminability ranking and the ordering defect. Expected qualitative result: the
initial condition and time-derivative terms are least eliminable, while the
diffusion term is eliminable at the benchmark's low viscosity.

    python examples/burgers_demo.py
"""

import torch

from frame_pinn import train, retrain_eliminability, ordering_defect
from benchmarks import burgers


def main(seed: int = 0, steps: int = 800):
    torch.manual_seed(seed)
    bf = burgers.batch_fn(n_pde=1000, n_ic=120, n_bc=120)

    model = burgers.build_model()
    log = train(model, bf, steps=steps, lr=2e-3, log_every=100)
    print(f"final training loss: {log.total_loss[-1]:.3e}\n")

    er = retrain_eliminability(burgers.build_model, bf, train,
                               steps=steps, lr=2e-3, repeats=1)
    print("Retrain eliminability (normalized, least eliminable first)")
    for name, val in er.ranking():
        print(f"  {name:6s} {val:+8.3f}")

    od = ordering_defect(burgers.build_model, bf,
                         data_terms={"ic", "bc"},
                         physics_terms={"u_t", "adv", "diff"},
                         probe=burgers.probe_grid(),
                         residual_term="adv", k=400, lr=2e-3, seed=seed)
    print(f"\nOrdering defect = {od.field_defect:.1f}"
          f"   (ranking changed under reordering: {od.rank_changed})")


if __name__ == "__main__":
    main()
