"""Five-minute Allen-Cahn demo.

Trains a small PINN on the Allen-Cahn equation and prints the retrain
eliminability ranking and the ordering defect. Expected qualitative result: the
boundary condition is the least eliminable term, and the tiny-coefficient
diffusion term is the most eliminable.

    python examples/allen_cahn_demo.py
"""

import torch

from frame_pinn import train, retrain_eliminability, ordering_defect
from benchmarks import allen_cahn


def main(seed: int = 0, steps: int = 700):
    torch.manual_seed(seed)
    bf = allen_cahn.batch_fn(n_pde=1000, n_ic=120, n_bc=120)

    model = allen_cahn.build_model()
    log = train(model, bf, steps=steps, lr=2e-3, log_every=100)
    print(f"final training loss: {log.total_loss[-1]:.3e}\n")

    er = retrain_eliminability(allen_cahn.build_model, bf, train,
                               steps=steps, lr=2e-3, repeats=1)
    print("Retrain eliminability (normalized, least eliminable first)")
    for name, val in er.ranking():
        print(f"  {name:12s} {val:+8.3f}")

    od = ordering_defect(allen_cahn.build_model, bf,
                         data_terms={"ic", "bc"},
                         physics_terms={"u_t", "diff", "react_cubic", "react_lin"},
                         probe=allen_cahn.probe_grid(),
                         residual_term="react_cubic", k=350, lr=2e-3, seed=seed)
    print(f"\nOrdering defect = {od.field_defect:.1f}"
          f"   (ranking changed under reordering: {od.rank_changed})")


if __name__ == "__main__":
    main()
