"""Run one seed of one benchmark and append its record to a per-phase JSONL
file. Lets expensive phases (Allen-Cahn, Navier-Stokes) be built up across
several invocations without losing work, then aggregated by aggregate.py."""

import sys
import json
import os
import time

sys.path.insert(0, "..")
sys.path.insert(0, ".")

from experiments.validation import run_seed

CONFIGS = {
    "burgers": dict(
        module="benchmarks.burgers",
        physics={"u_t", "adv", "diff"}, data={"ic", "bc"}, residual="adv",
        train_steps=700, retrain_steps=600, order_k=300, lr=2e-3,
        batch_kw=dict(n_pde=900, n_ic=110, n_bc=110),
    ),
    "allen_cahn": dict(
        module="benchmarks.allen_cahn",
        physics={"u_t", "diff", "react_cubic", "react_lin"}, data={"ic", "bc"},
        residual="react_cubic",
        train_steps=700, retrain_steps=600, order_k=300, lr=2e-3,
        batch_kw=dict(n_pde=900, n_ic=110, n_bc=110),
    ),
    "navier_stokes": dict(
        module="benchmarks.navier_stokes",
        physics={"adv_x", "adv_y", "press_x", "press_y", "visc_x", "visc_y", "cont"},
        data={"bc"}, residual="cont",
        train_steps=250, retrain_steps=200, order_k=120, lr=2e-3,
        batch_kw=dict(n_pde=400, n_bc=120), grouped=True,
    ),
}


def main(bench: str, seed: int):
    cfg = CONFIGS[bench]
    mod = __import__(cfg["module"], fromlist=["x"])
    t = time.time()
    rec = run_seed(
        mod, seed,
        physics_terms=cfg["physics"], data_terms=cfg["data"],
        residual_term=cfg["residual"],
        train_steps=cfg["train_steps"], retrain_steps=cfg["retrain_steps"],
        order_k=cfg["order_k"], lr=cfg["lr"], batch_kw=cfg["batch_kw"],
        grouped_retrain=cfg.get("grouped", False),
    )
    path = f"{bench}.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(rec.__dict__) + "\n")
    print(f"{bench} seed {seed}: loss={rec.final_loss:.3e} "
          f"retrain_top={rec.retrain_rank[:3]} order={rec.ordering_field:.2e} "
          f"({time.time()-t:.0f}s) -> {path}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
