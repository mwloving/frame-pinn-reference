"""FRAME-PINN v1.0 experimental validation harness.

Mission: determine whether frozen eliminability, retrain eliminability, and
ordering defect produce stable, physically meaningful structure in PINNs. This
is diagnostic validation. The network is NOT optimized and hyperparameters are
NOT tuned; every benchmark uses its shipped defaults. The only knobs here are
seed count and the fixed step budgets needed to reach a usable operating point.

Output per phase: per-seed records and cross-seed mu/sigma per term, plus an
explicit pass/fail against the success and failure criteria.
"""

from __future__ import annotations

import json
import statistics as stats
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from frame_pinn import (
    train,
    eliminability,
    retrain_eliminability,
    ordering_defect,
)


@dataclass
class SeedRecord:
    seed: int
    final_loss: float
    pde_residual: float
    frozen: Dict[str, float]            # normalized frozen eliminability
    retrain: Dict[str, float]           # normalized retrain eliminability
    retrain_raw: Dict[str, float]
    frozen_rank: List[str]
    retrain_rank: List[str]
    ordering_field: float
    ordering_elim_l1: float
    ordering_rank_changed: bool
    ordering_rank_AB: List[str]
    ordering_rank_BA: List[str]


def _pde_residual(model, batch, physics_terms: set) -> float:
    """Mean PDE residual across the physics terms only (excludes data terms)."""
    ld = model.loss_dict(batch)
    vals = [float(ld[n].detach()) for n in ld if n in physics_terms]
    return float(sum(vals) / max(len(vals), 1))


def run_seed(
    mod,
    seed: int,
    physics_terms: set,
    data_terms: set,
    residual_term: str,
    train_steps: int,
    retrain_steps: int,
    order_k: int,
    lr: float,
    batch_kw: dict,
    grouped_retrain: bool = False,
) -> SeedRecord:
    torch.manual_seed(seed)
    bf = mod.batch_fn(**batch_kw)

    model = mod.build_model()
    log = train(model, bf, steps=train_steps, lr=lr, log_every=max(1, train_steps // 10))
    eval_batch = bf()
    final_loss = float(log.total_loss[-1])
    pde_res = _pde_residual(model, eval_batch, physics_terms)

    fe = eliminability(model, eval_batch, grouped=grouped_retrain)
    re = retrain_eliminability(
        mod.build_model, bf, train,
        steps=retrain_steps, lr=lr, repeats=1, eval_batches=2,
        grouped=grouped_retrain,
    )
    od = ordering_defect(
        mod.build_model, bf, data_terms, physics_terms,
        mod.probe_grid(), residual_term, k=order_k, lr=lr, seed=seed,
    )

    return SeedRecord(
        seed=seed,
        final_loss=final_loss,
        pde_residual=pde_res,
        frozen=dict(fe.normalized),
        retrain=dict(re.normalized),
        retrain_raw=dict(re.raw),
        frozen_rank=[n for n, _ in fe.ranking()],
        retrain_rank=[n for n, _ in re.ranking()],
        ordering_field=od.field_defect,
        ordering_elim_l1=od.elim_l1,
        ordering_rank_changed=od.rank_changed,
        ordering_rank_AB=od.ranking_AB,
        ordering_rank_BA=od.ranking_BA,
    )


# ---- cross-seed statistics -------------------------------------------------

def term_stats(records: List[SeedRecord], which: str) -> Dict[str, Dict[str, float]]:
    """mu and sigma of (normalized) eliminability per term across seeds.
    which in {'frozen','retrain'}."""
    names = list(getattr(records[0], which).keys())
    out: Dict[str, Dict[str, float]] = {}
    for n in names:
        vals = [getattr(r, which)[n] for r in records]
        mu = stats.mean(vals)
        sd = stats.pstdev(vals) if len(vals) > 1 else 0.0
        out[n] = {"mu": mu, "sigma": sd, "vals": vals}
    return out


def rank_consistency(records: List[SeedRecord], which: str) -> Dict[str, object]:
    """How stable is the top-of-ranking across seeds? Reports the modal rank-1
    term, the fraction of seeds that agree on it, and mean Spearman-ish overlap
    of the full ordering (Kendall-tau style agreement on pairwise order)."""
    rank_attr = "retrain_rank" if which == "retrain" else "frozen_rank"
    ranks = [getattr(r, rank_attr) for r in records]
    top1 = [rk[0] for rk in ranks]
    modal = max(set(top1), key=top1.count)
    top1_agreement = top1.count(modal) / len(top1)

    # mean pairwise Kendall tau across seed rankings
    def kendall(a: List[str], b: List[str]) -> float:
        idx_b = {x: i for i, x in enumerate(b)}
        items = [x for x in a if x in idx_b]
        n = len(items)
        if n < 2:
            return 1.0
        conc = disc = 0
        for i in range(n):
            for j in range(i + 1, n):
                ai, aj = i, j
                bi, bj = idx_b[items[i]], idx_b[items[j]]
                s = (aj - ai) * (bj - bi)
                if s > 0:
                    conc += 1
                elif s < 0:
                    disc += 1
        tot = conc + disc
        return (conc - disc) / tot if tot else 1.0

    taus = []
    for i in range(len(ranks)):
        for j in range(i + 1, len(ranks)):
            taus.append(kendall(ranks[i], ranks[j]))
    mean_tau = stats.mean(taus) if taus else 1.0
    return {"modal_top1": modal, "top1_agreement": top1_agreement, "mean_kendall_tau": mean_tau}


def evaluate_criteria(
    records: List[SeedRecord],
    expected_important: List[str],
    expected_eliminable: List[str],
    sigma_modest_frac: float = 1.0,
) -> Dict[str, object]:
    """Apply the success/failure criteria.

    A term is 'structurally important' if mean retrain eliminability is positive,
    variance is modest (sigma < sigma_modest_frac * |mu|), and its ranking is
    consistent. Returns a dict of boolean checks plus the supporting numbers."""
    rt = term_stats(records, "retrain")
    rc = rank_consistency(records, "retrain")

    def important(n: str) -> bool:
        s = rt[n]
        modest = s["sigma"] <= sigma_modest_frac * (abs(s["mu"]) + 1e-12)
        return s["mu"] > 0 and modest

    important_terms = [n for n in rt if important(n)]

    # criterion checks
    rankings_stable = rc["mean_kendall_tau"] >= 0.5 and rc["top1_agreement"] >= 0.6
    physics_match = all(e in important_terms for e in expected_important)
    dominant_not_eliminable = all(rt[e]["mu"] > 0 for e in expected_important if e in rt)
    ordering_nonzero = stats.mean([r.ordering_field for r in records]) > 1e-6
    eliminable_ok = all(
        rt[e]["mu"] <= rt[expected_important[0]]["mu"] for e in expected_eliminable if e in rt
    ) if expected_important else True

    return {
        "retrain_stats": {n: {"mu": rt[n]["mu"], "sigma": rt[n]["sigma"]} for n in rt},
        "rank_consistency": rc,
        "important_terms": important_terms,
        "checks": {
            "rankings_stable": rankings_stable,
            "rankings_match_physics": physics_match,
            "dominant_terms_not_eliminable": dominant_not_eliminable,
            "ordering_defect_nonzero": ordering_nonzero,
            "expected_eliminable_rank_low": eliminable_ok,
        },
    }


def run_phase(
    mod,
    name: str,
    physics_terms: set,
    data_terms: set,
    residual_term: str,
    expected_important: List[str],
    expected_eliminable: List[str],
    seeds: List[int],
    train_steps: int,
    retrain_steps: int,
    order_k: int,
    lr: float,
    batch_kw: dict,
) -> dict:
    print(f"\n########## PHASE: {name}  ({len(seeds)} seeds) ##########")
    t0 = time.time()
    records: List[SeedRecord] = []
    for s in seeds:
        ts = time.time()
        r = run_seed(mod, s, physics_terms, data_terms, residual_term,
                     train_steps, retrain_steps, order_k, lr, batch_kw)
        records.append(r)
        print(f"  seed {s}: loss={r.final_loss:.3e} pde_res={r.pde_residual:.3e} "
              f"retrain_top={r.retrain_rank[:3]} "
              f"order_field={r.ordering_field:.2e} rank_changed={r.ordering_rank_changed} "
              f"({time.time()-ts:.0f}s)")

    fstats = term_stats(records, "frozen")
    rstats = term_stats(records, "retrain")
    crit = evaluate_criteria(records, expected_important, expected_eliminable)

    print(f"\n  --- retrain eliminability mu +/- sigma (normalized) ---")
    for n in sorted(rstats, key=lambda k: rstats[k]["mu"], reverse=True):
        print(f"    {n:14s} mu={rstats[n]['mu']:+.3e}  sigma={rstats[n]['sigma']:.3e}")
    print(f"  rank consistency: {crit['rank_consistency']}")
    print(f"  important terms: {crit['important_terms']}")
    print(f"  CHECKS: {json.dumps(crit['checks'])}")
    print(f"  phase time: {time.time()-t0:.0f}s")

    return {
        "phase": name,
        "seeds": seeds,
        "records": [r.__dict__ for r in records],
        "frozen_stats": {k: {kk: vv for kk, vv in v.items() if kk != 'vals'} for k, v in fstats.items()},
        "retrain_stats": {k: {kk: vv for kk, vv in v.items() if kk != 'vals'} for k, v in rstats.items()},
        "criteria": crit,
    }
