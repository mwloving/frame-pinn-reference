"""Aggregate persisted seed records (JSONL per benchmark) into cross-seed
statistics, evaluate success/failure criteria per phase, and emit a final
report (JSON + human-readable markdown). Reconstructs SeedRecord objects from
disk so no retraining happens here."""

import sys
import json

sys.path.insert(0, "..")
sys.path.insert(0, ".")

from validation import (
    SeedRecord,
    term_stats,
    rank_consistency,
    evaluate_criteria,
)

PHASES = {
    "Burgers": dict(
        file="seed_results/burgers.jsonl",
        expected_important=["u_t", "ic"],
        expected_eliminable=["diff"],
        grouped=False,
    ),
    "Allen-Cahn": dict(
        file="seed_results/allen_cahn.jsonl",
        expected_important=["bc", "ic"],
        expected_eliminable=["diff"],
        grouped=False,
    ),
    "Navier-Stokes": dict(
        file="seed_results/navier_stokes.jsonl",
        expected_important=["data", "pressure"],
        expected_eliminable=["viscous"],   # at this Re, expect viscous low-ish
        grouped=True,
    ),
}


def load(path):
    recs = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        recs.append(SeedRecord(**json.loads(line)))
    return sorted(recs, key=lambda r: r.seed)


def summarize_phase(name, cfg):
    recs = load(cfg["file"])
    rstats = term_stats(recs, "retrain")
    fstats = term_stats(recs, "frozen")
    rc = rank_consistency(recs, "retrain")
    crit = evaluate_criteria(recs, cfg["expected_important"], cfg["expected_eliminable"])

    return {
        "phase": name,
        "n_seeds": len(recs),
        "final_loss": [round(r.final_loss, 4) for r in recs],
        "pde_residual": [round(r.pde_residual, 4) for r in recs],
        "retrain_rank_per_seed": [r.retrain_rank for r in recs],
        "ordering_field_per_seed": [round(r.ordering_field, 3) for r in recs],
        "ordering_rank_changed": [r.ordering_rank_changed for r in recs],
        "retrain_stats": {k: {"mu": round(v["mu"], 4), "sigma": round(v["sigma"], 4)}
                          for k, v in rstats.items()},
        "frozen_stats": {k: {"mu": round(v["mu"], 4), "sigma": round(v["sigma"], 4)}
                         for k, v in fstats.items()},
        "rank_consistency": rc,
        "important_terms": crit["important_terms"],
        "checks": crit["checks"],
    }


def phase_verdict(checks):
    return "PASS" if all(checks.values()) else "PARTIAL/FAIL"


def main():
    report = {"phases": []}
    for name, cfg in PHASES.items():
        report["phases"].append(summarize_phase(name, cfg))

    with open("validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # console summary
    print("=" * 64)
    print("FRAME-PINN v1.0  VALIDATION REPORT")
    print("=" * 64)
    for p in report["phases"]:
        print(f"\n### {p['phase']}  ({p['n_seeds']} seeds)  -> {phase_verdict(p['checks'])}")
        print(f"  final loss per seed : {p['final_loss']}")
        print(f"  retrain ranking/seed:")
        for rk in p["retrain_rank_per_seed"]:
            print(f"      {rk}")
        print(f"  rank consistency    : top1={p['rank_consistency']['modal_top1']} "
              f"agree={p['rank_consistency']['top1_agreement']:.2f} "
              f"kendall_tau={p['rank_consistency']['mean_kendall_tau']:.2f}")
        print(f"  retrain mu+/-sigma  :")
        for k in sorted(p["retrain_stats"], key=lambda x: p["retrain_stats"][x]["mu"], reverse=True):
            s = p["retrain_stats"][k]
            print(f"      {k:14s} mu={s['mu']:+.3f}  sigma={s['sigma']:.3f}")
        print(f"  ordering field/seed : {p['ordering_field_per_seed']}  "
              f"changed={p['ordering_rank_changed']}")
        print(f"  important terms     : {p['important_terms']}")
        print(f"  checks              : {json.dumps(p['checks'])}")

    # overall
    all_checks = {}
    for p in report["phases"]:
        for k, v in p["checks"].items():
            all_checks.setdefault(k, []).append(v)
    print("\n" + "=" * 64)
    print("OVERALL")
    for k, vs in all_checks.items():
        print(f"  {k:34s}: {sum(vs)}/{len(vs)} phases")
    print("=" * 64)


if __name__ == "__main__":
    main()
