"""Run the FRAME guided Monte Carlo v0.1 experiment and save figures.

Two sweeps are executed:

    (1) convergence sweep at fixed beta:
            for each sample count N in a log spaced grid, and for R
            independent seeds, compute naive MC and FRAME SNIS estimates.
            Plot estimator mean plus or minus one standard deviation across
            seeds, and empirical variance across seeds, both as a function
            of N.

    (2) beta sweep at fixed N:
            hold sample count fixed and sweep beta. Plot the FRAME estimate
            and the effective sample fraction N_eff / N to show the
            reweighting regime versus the weight collapse regime.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from frame_mc import (
    PathConfig,
    admissibility_cost,
    effective_sample_size,
    endpoint_observable,
    frame_estimator,
    frame_weights,
    linear_reference,
    naive_estimator,
    sample_brownian_paths,
)


def run_convergence_sweep(
    cfg: PathConfig,
    theta_star: float,
    beta: float,
    sample_sizes: np.ndarray,
    n_repetitions: int,
    seed: int = 0,
) -> dict:
    """Record naive and FRAME estimates across sample counts and seeds."""
    rng = np.random.default_rng(seed)
    reference = linear_reference(theta_star, cfg)
    n_max = int(sample_sizes.max())

    naive_vals = np.empty((n_repetitions, len(sample_sizes)))
    frame_vals = np.empty((n_repetitions, len(sample_sizes)))
    ess_vals = np.empty((n_repetitions, len(sample_sizes)))

    for r in range(n_repetitions):
        paths = sample_brownian_paths(n_max, cfg, rng)
        phi_all = endpoint_observable(paths, theta_star)
        cost_all = admissibility_cost(paths, reference, cfg)

        for j, n in enumerate(sample_sizes):
            n = int(n)
            phi = phi_all[:n]
            w = frame_weights(cost_all[:n], beta)
            naive_vals[r, j] = naive_estimator(phi)
            frame_vals[r, j] = frame_estimator(phi, w)
            ess_vals[r, j] = effective_sample_size(w)

    return {
        "sample_sizes": sample_sizes,
        "naive_vals": naive_vals,
        "frame_vals": frame_vals,
        "ess_vals": ess_vals,
    }


def run_beta_sweep(
    cfg: PathConfig,
    theta_star: float,
    betas: np.ndarray,
    n_samples: int,
    seed: int = 1,
) -> dict:
    """Examine FRAME estimate and weight degeneracy versus beta at fixed N."""
    rng = np.random.default_rng(seed)
    reference = linear_reference(theta_star, cfg)
    paths = sample_brownian_paths(n_samples, cfg, rng)
    phi = endpoint_observable(paths, theta_star)
    cost = admissibility_cost(paths, reference, cfg)

    frame_vals = np.empty(len(betas))
    ess_vals = np.empty(len(betas))
    for i, beta in enumerate(betas):
        w = frame_weights(cost, beta)
        frame_vals[i] = frame_estimator(phi, w)
        ess_vals[i] = effective_sample_size(w)

    return {
        "betas": betas,
        "frame_vals": frame_vals,
        "ess_vals": ess_vals,
        "naive_val": naive_estimator(phi),
    }


def plot_convergence(results: dict, beta: float, out_path: Path) -> None:
    sizes = results["sample_sizes"]
    naive = results["naive_vals"]
    frame = results["frame_vals"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(
        sizes, naive.mean(axis=0),
        label=r"naive MC (targets $E_{P_0}[\phi]$)",
        color="C0", lw=2,
    )
    ax.fill_between(
        sizes,
        naive.mean(0) - naive.std(0),
        naive.mean(0) + naive.std(0),
        alpha=0.2, color="C0",
    )
    ax.plot(
        sizes, frame.mean(axis=0),
        label=fr"FRAME SNIS (targets $E_{{\mu_F}}[\phi]$, $\beta={beta}$)",
        color="C3", lw=2,
    )
    ax.fill_between(
        sizes,
        frame.mean(0) - frame.std(0),
        frame.mean(0) + frame.std(0),
        alpha=0.2, color="C3",
    )
    ax.set_xscale("log")
    ax.set_xlabel("sample count $N$")
    ax.set_ylabel("estimator value")
    ax.set_title("Estimator value vs sample count")
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(sizes, naive.var(axis=0), label="Var[naive]", color="C0", lw=2)
    ax.plot(sizes, frame.var(axis=0), label="Var[FRAME]", color="C3", lw=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sample count $N$")
    ax.set_ylabel("empirical variance across seeds")
    ax.set_title("Estimator variance vs sample count")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_beta_sweep(results: dict, n_samples: int, out_path: Path) -> None:
    betas = results["betas"]
    frame_vals = results["frame_vals"]
    ess_vals = results["ess_vals"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(betas, frame_vals, color="C3", marker="o", lw=2, ms=5)
    ax.axhline(
        results["naive_val"],
        color="C0", linestyle="--", lw=1.5,
        label=r"naive estimate ($\beta = 0$)",
    )
    ax.set_xlabel(r"inverse temperature $\beta$")
    ax.set_ylabel(r"FRAME estimate of $E_{\mu_F}[\phi]$")
    ax.set_title("FRAME estimate vs inverse temperature")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(betas, ess_vals / n_samples, color="C2", marker="o", lw=2, ms=5)
    ax.set_xlabel(r"inverse temperature $\beta$")
    ax.set_ylabel(r"effective sample fraction $N_{\mathrm{eff}} / N$")
    ax.set_title(f"Weight degeneracy at $N = {n_samples}$")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    out_dir = Path(__file__).parent / "figures"
    out_dir.mkdir(exist_ok=True)

    cfg = PathConfig(n_steps=100, T=1.0, theta_0=0.0, sigma=1.0)
    theta_star = 1.0
    beta = 2.0

    sample_sizes = np.unique(
        np.round(np.logspace(np.log10(50), np.log10(5000), 15)).astype(int)
    )
    n_repetitions = 200

    print(
        f"Running convergence sweep: {n_repetitions} seeds, "
        f"N in {sample_sizes.min()}..{sample_sizes.max()}, beta = {beta}"
    )
    conv = run_convergence_sweep(
        cfg, theta_star, beta, sample_sizes, n_repetitions, seed=0,
    )
    plot_convergence(conv, beta, out_dir / "v01_convergence.png")
    print(f"  saved {out_dir / 'v01_convergence.png'}")

    betas = np.linspace(0.0, 10.0, 21)
    n_beta = 5000
    print(
        f"Running beta sweep: N = {n_beta}, "
        f"beta in {betas.min()}..{betas.max()} across {len(betas)} points"
    )
    bsweep = run_beta_sweep(cfg, theta_star, betas, n_samples=n_beta, seed=1)
    plot_beta_sweep(bsweep, n_beta, out_dir / "v01_beta_sweep.png")
    print(f"  saved {out_dir / 'v01_beta_sweep.png'}")

    print("\n--- v0.1 summary ---")
    print(f"theta_star = {theta_star}, beta = {beta}")
    n_final = int(sample_sizes[-1])
    print(f"At N = {n_final}:")
    print(
        f"  naive MC estimate:  "
        f"{conv['naive_vals'][:, -1].mean():.4f}"
        f" +/- {conv['naive_vals'][:, -1].std():.4f}"
    )
    print(
        f"  FRAME MC estimate:  "
        f"{conv['frame_vals'][:, -1].mean():.4f}"
        f" +/- {conv['frame_vals'][:, -1].std():.4f}"
    )
    print(
        f"  FRAME effective sample fraction: "
        f"{conv['ess_vals'][:, -1].mean() / n_final:.3f}"
    )
    print(
        "Note: the two estimators target different integrals. Naive MC "
        "converges to E_{P_0}[phi]; FRAME SNIS converges to E_{mu_F}[phi]."
    )


if __name__ == "__main__":
    main()
