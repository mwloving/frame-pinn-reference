"""FRAME guided Monte Carlo: v0.1 minimal scalar case.

Path space is the set of discretized scalar paths gamma: [0, T] -> R driven by
a Brownian baseline P_0. The admissibility cost C(gamma) is the time integrated
squared deviation from a linear reference path p(t) from theta_0 to theta_star.
The FRAME measure is the Boltzmann tilt

    d mu_F(gamma) propto exp(-beta * C(gamma)) d P_0(gamma).

Two path functional estimators are compared on the same pool of iid Brownian
samples:

    naive MC:        estimates E_{P_0}[phi(theta_T)]
    FRAME SNIS:      estimates E_{mu_F}[phi(theta_T)]

where phi(theta_T) = (theta_T - theta_star)^2 is the endpoint alignment
observable. These target different integrals by design. The point of v0.1 is
to validate the reweighting pipeline and show how the FRAME estimator responds
to inverse temperature beta, weight degeneracy, and sample count.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class PathConfig:
    """Discretization parameters for the Brownian baseline."""

    n_steps: int = 100
    T: float = 1.0
    theta_0: float = 0.0
    sigma: float = 1.0

    @property
    def dt(self) -> float:
        return self.T / self.n_steps

    @property
    def time_grid(self) -> np.ndarray:
        return np.linspace(0.0, self.T, self.n_steps + 1)


def sample_brownian_paths(
    n_samples: int,
    cfg: PathConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw n_samples iid Brownian paths from P_0.

    Returns an array of shape (n_samples, n_steps + 1) with theta[:, 0] equal
    to cfg.theta_0 and increments distributed as N(0, sigma^2 * dt).
    """
    increments = rng.normal(
        loc=0.0,
        scale=cfg.sigma * np.sqrt(cfg.dt),
        size=(n_samples, cfg.n_steps),
    )
    paths = np.empty((n_samples, cfg.n_steps + 1))
    paths[:, 0] = cfg.theta_0
    paths[:, 1:] = cfg.theta_0 + np.cumsum(increments, axis=1)
    return paths


def linear_reference(theta_star: float, cfg: PathConfig) -> np.ndarray:
    """Geodesic reference path p(t) from theta_0 to theta_star.

    This is the attractor that the admissibility cost penalizes deviation from.
    Any smooth admissible trajectory could be substituted here; the linear case
    is chosen for v0.1 because the tilted measure is then analytically
    tractable as a Gaussian process.
    """
    t = cfg.time_grid
    return cfg.theta_0 + (theta_star - cfg.theta_0) * (t / cfg.T)


def admissibility_cost(
    paths: np.ndarray,
    reference: np.ndarray,
    cfg: PathConfig,
) -> np.ndarray:
    """Time integrated squared deviation from the reference path.

    C(gamma) = integral_0^T (gamma(t) - p(t))^2 dt, approximated by the
    trapezoidal rule on the discretization grid. Returns a (n_samples,) vector.
    """
    deviation_sq = (paths - reference[None, :]) ** 2
    return np.trapezoid(deviation_sq, dx=cfg.dt, axis=1)


def endpoint_observable(paths: np.ndarray, theta_star: float) -> np.ndarray:
    """Scalar endpoint observable phi(theta_T) = (theta_T - theta_star)^2."""
    return (paths[:, -1] - theta_star) ** 2


def frame_weights(cost: np.ndarray, beta: float) -> np.ndarray:
    """Boltzmann weights w(gamma) = exp(-beta * C(gamma)).

    The log weights are shifted by their max before exponentiation for
    numerical stability. This constant shift cancels in every self normalized
    estimator that consumes these weights, so the downstream estimator value
    is unchanged.
    """
    log_w = -beta * np.asarray(cost, dtype=float)
    log_w -= log_w.max()
    return np.exp(log_w)


def naive_estimator(phi: np.ndarray) -> float:
    """Plain Monte Carlo estimate of E_{P_0}[phi]."""
    return float(np.mean(phi))


def frame_estimator(phi: np.ndarray, weights: np.ndarray) -> float:
    """Self normalized importance sampling estimate of E_{mu_F}[phi]."""
    return float(np.sum(weights * phi) / np.sum(weights))


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size: (sum w)^2 / sum w^2.

    Reports how many independent samples the weighted pool is equivalent to.
    At beta = 0 all weights are unity and ESS = N; as beta grows, weights
    concentrate on a shrinking subset of paths and ESS collapses toward 1.
    """
    s1 = np.sum(weights)
    s2 = np.sum(weights ** 2)
    return float(s1 * s1 / s2)
