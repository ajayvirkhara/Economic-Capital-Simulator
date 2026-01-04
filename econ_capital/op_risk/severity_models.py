"""
Severity Models for Op Risk LDA
===============================

Hybrid: Lognormal (body) + GPD (tail).

- Fits: Lognormal to losses < threshold, GPD to losses >= threshold
- Simulation: Draw from mixture using fitted tail probability

Usage:
    params = fit_lognormal_gpd(losses, threshold=None)
    sevs = simulate_severity(n_draws, params)
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
from scipy import stats
from scipy.optimize import minimize


def fit_lognormal_gpd(
    losses: np.ndarray, threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Fit hybrid severity model using:
      - Lognormal distribution for the body (bulk of losses)
      - Generalized Pareto Distribution (GPD) for the tail using Peaks-Over-Threshold (POT)

    This is a standard Extreme Value Theory (EVT) approach for operational risk tail modelling,
    consistent with LDA best practices under Basel II/III and ICAAP.

    Args:
        losses: Array of positive loss amounts
        threshold: Threshold for tail. If None, use 99th percentile

    Returns:
        dict: Fitted parameters including tail probability
    """
    losses = np.asarray(losses)
    if len(losses) < 20 or np.any(losses <= 0):
        raise ValueError("Invalid losses: need >20 positive values")

    if threshold is None:
        threshold = np.quantile(losses, 0.99)

    body_losses = losses[losses < threshold]
    tail_losses = (
        losses[losses >= threshold] - threshold
    )  # excess over threshold for GPD

    # ---- Fit Lognormal to body (below threshold) ----
    if len(body_losses) > 0:
        log_body = np.log(body_losses)
        lognormal_mu = np.mean(log_body)
        lognormal_sigma = np.std(log_body, ddof=1)
    else:
        lognormal_mu, lognormal_sigma = np.log(np.median(losses)), 0.5

    # ---- Fit GPD to tail (above threshold) ----
    if len(tail_losses) > 0:

        def neg_log_lik(params):
            xi, beta = params
            if beta <= 0:  # Beta must be greater than 0
                return np.inf
            if xi == 0:  # GPD reduces to exponential distribution if xi equals 0
                return -np.sum(stats.expon.logpdf(tail_losses, scale=beta))
            return -np.sum(stats.genpareto.logpdf(tail_losses, c=xi, scale=beta))

        # ---- MLE Estimation for negative log likelihood function ----
        res = minimize(
            neg_log_lik,
            x0=[0.1, np.std(tail_losses)],
            bounds=[(-0.25, 0.5), (1e-6, None)],
            method="L-BFGS-B",
        )
        if not res.success:
            gpd_xi, gpd_beta = 0.0, np.mean(tail_losses)
        else:
            gpd_xi, gpd_beta = res.x
    else:
        gpd_xi, gpd_beta = 0.0, threshold * 0.1

    tail_prob = len(tail_losses) / len(losses)  # likelihood of loss exceeding threshold

    return {
        "lognormal_mu": float(lognormal_mu),
        "lognormal_sigma": float(lognormal_sigma),
        "gpd_xi": float(gpd_xi),
        "gpd_beta": float(gpd_beta),
        "threshold": float(threshold),
        "tail_prob": float(tail_prob),
    }


def simulate_severity(
    n_draws: int,
    params: Dict[str, Any],
    threshold=None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Simulate severities from fitted hybrid model.

    Args:
        n_draws: Number of simulated losses
        params: Dictionary from fit_lognormal_gpd

    Returns:
        np.ndarray: Simulated severities
    """

    if rng is None:
        rng = np.random.default_rng()  # fallback

    tail_prob = params.get("tail_prob", 0.05)
    threshold = params["threshold"]

    is_tail = rng.binomial(1, tail_prob, n_draws)
    sevs = np.zeros(n_draws)

    # ---- Body draws ----
    body_mask = is_tail == 0
    n_body = np.sum(body_mask)
    if n_body > 0:
        sevs[body_mask] = stats.lognorm.rvs(
            s=params["lognormal_sigma"],
            scale=np.exp(params["lognormal_mu"]),
            size=n_body,
        )

    # ---- Tail draws ----
    tail_mask = is_tail == 1
    n_tail = np.sum(tail_mask)
    if n_tail > 0:
        xi = params.get(
            "gpd_xi", 0.0
        )  # Tail draws use GPD → EVT modelling of extreme losses
        if xi == 0:
            excess = stats.expon.rvs(scale=params["gpd_beta"], size=n_tail)
        else:
            excess = stats.genpareto.rvs(
                c=params["gpd_xi"], scale=params["gpd_beta"], size=n_tail
            )
        sevs[tail_mask] = threshold + excess

    return sevs
