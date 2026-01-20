"""
Statistical utilities for market risk capital calculations.

Notes
-----
These operate on simulated portfolio P&L vectors and return positive
capital requirements (loss magnitudes).
"""

import numpy as np


def left_tail_var(pnl: np.ndarray, q: float) -> float:
    """Left-tail VaR at confidence q."""
    return -np.quantile(pnl, 1.0 - q)


def left_tail_es(pnl: np.ndarray, q: float) -> float:
    """Left-tail Expected Shortfall at confidence q."""
    cutoff = np.quantile(pnl, 1.0 - q)
    return -pnl[pnl <= cutoff].mean()


def compute_covar(
    portfolio_losses: np.ndarray, position_losses: np.ndarray, alpha: float = 0.99
) -> tuple[float, float]:
    """
    Compute Conditional VaR (CVaR) - systemic risk contribution.

    Parameters
    ----------
    portfolio_losses : np.ndarray
        Total portfolio loss distribution (n_sims,)
    position_losses : np.ndarray
        Individual position loss distribution (n_sims,)
    alpha : float
        Confidence level

    Returns
    -------
    covar : float
        Conditional VaR (portfolio VaR | position at its VaR)
    delta_covar : float
        Incremental systemic risk (CoVaR - baseline VaR)
    """
    # Baseline portfolio VaR
    var_portfolio = np.quantile(portfolio_losses, alpha)

    # Position VaR
    var_position = np.quantile(position_losses, alpha)

    # Conditional: portfolio losses when position is stressed
    stressed_mask = position_losses >= var_position
    covar = np.quantile(portfolio_losses[stressed_mask], alpha)

    delta_covar = covar - var_portfolio

    return covar, delta_covar
