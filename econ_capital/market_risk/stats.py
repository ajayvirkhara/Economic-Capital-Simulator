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
