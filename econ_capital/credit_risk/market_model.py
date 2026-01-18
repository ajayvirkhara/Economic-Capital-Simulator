"""Simple market/credit factor models used by Credit Risk modules."""

from __future__ import annotations
import numpy as np


def simulate_credit_factors(
    n_paths: int,
    n_steps: int,
    corr: float = 0.2,
    vol: float = 0.20,
    mean_reversion: float = 0.0,
    dt: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate (n_paths x n_steps) credit-systematic factors using 1-factor Gaussian model:
      F_{i,t} = rho * Z_sys_i + sqrt(1 - rho^2) * Z_idio_{i,t}

    Returns
    -------
    factors : np.ndarray, shape (n_paths, n_steps)
    """
    # Handle list input (e.g., from YAML)
    if isinstance(corr, list):
        corr = np.array(corr)

    if isinstance(corr, np.ndarray):
        if corr.ndim == 0:  # scalar array
            corr = float(corr)
        elif corr.ndim == 1:  # 1D array → treat as invalid or take mean
            if corr.size == 1:
                corr = float(corr[0])
            else:
                raise ValueError(
                    f"1D correlation array of length > 1 is ambiguous. "
                    f"Use scalar or square matrix. Got: {corr}"
                )
        elif corr.ndim == 2:
            if corr.shape[0] != corr.shape[1]:
                raise ValueError("Correlation matrix must be square")
            mask = ~np.eye(corr.shape[0], dtype=bool)
            off_diag_mean = (
                corr[mask].mean() if np.any(mask) else corr.diagonal().mean()
            )
            corr = float(off_diag_mean)
        else:
            raise ValueError("Correlation must be scalar or 2D square matrix")

    corr = float(corr)  # Ensure it's a Python float
    corr = np.clip(corr, -1.0, 1.0)  # Add validation to prevent invalid correlations

    rng = np.random.default_rng(seed)
    z_sys = rng.standard_normal(n_paths)[:, None]  # (n_paths, 1)
    z_idio = rng.standard_normal((n_paths, n_steps))

    # Scale shocks by volatility parameter
    sys_shock = vol * z_sys
    idio_shock = vol * z_idio

    return corr * sys_shock + np.sqrt(max(0.0, 1.0 - corr**2)) * idio_shock


def simulate_term_structure_volatility(
    times: np.ndarray,
    vol_short: float = 0.25,
    vol_long: float = 0.15,
    mean_reversion: float = 0.3,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate time-varying volatility term structure with mean reversion.

    Uses a determinisitic exponential decay model.

    Parameters
    ----------
    times : np.ndarray
        Time points for simulation
    vol_short : float
        Short-term (spot) volatility
    vol_long : float
        Long-term equilibrium volatility
    mean_reversion : float
        Speed of mean reversion to long-term vol

    Returns
    -------
    vol_curve : np.ndarray
        Time-varying volatility at each time point
    """
    vol_curve = vol_long + (vol_short - vol_long) * np.exp(-mean_reversion * times)
    return vol_curve
