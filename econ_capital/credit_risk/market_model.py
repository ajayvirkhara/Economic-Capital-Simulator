"""Simple market/credit factor models used by Credit Risk modules."""

from __future__ import annotations
import numpy as np


def simulate_credit_factors(
    n_paths: int,
    n_steps: int,
    rho: float = 0.2,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate (n_paths x n_steps) credit-systematic factors using 1-factor Gaussian model:
      F_{i,t} = rho * Z_sys_i + sqrt(1 - rho^2) * Z_idio_{i,t}

    Returns
    -------
    factors : np.ndarray, shape (n_paths, n_steps)
    """
    rng = np.random.default_rng(seed)
    z_sys = rng.standard_normal(n_paths)[:, None]  # (n_paths, 1)
    z_idio = rng.standard_normal((n_paths, n_steps))
    return rho * z_sys + (np.sqrt(max(0.0, 1.0 - rho**2)) * z_idio)
