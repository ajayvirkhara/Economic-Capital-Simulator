"""
Shock generators for market risk simulations.

Notes
-----
The Student-t distribution is used to capture fat-tailed
factor return distributions, consistent with Basel/EC modelling practice.
"""

import numpy as np


# ---------------------------------------------------------------------
# Multivariate Student-t draws & tail risk stats
# ---------------------------------------------------------------------
def mv_t_draws(
    n: int,
    mu: np.ndarray,
    cov: np.ndarray,
    df: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draws from a multivariate Student-t via normal/chi-square mixture."""
    g = rng.chisquare(df, size=n) / df  # shape (n,)
    z = rng.multivariate_normal(mean=np.zeros(cov.shape[0]), cov=cov, size=n)
    return mu + z / np.sqrt(g)[:, None]
