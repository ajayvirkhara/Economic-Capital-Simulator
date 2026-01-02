"""
Shock generators for market risk simulations.

Notes
-----
The Student-t distribution is used to capture fat-tailed
factor return distributions, consistent with Basel/EC modelling practice.
"""

import numpy as np
from econ_capital.utils import setup_logging, validate_shape

logger = setup_logging(__name__)


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
    """
    Generate draws from a multivariate Student-t distribution.

    This uses the standard Gaussian mixture representation:
        X = μ + Z / sqrt(χ²_df / df)
    where Z ~ N(0, Σ).

    This is mathematically equivalent to applying a Gaussian copula
    followed by inverse t-CDF marginals (t-copula with ν = df degrees of freedom).
    The direct mixture method is used here for computational efficiency and numerical stability.

    Commonly used in economic capital modelling to capture fat-tailed correlated market shocks
    consistent with Basel and ICAAP practices.

    Parameters
    ----------
    n : int
        Number of simulation paths.
    mu : np.ndarray
        Mean vector (typically zero for risk-neutral shocks).
    cov : np.ndarray
        Covariance matrix of the underlying Gaussian.
    df : float
        Degrees of freedom (ν > 2 required for finite variance).
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Shape (n, k) shocks where k = dimension of cov.
    """
    g = rng.chisquare(df, size=n) / df  # shape (n,)
    z = rng.multivariate_normal(mean=np.zeros(cov.shape[0]), cov=cov, size=n)
    shocks = mu + z / np.sqrt(g)[:, None]
    expected_shape = (n, cov.shape[0])
    validate_shape(shocks, expected_shape, name="shocks")  # fixed argument order
    logger.debug(
        "Generated shocks with mean %.4f and std %.4f",
        float(shocks.mean()),
        float(shocks.std()),
    )
    return shocks
