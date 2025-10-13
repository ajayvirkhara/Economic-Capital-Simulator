"""
Unit tests for econ_capital.market_risk.shocks module.
"""

import numpy as np

from econ_capital.market_risk.shocks import mv_t_draws


def test_mv_t_draws_shape():
    """Tests that the multivariate Student-t draws function returns the correct dimensions (n_paths x n_factors)."""
    n, k = 500, 3
    mu = np.zeros(k)
    cov = np.eye(k)
    df = 7
    rng = np.random.default_rng(123)

    # Generate the draws
    draws = mv_t_draws(n, mu, cov, df, rng)

    # Check that the output shape matches the requested number of paths (n) and factors (k)
    assert draws.shape == (n, k), "Draws have wrong shape"
