"""
Unit tests for functions in econ_capital.credit_risk.market_model, verifying the
'simulate_credit_factors' function which generates correlated random credit
factor paths.
"""

import numpy as np

from econ_capital.credit_risk.market_model import simulate_credit_factors


def test_credit_factor_dimensions():
    # Tests that the output array has the correct dimensions: (n_paths, n_names).

    n_paths, n_names = 10000, 5
    rho = 0.5

    factors = simulate_credit_factors(n_paths, n_names, rho, seed=42)

    empirical_corr = np.corrcoef(factors.T)

    # Assert correlation is close to rho^2 (for a one-factor model)
    expected_corr = np.full((n_names, n_names), rho**2)
    np.fill_diagonal(expected_corr, 1.0)

    np.testing.assert_allclose(empirical_corr, expected_corr, atol=0.05)

    assert factors.shape == (n_paths, n_names)


def test_credit_factor_correlation():
    """
    Tests that the empirical cross-correlation of simulated factors aligns with
    the expected value (rho^2) for a one-factor Gaussian credit model.

    The function simulate_credit_factors is assumed to implement a one-factor
    model where factors are correlated only through a single systematic factor
    with correlation rho.
    """
    n_paths, n_steps = 10000, 3
    rho = 0.5  # The scalar systematic correlation factor for the 1-factor model.
    factors = simulate_credit_factors(n_paths, n_steps, rho, seed=42)

    # Calculate the empirical correlation matrix from the simulated factors (variables as rows).
    empirical_corr = np.corrcoef(factors.T)

    # For a one-factor model, the expected off-diagonal correlation between factors
    # is the square of the systematic correlation: rho^2.
    expected_corr = np.full((n_steps, n_steps), rho**2)
    np.fill_diagonal(expected_corr, 1.0)  # Diagonals must be 1.0

    # Check that the empirical correlation matches the theoretical rho^2 correlation.
    np.testing.assert_allclose(empirical_corr, expected_corr, atol=0.05)
