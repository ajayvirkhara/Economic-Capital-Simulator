"""
Unit tests for functions in econ_capital.credit_risk.market_model, verifying the
'simulate_credit_factors' function which generates correlated random credit
factor paths.
"""

import numpy as np
import pytest

from econ_capital.credit_risk.market_model import (
    simulate_credit_factors,
    simulate_term_structure_volatility,
)


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


# Test for corr as list
def test_simulate_credit_factors_corr_as_list_raises_error():
    corr_list = [0.3, 0.4, 0.5]
    with pytest.raises(
        ValueError, match="1D correlation array of length > 1 is ambiguous"
    ):
        simulate_credit_factors(n_paths=100, n_steps=3, corr=corr_list, seed=42)


# Confirm single value lists work correctly
def test_simulate_credit_factors_corr_as_single_value_list():
    corr_list = [0.35]  # length 1 → treated as scalar
    factors = simulate_credit_factors(n_paths=10000, n_steps=3, corr=corr_list, seed=42)
    assert factors.shape == (10000, 3)

    empirical_corr = np.corrcoef(factors.T)
    off_diagonals = empirical_corr[~np.eye(3, dtype=bool)]

    rho = 0.35
    expected_off_diag = rho**2  # ρ² ≈ 0.1225
    np.testing.assert_allclose(
        off_diagonals,
        expected_off_diag,
        atol=0.05,  # generous tolerance
    )


# Test for corr as ndarray: scalar (0-d)
def test_simulate_credit_factors_corr_as_ndarray_scalar():
    corr_array = np.array(0.25)  # 0-d scalar array
    factors = simulate_credit_factors(n_paths=100, n_steps=3, corr=corr_array, seed=42)
    assert factors.shape == (100, 3)
    empirical_corr = np.corrcoef(factors.T)
    expected_off_diag = 0.25
    np.testing.assert_allclose(empirical_corr[0, 1], expected_off_diag, atol=0.1)


# Test for corr as ndarray: 1-element
def test_simulate_credit_factors_corr_as_ndarray_one_element():
    corr_array = np.array([0.35])  # 1-element array
    factors = simulate_credit_factors(
        n_paths=10000, n_steps=3, corr=corr_array, seed=42
    )
    assert factors.shape == (10000, 3)
    empirical_corr = np.corrcoef(factors.T)
    expected_off_diag = 0.35**2
    np.testing.assert_allclose(empirical_corr[0, 1], expected_off_diag, atol=0.1)


# Test for corr as ndarray: matrix, mean off-diagonal
def test_simulate_credit_factors_corr_as_ndarray_matrix():
    corr_matrix = np.array(
        [[1.0, 0.2, 0.3], [0.2, 1.0, 0.4], [0.3, 0.4, 1.0]]
    )  # Off-diag mean ≈ 0.3
    factors = simulate_credit_factors(n_paths=100, n_steps=3, corr=corr_matrix, seed=42)
    assert factors.shape == (100, 3)
    empirical_corr = np.corrcoef(factors.T)
    expected_off_diag = 0.3**2
    np.testing.assert_allclose(empirical_corr[0, 1], expected_off_diag, atol=0.1)


# Test for simulate_term_structure_volatility computation
def test_simulate_term_structure_volatility_basic():
    times = np.array([0.0, 1.0, 2.0, 3.0])
    vol_short = 0.30
    vol_long = 0.10
    mean_reversion = 0.5
    vol_curve = simulate_term_structure_volatility(
        times, vol_short=vol_short, vol_long=vol_long, mean_reversion=mean_reversion
    )
    assert vol_curve.shape == times.shape
    assert np.isclose(vol_curve[0], vol_short)
    assert vol_curve[-1] < vol_curve[0]
    assert vol_curve[-1] > vol_long  # Not fully reverted yet
