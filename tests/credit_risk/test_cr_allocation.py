"""
Unit tests for functions in econ_capital.credit_risk.allocation, verifying the
correct calculation of marginal contribution (Euler allocation) and the
allocation of total Economic Capital (EC) based on simulated losses.
"""

import numpy as np

# Import the functions under test from the module.
from econ_capital.credit_risk.allocation import marginal_contribution, allocate_ec


def test_marginal_contribution_shape_and_sum():
    # Tests that 'marginal_contribution' returns an array of the correct shape
    # (n_entities) and that the fractional contributions sum up to 1 (100%).

    # Set seed for reproducibility and create a variation factor to ensure
    # non-zero covariance (Euler allocation requirement).

    # Create a random factor (100 sims, 1 factor) to multiply the base losses
    np.random.seed(42)  # fixed seed for reproducability
    factor = np.random.uniform(0.5, 1.5, 100)[:, np.newaxis]
    base_losses = np.array([10.0, 20.0, 30.0])
    # Loss data: (100 sims, 3 entities). Shape (100, 3).
    loss_data = base_losses * factor

    # Loss data: (100 sims, 3 entities). Shape (100, 3).
    # This shape is used because the function's internal logic transposes it to (3, 100).
    loss_data = base_losses * factor

    # marginal_contribution only accepts the 2D loss array.
    contrib = marginal_contribution(loss_data)

    # Assertion 1: Check that the output shape is (n_entities,).
    assert contrib.shape == (3,)

    # Assertion 2: Check that the fractional contributions sum to 1.0.
    np.testing.assert_almost_equal(contrib.sum(), 1.0, decimal=6)

    # Check the actual fractional contributions (which should be 10/60, 20/60, 30/60).
    expected_fracs = np.array([1 / 6, 2 / 6, 3 / 6])
    np.testing.assert_allclose(contrib, expected_fracs, atol=1e-6)


def test_allocate_ec_proportional():
    # Tests that 'allocate_ec' correctly uses the fractional contributions to
    # allocate a scalar total EC amount back to individual entities.

    total_ec = 60.0  # Total Economic Capital to be allocated.

    # Use the same loss data generation to ensure a non-degenerate calculation.
    np.random.seed(42)  # fixed seed for reproducability
    factor = np.random.uniform(0.5, 1.5, 100)[:, np.newaxis]
    base_losses = np.array([10.0, 20.0, 30.0])
    # Loss data: (100 sims, 3 entities). Shape (100, 3).
    loss_data = base_losses * factor

    # allocate_ec takes (total_ec, losses).
    allocated = allocate_ec(total_ec, loss_data)

    # The expected allocation is based on the input proportionality:
    # 60 * (1/6, 2/6, 3/6) = (10, 20, 30).
    expected = np.array([10.0, 20.0, 30.0])

    # Assert that the calculated allocation matches the expected allocation.
    np.testing.assert_almost_equal(allocated, expected, decimal=6)
