"""
Unit tests for functions in econ_capital.credit_risk.wwr, specifically verifying the
'adjust_for_wwr' function which modifies exposure (Expected Loss or similar)
to account for Wrong-Way Risk (WWR).
"""

import numpy as np
from econ_capital.credit_risk.wwr import (
    adjust_for_wwr,
)  # Import the function under test


def test_wwr_linear_adjustment():
    # Tests that the 'adjust_for_wwr' function applies a plausible linear adjustment
    # to the exposure based on correlation, verifying shape, non-negativity,
    # and the expected directional effect of increased correlation.

    n_paths = 100
    sensitivity = 0.5

    # Create a 2D exposures array (100 paths, 3 entities) from a 1D base.
    base_exp = np.array([100.0, 200.0, 300.0])
    exposures = np.tile(base_exp, (n_paths, 1))

    # Create a 2D credit factors array (100 paths, 1 factor)
    # This represents the systematic credit factor driving WWR.
    credit_factors_array = np.full((n_paths, 1), 0.5)

    # Calculate the WWR-adjusted exposure using 2D inputs.
    adj = adjust_for_wwr(exposures, credit_factors_array, sensitivity=sensitivity)

    # Assertion 1: Check that the output shape matches the input shape (n_paths, n_entities).
    assert adj.shape == exposures.shape
    assert np.all(adj >= 0)

    # Test the directional effect: higher correlation factor should increase exposure.
    credit_factors_high = np.full((n_paths, 1), 0.9)
    adj_high = adjust_for_wwr(exposures, credit_factors_high, sensitivity=sensitivity)

    assert np.all(adj_high >= adj)
