"""
Unit tests for econ_capital.market_risk.covariance module.
"""

import numpy as np
import pandas as pd

from econ_capital.market_risk.covariance import ewma_cov


def test_ewma_cov_symmetry():
    """Tests that the EWMA covariance matrix is symmetric, as expected for a covariance measure."""
    np.random.seed(42)
    # Create dummy returns data
    returns = pd.DataFrame(np.random.randn(100, 3), columns=["A", "B", "C"])

    cov = ewma_cov(returns, lamb=0.97)

    # Check for symmetry by comparing the matrix to its transpose
    assert np.allclose(cov, cov.T), "Covariance matrix is not symmetric"
