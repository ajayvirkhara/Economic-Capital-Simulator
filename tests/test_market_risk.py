"""Unit tests for Market Risk Economic Capital module."""

import numpy as np
import pandas as pd

from econ_capital.market_risk.market_risk import (
    ewma_cov,
    mv_t_draws,
    MarketRiskEconomicCapital,
)
from econ_capital.market_risk.data_loaders import load_dummy_positions


def test_ewma_cov_symmetry():
    """EWMA covariance matrix should be symmetric."""
    np.random.seed(42)
    returns = pd.DataFrame(np.random.randn(100, 3), columns=["A", "B", "C"])
    cov = ewma_cov(returns, 0.97)
    assert np.allclose(cov, cov.T), "Covariance matrix is not symmetric"


def test_mv_t_draws_shape():
    """Multivariate Student-t draws should return the right shape."""
    n, k = 500, 3
    mu = np.zeros(k)
    cov = np.eye(k)
    df = 7
    rng = np.random.default_rng(123)

    draws = mv_t_draws(n, mu, cov, df, rng)
    assert draws.shape == (n, k), "Draws have wrong shape"


def test_engine_run_outputs():
    """Engine run should return expected keys with numeric values."""
    # Simulate dummy risk factors (100 days of returns for 3 assets)
    np.random.seed(123)
    rf = pd.DataFrame(np.random.randn(100, 3) * 0.01, columns=["SPY", "EEM", "TLT"])
    pos = load_dummy_positions()

    engine = MarketRiskEconomicCapital(rf, pos, config={"n_paths": 1000, "seed": 42})
    results = engine.run()

    expected_keys = {
        "var_10d_999",
        "es_10d_999",
        "var_1y_999",
        "es_1y_999",
        "capital_breakdown",
    }
    assert expected_keys.issubset(results.keys()), "Missing expected output keys"

    # Check that numbers are finite
    for key in ["var_10d_999", "es_10d_999", "var_1y_999", "es_1y_999"]:
        assert np.isfinite(results[key]), f"{key} is not finite"
