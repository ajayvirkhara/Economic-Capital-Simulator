"""
Unit tests for the Market Risk Economic Capital module.

Covers:
- Covariance estimators (ewma_cov)
- Shock generator (mv_t_draws)
- Full engine (MarketRiskEconomicCapital.run)
"""

import numpy as np
import pandas as pd

from econ_capital.market_risk.covariance import ewma_cov
from econ_capital.market_risk.shocks import mv_t_draws
from econ_capital.market_risk.engine import MarketRiskEconomicCapital
from econ_capital.market_risk.data_loaders import load_dummy_positions


def test_ewma_cov_symmetry():
    """EWMA covariance matrix should be symmetric."""
    np.random.seed(42)
    returns = pd.DataFrame(np.random.randn(100, 3), columns=["A", "B", "C"])

    cov = ewma_cov(returns, lamb=0.97)

    assert np.allclose(cov, cov.T), "Covariance matrix is not symmetric"


def test_mv_t_draws_shape():
    """Multivariate Student-t draws should return the correct shape."""
    n, k = 500, 3
    mu = np.zeros(k)
    cov = np.eye(k)
    df = 7
    rng = np.random.default_rng(123)

    draws = mv_t_draws(n, mu, cov, df, rng)

    assert draws.shape == (n, k), "Draws have wrong shape"


def test_engine_run_outputs():
    """Engine run should return expected result keys with finite values."""
    # Dummy factor returns (100 days × 3 assets)
    np.random.seed(123)
    rf = pd.DataFrame(np.random.randn(100, 3) * 0.01, columns=["SPY", "EEM", "TLT"])
    pos = load_dummy_positions()

    engine = MarketRiskEconomicCapital(
        risk_factors=rf, positions=pos, config={"n_paths": 1000, "seed": 42}
    )
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
