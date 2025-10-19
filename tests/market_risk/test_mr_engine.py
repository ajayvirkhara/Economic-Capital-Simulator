"""
Unit tests for econ_capital.market_risk.engine module.
"""

import numpy as np
import pandas as pd

from econ_capital.market_risk import MarketRiskEconomicCapital, load_dummy_positions


def test_engine_run_outputs():
    """Tests that the full Market Risk Economic Capital engine runs successfully and produces all expected keys with finite numeric values."""
    # Dummy factor returns (100 days × 3 assets)
    np.random.seed(123)
    rf = pd.DataFrame(np.random.randn(100, 3) * 0.01, columns=["SPY", "EEM", "TLT"])
    pos = load_dummy_positions()

    # Initialize and run the engine
    engine = MarketRiskEconomicCapital(
        risk_factors=rf, positions=pos, config={"n_paths": 1000, "seed": 42}
    )
    results = engine.run()

    # Define the required output keys
    expected_keys = {
        "var_10d_999",
        "es_10d_999",
        "var_1y_999",
        "es_1y_999",
        "capital_breakdown",
    }
    assert expected_keys.issubset(results.keys()), "Missing expected output keys"

    # Check that the core VaR and ES metrics are finite numbers
    for key in ["var_10d_999", "es_10d_999", "var_1y_999", "es_1y_999"]:
        assert np.isfinite(results[key]), f"{key} is not finite"
