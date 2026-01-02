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


def test_stress_testing_returns_stressed_capital():
    """Test that stress_shocks config produces stressed VaR/ES in results."""
    # Load dummy positions (ensure some exposure to shocked factors)
    positions = load_dummy_positions()

    # Assume dummy risk factors exist
    dummy_rf = pd.DataFrame(
        np.random.normal(0, 0.01, (100, len(positions.columns))),
        columns=positions.columns,
    )

    # Custom config with stress shocks
    config = {
        "n_paths": 1000,  # Small for test speed
        "horizon_days": 10,
        "scaling_days_year": 252,
        "stress_shocks": {
            "SPY": -0.40,  # Expect negative P&L if long equity
            "TLT": 0.02,  # Positive shock (duration negative → loss if long bonds)
            "Unknown_Factor": -0.10,  # Should be ignored with warning
        },
    }

    engine = MarketRiskEconomicCapital(
        risk_factors=dummy_rf, positions=positions, config=config
    )

    results = engine.run()

    # Assert stressed fields are present and non-None
    assert "stressed_var_1y_999" in results
    assert results["stressed_var_1y_999"] is not None
    assert results["stressed_var_1y_999"] == results["stressed_es_1y_999"]
    assert (
        results["stressed_var_1y_999"] > 100_000
    )  # Significant impact from 40% SPY drop on £5M position
