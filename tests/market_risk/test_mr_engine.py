"""
Unit tests for econ_capital.market_risk.engine module.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from econ_capital.market_risk import MarketRiskEconomicCapital, load_dummy_positions

# --- Fixtures ---


@pytest.fixture
def dummy_engine():
    # Historical returns (factors as columns)
    dates = pd.date_range("2024-01-01", periods=100)
    rf = pd.DataFrame(
        np.random.randn(100, 3) * 0.01, index=dates, columns=["SPY", "EEM", "TLT"]
    )

    pos = pd.DataFrame(
        [[25_000_000, 15_000_000, -100_000_000]],
        columns=["SPY", "EEM", "TLT"],
        index=["MainPortfolio"],
    )

    engine = MarketRiskEconomicCapital(
        risk_factors=rf, positions=pos, config={"n_paths": 1000, "seed": 42, "use_full_revaluation": False}
    )
    return engine


@pytest.fixture
def positions_data():
    """Position dictionary mimicking the structure expected by _build_pricing_portfolio."""
    return {
        "Equity_Pos": {
            "type": "equity",
            "quantity": 100,
            "price": 150.0,
            "factor": "SPY",
        },
        "Bond_Pos": {
            "type": "bond",
            "notional": 1000000,
            "price": 100.0,
            "duration": 5.0,
            "convexity": 0.5,
            "factor": "TLT",
            "yield": 0.04,
        },
        "FX_Pos": {
            "type": "fxforward",
            "notional": 500000,
            "strike": 1.1,
            "maturity": 0.5,
            "factor": "EURUSD=X",
            "spot": 1.08,
            "r_dom": 0.03,
            "r_for": 0.01,
        },
        "Option_Pos": {
            "type": "option",
            "strike": 450,
            "maturity": 0.25,
            "option_type": "call",
            "vol": 0.2,
            "quantity": 10,
            "factor": "SPY",
            "spot": 440,
            "rf": 0.05,
        },
        "Swap_Pos": {
            "type": "swap",
            "notional": 2000000,
            "fixed_rate": 0.035,
            "tenor": 10,
            "factor": "LIBOR3M",
            "rate": 0.03,
            "freq": 2,
        },
        "Unknown_Pos": {
            "type": "commodity",
            "quantity": 50,
            "price": 2000.0,
            "factor": "GLD",
        },  # Fallback test
    }


@pytest.fixture
def complex_engine(positions_data):
    """Engine initialized with a DataFrame of positions to test matrix alignment."""
    dates = pd.date_range("2024-01-01", periods=10)
    rf = pd.DataFrame(
        np.random.randn(10, 4) * 0.01,
        index=dates,
        columns=["SPY", "TLT", "EURUSD=X", "LIBOR3M"],
    )

    # Convert dict to DF for the engine's __init__
    pos_df = pd.DataFrame(positions_data).T
    # Add dummy exposure columns for _build_exposures
    for f in rf.columns:
        pos_df[f] = 1.0
        pos_df[f"gamma_{f}"] = 0.1
        pos_df[f"vega_{f}"] = 0.05

    return MarketRiskEconomicCapital(
        risk_factors=rf, positions=pos_df, config={"use_full_revaluation": True}
    )


# --- Tests ---


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
        "n_paths": 2000,  # Small for test speed
        "horizon_days": 10,
        "scaling_days_year": 252,
        "df_t": 3.0,
        "use_full_revaluation": False,  # Disable full revaluation for simple delta test
        "stress_enabled": True,  # Explicitly enable stress testing
        "stress_shocks": {
            "SPY": -0.40,  # Expect negative P&L if long equity
            "TLT": 0.02,  # Positive shock (duration negative → loss if long bonds)
        },
    }

    # Fix seed for reproducibility in test
    np.random.seed(42)

    engine = MarketRiskEconomicCapital(
        risk_factors=dummy_rf, positions=positions, config=config
    )

    results = engine.run()

    stressed_var = results["stressed_var_1y_999"]
    stressed_es = results["stressed_es_1y_999"]
    baseline_var = results["var_1y_999"]

    # --- Basic presence checks ---
    assert "stressed_var_1y_999" in results
    assert "stressed_es_1y_999" in results
    assert results["stressed_var_1y_999"] is not None
    assert results["stressed_es_1y_999"] is not None

    # Stressed capital must be positive and meaningfully higher than baseline
    assert stressed_var > 0
    assert stressed_var > baseline_var * 1.3  # At least 30% uplift from large shock

    # ES should be strictly greater than VaR (fat tails → tail expectation worse)
    assert stressed_es > stressed_var

    # Reasonable upper bound to prevent explosion
    assert stressed_var < baseline_var * 15


def test_build_exposures(dummy_engine):
    delta, gamma, vega = dummy_engine._build_exposures()

    # delta should have shape (1 row × 3 factors)
    assert delta.shape == (1, 3)
    assert list(delta.columns) == ["SPY", "EEM", "TLT"]
    assert np.allclose(
        delta.iloc[0].values, [25_000_000, 15_000_000, -100_000_000], atol=1e-6
    )

    # gamma & vega should be zero (no quadratic/vega in dummy data)
    assert gamma.shape == (1, 3)
    assert np.all(gamma.values == 0)

    assert vega.shape == (1, 3)
    assert np.all(vega.values == 0)


@patch("econ_capital.market_risk.engine.ewma_cov")
def test_estimate_mu_cov_ewma(mock_ewma, dummy_engine):
    """Test EWMA cov estimation."""
    dummy_engine.config["cov_method"] = "EWMA"
    dummy_engine.config["fix_mean"] = True
    mock_ewma.return_value = pd.DataFrame(
        np.eye(3), index=["A", "B", "C"], columns=["A", "B", "C"]
    )
    mu, cov = dummy_engine._estimate_mu_cov()
    assert np.allclose(mu, 0.0)
    assert np.allclose(cov, np.eye(3))


@patch("econ_capital.market_risk.engine.garch_cov")
def test_estimate_mu_cov_garch(mock_garch, dummy_engine):
    """Test GARCH cov estimation."""
    dummy_engine.config["cov_method"] = "GARCH"
    mock_garch.return_value = pd.DataFrame(
        np.eye(3), index=["A", "B", "C"], columns=["A", "B", "C"]
    )
    _, cov = dummy_engine._estimate_mu_cov()
    mock_garch.assert_called_once()
    assert np.allclose(cov, np.eye(3))


def test_pnl_from_shocks_full_revaluation(dummy_engine):
    """Test full reval P&L (but since pricing_portfolio None, need to set)."""
    dummy_engine.use_full_revaluation = True
    dummy_engine.pricing_portfolio = MagicMock()
    dummy_engine.pricing_portfolio.revalue_all.return_value = np.random.randn(1000)
    shocks = np.random.randn(1000, 3)
    pnl_port, pnl_by_pos = dummy_engine._pnl_from_shocks_full_revaluation(shocks)
    assert pnl_port.shape == (1000,)
    assert isinstance(pnl_by_pos, dict)


@patch("econ_capital.market_risk.data_loaders.load_historical_returns")
def test_compute_historical_var(mock_load, dummy_engine):
    # Mock returns with enough history
    mock_hist = pd.DataFrame(np.random.randn(300, 3), columns=["SPY", "EEM", "TLT"])
    mock_load.return_value = mock_hist

    results = dummy_engine.compute_historical_var(lookback_days=252)

    assert "historical_var_1y_999" in results
    assert results["historical_lookback_days"] == 252


@patch("econ_capital.market_risk.engine.yf.Ticker")
def test_get_current_market_levels(mock_ticker, dummy_engine):
    """Test fetching current levels."""
    mock_hist = MagicMock()
    mock_hist.history.return_value = pd.DataFrame(
        {"Close": [100.0]}, index=[pd.Timestamp.now()]
    )
    mock_ticker.return_value = mock_hist
    levels = dummy_engine._get_current_market_levels()
    assert levels["SPY"] == 100.0  # Fallback for failed fetch


def test_build_pricing_portfolio_instrument_types(complex_engine, positions_data):
    """Tests building of the pricing portfolio for all instrument types."""
    # Manually trigger the builder using the dict-like structure it expects internally
    complex_engine.positions = positions_data
    portfolio = complex_engine._build_pricing_portfolio()

    # Verify we have all 6 positions
    assert len(portfolio.positions) == 6

    # Assert specific class types from marketrisk_pricing were used
    types = [type(p).__name__ for p in portfolio.positions]
    assert "EquityPosition" in types
    assert "BondPosition" in types
    assert "FXForward" in types
    assert "EuropeanOption" in types
    assert "InterestRateSwap" in types
    # Check fallback (Unknown_Pos becomes EquityPosition)
    assert types.count("EquityPosition") == 2


def test_pnl_from_shocks_full_revaluation_logic(complex_engine):
    """Tests mapping of shocks to specific position factors during full reval."""
    mock_portfolio = MagicMock()
    # Mock individual positions with factor attributes
    pos1 = MagicMock(underlying_factor="SPY")
    pos1.revalue.return_value = np.array([10.0])

    complex_engine.pricing_portfolio = mock_portfolio
    complex_engine.pricing_portfolio.positions = [pos1]
    complex_engine.pricing_portfolio.revalue_all.return_value = np.array([100.0])

    shocks = np.random.randn(1, 4)
    with patch.object(
        complex_engine,
        "_get_current_market_levels",
        return_value={"SPY": 100, "TLT": 100, "EURUSD=X": 1, "LIBOR3M": 0.05},
    ):
        pnl_port, pnl_by_pos = complex_engine._pnl_from_shocks_full_revaluation(shocks)

    assert pnl_port[0] == 100.0
    assert "SPY" in pnl_by_pos
    assert pnl_by_pos["SPY"][0] == 10.0


def test_compute_historical_var_error_handling(complex_engine):
    """Tests exception handling in historical VaR."""
    with patch(
        "econ_capital.market_risk.data_loaders.load_historical_returns",
        side_effect=Exception("Data Timeout"),
    ):
        results = complex_engine.compute_historical_var()
        assert results["historical_var_1y_999"] is None
        assert "Data Timeout" in results["historical_error"]


def test_compute_covar_metrics_missing_pnl(complex_engine):
    """Tests CoVaR safety check when PnL is not yet computed."""
    if hasattr(complex_engine, "pnl_by_pos"):
        del complex_engine.pnl_by_pos

    results = complex_engine._compute_covar_metrics()
    assert results == {}


def test_compute_covar_metrics_calculation(complex_engine):
    """Tests exception handling inside the CoVaR position loop."""
    complex_engine.pnl_port = np.random.randn(100)
    # Create one valid and one "broken" position PnL
    complex_engine.pnl_by_pos = {
        "ValidPos": np.random.randn(100),
        "BrokenPos": np.array([np.nan] * 100),  # Should trigger the try-except block
    }

    with patch(
        "econ_capital.market_risk.engine.compute_covar",
        side_effect=[(1.0, 0.5), Exception("Math Error")],
    ):
        results = complex_engine._compute_covar_metrics()

    assert "ValidPos" in results
    assert "BrokenPos" not in results


def test_get_current_market_levels_fallbacks(complex_engine):
    """Tests yfinance fallback when fetching fails."""
    with patch("yfinance.Ticker") as mock_ticker:
        # Simulate a network failure for one ticker
        mock_ticker.side_effect = Exception("Connection Error")
        levels = complex_engine._get_current_market_levels()

        # Verify it hit the fallback
        assert levels["SPY"] == 100.0
