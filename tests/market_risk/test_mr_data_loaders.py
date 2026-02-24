"""
Unit tests for econ_capital.market_risk.data_loaders module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from econ_capital.market_risk.data_loaders import (
    load_real_risk_factors,
    load_dummy_positions,
    load_historical_returns,
)


# --- Tests for load_real_risk_factors ---


@patch("econ_capital.market_risk.data_loaders.Fred")
def test_load_real_risk_factors_explicit_tickers(mock_fred_class):
    """Test loading with explicit FRED series IDs."""
    mock_fred = mock_fred_class.return_value

    dates = pd.date_range("2020-12-01", periods=11, freq="B")

    # Mock two example FRED series
    mock_dgs10 = pd.Series(np.linspace(1.5, 1.8, 11), index=dates, name="DGS10")
    mock_vix = pd.Series(np.random.uniform(15, 35, 11), index=dates, name="VIXCLS")

    def get_series_side_effect(series_id, observation_start=None, observation_end=None):
        if series_id == "DGS10":
            return mock_dgs10
        if series_id == "VIXCLS":
            return mock_vix
        raise ValueError(f"Unexpected FRED series: {series_id}")

    mock_fred.get_series.side_effect = get_series_side_effect

    tickers = {"10Y_Yield": "DGS10", "VIX": "VIXCLS"}  # real FRED IDs → column names
    levels = load_real_risk_factors(
        start="2020-12-01", end="2020-12-20", tickers=tickers
    )

    assert isinstance(levels, pd.DataFrame)
    assert list(levels.columns) == ["10Y_Yield", "VIX"]
    assert levels.shape == (11, 2)  # levels, no pct_change drop
    assert mock_fred.get_series.call_count == 2


@patch("econ_capital.market_risk.data_loaders.load_market_yaml")
@patch("econ_capital.market_risk.data_loaders.Fred")
def test_load_real_risk_factors_yaml_override(mock_fred_class, mock_yaml):
    """Test loading with YAML overrides."""
    mock_yaml.return_value = {
        "tickers": {
            "10Y_Yield": "DGS10",
            "Unemployment": "UNRATE",
        }  # FIXED: Column → FRED ID
    }

    mock_fred = mock_fred_class.return_value

    dates = pd.date_range("2020-12-01", periods=11, freq="B")
    mock_dgs10 = pd.Series(np.linspace(1.5, 1.8, 11), index=dates, name="DGS10")
    mock_unrate = pd.Series(np.linspace(5.0, 4.5, 11), index=dates, name="UNRATE")

    def get_series_side_effect(series_id, observation_start=None, observation_end=None):
        if series_id == "DGS10":
            return mock_dgs10
        if series_id == "UNRATE":
            return mock_unrate
        raise ValueError(f"Unexpected FRED series: {series_id}")

    mock_fred.get_series.side_effect = get_series_side_effect

    levels = load_real_risk_factors(start="2020-01-01", end="2021-01-01")

    assert list(levels.columns) == ["10Y_Yield", "Unemployment"]
    assert levels.shape[0] > 0
    assert mock_fred.get_series.call_count == 2
    mock_yaml.assert_called_once()


@patch("econ_capital.market_risk.data_loaders.Fred")
def test_load_real_risk_factors_fallback(mock_fred_class):
    """Test basic fallback / error handling."""
    mock_fred = mock_fred_class.return_value

    dates = pd.date_range("2020-12-01", periods=11, freq="B")
    mock_series = pd.Series(np.random.randn(11), index=dates, name="DGS10")

    mock_fred.get_series.return_value = mock_series

    tickers = {"10Y_Yield": "DGS10"}
    levels = load_real_risk_factors(
        start="2020-01-01", end="2021-01-01", tickers=tickers
    )

    assert "10Y_Yield" in levels.columns
    assert levels.shape[0] > 0


# --- Tests for load_dummy_positions ---


def test_load_dummy_positions_explicit():
    """Test with explicit positions argument."""
    positions = {"Equity_US": {"SPY": 1000000}, "Rates": {"TLT": -500000}}
    df = load_dummy_positions(positions)

    assert isinstance(df, pd.DataFrame)
    assert df.loc["Equity_US", "SPY"] == 1000000
    assert df.loc["Rates", "TLT"] == -500000


@patch("econ_capital.market_risk.data_loaders.load_market_yaml")
def test_load_dummy_positions_yaml_override(mock_yaml):
    """Test with YAML positions override."""
    mock_yaml.return_value = {
        "positions": {"Equity_US": {"SPY": 2000000}, "Rates": {"TLT": -1000000}}
    }
    df = load_dummy_positions()

    assert df.loc["Equity_US", "SPY"] == 2000000
    assert df.loc["Rates", "TLT"] == -1000000


def test_load_dummy_positions_hardcoded_defaults():
    """Test hardcoded default positions."""
    df = load_dummy_positions()

    assert "SPY" in df.columns
    assert df.loc["Equity_US", "SPY"] == 250_000_000
    assert df.loc["Rates", "TLT"] == -100_000_000
    assert df.shape == (9, 9)


# --- Tests for load_historical_returns ---


@patch("econ_capital.market_risk.data_loaders.Fred")
def test_load_historical_returns_fred(mock_fred_class):
    """Test log returns from FRED series."""
    mock_fred = mock_fred_class.return_value

    dates = pd.date_range("2020-01-01", periods=20, freq="B")
    mock_series = pd.Series(np.linspace(1.5, 2.0, 20), index=dates, name="DGS10")

    mock_fred.get_series.return_value = mock_series

    returns = load_historical_returns(
        tickers=["DGS10"],
        start_date="2020-01-01",
        source="fred",
    )

    assert returns.shape == (19, 1)
    assert returns.columns.tolist() == ["DGS10"]


def test_load_historical_returns_unknown_source():
    """Test error for unknown source."""
    with pytest.raises(ValueError, match="Only source='fred'"):
        load_historical_returns(tickers=["DGS10"], source="yahoo")
