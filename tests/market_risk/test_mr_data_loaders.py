"""
Unit tests for econ_capital.market_risk.data_loaders module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, ANY

from econ_capital.market_risk.data_loaders import (
    load_real_risk_factors,
    load_dummy_positions,
    load_historical_returns,
)


# --- Tests for load_real_risk_factors ---


@patch("econ_capital.market_risk.data_loaders.yf.download")
def test_load_real_risk_factors_explicit_tickers(mock_download):
    """Test loading with explicit tickers argument."""
    dates = pd.date_range("2020-12-01", periods=11)

    mock_adj = pd.DataFrame(
        np.random.randn(11, 2) * 5 + 100, index=dates, columns=["SPY", "TLT"]
    )
    mock_close = mock_adj * 1.002  # slightly different just to be realistic

    mock_data = pd.concat(
        {
            "Adj Close": mock_adj,
            "Close": mock_close,
        },
        axis=1,
        names=["Price Type", "Ticker"],
    ).sort_index(axis=1)  # NO swaplevel

    mock_download.return_value = mock_data

    tickers = {"SPY": "Equity_US", "TLT": "Rates"}
    returns = load_real_risk_factors(
        start="2020-12-01", end="2020-12-20", tickers=tickers
    )

    assert isinstance(returns, pd.DataFrame)
    assert list(returns.columns) == ["SPY", "TLT"]
    assert returns.shape == (10, 2)
    mock_download.assert_called_once_with(
        ["SPY", "TLT"], start="2020-12-01", end="2020-12-20", progress=False
    )


@patch("econ_capital.market_risk.data_loaders.load_market_yaml")
@patch("econ_capital.market_risk.data_loaders.yf.download")
def test_load_real_risk_factors_yaml_override(mock_download, mock_yaml):
    """Test loading with YAML overrides."""
    mock_yaml.return_value = {"tickers": {"SPY": "Equity_US", "EEM": "Equity_EM"}}

    dates = pd.date_range("2020-12-01", periods=11)
    mock_adj = pd.DataFrame(
        np.random.randn(11, 2) * 5 + 100, index=dates, columns=["SPY", "EEM"]
    )
    mock_close = mock_adj * 1.0015

    mock_data = pd.concat(
        {"Adj Close": mock_adj, "Close": mock_close},
        axis=1,
        names=["Price Type", "Ticker"],
    ).sort_index(axis=1)  # NO swaplevel

    mock_download.return_value = mock_data

    returns = load_real_risk_factors(start="2020-01-01", end="2021-01-01")

    assert list(returns.columns) == ["SPY", "EEM"]
    assert returns.shape == (10, 2)
    mock_yaml.assert_called_once()


@patch("econ_capital.market_risk.data_loaders.yf.download")
def test_load_real_risk_factors_fallback_close(mock_download):
    """Test fallback to 'Close' if 'Adj Close' not available."""
    dates = pd.date_range("2020-12-01", periods=11)
    mock_close = pd.DataFrame(
        np.random.randn(11, 1) * 5 + 100, index=dates, columns=["SPY"]
    )

    mock_data = pd.concat(
        {"Close": mock_close}, axis=1, names=["Price Type", "Ticker"]
    ).sort_index(axis=1)  # NO swaplevel

    mock_download.return_value = mock_data

    tickers = {"SPY": "Equity_US"}
    returns = load_real_risk_factors(
        start="2020-01-01", end="2021-01-01", tickers=tickers
    )

    assert "SPY" in returns.columns
    assert returns.shape[0] == 10  # pct_change drops one


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
    assert df.shape == (9, 9)  # From the hardcoded idx and columns


# --- Tests for load_historical_returns ---


@patch("econ_capital.market_risk.data_loaders.yf.download")
def test_load_historical_returns_yahoo(mock_download):
    """Test Yahoo source."""
    dates = pd.date_range("2020-12-01", periods=11)
    mock_adj = pd.DataFrame(
        np.random.randn(11, 2) * 5 + 100, index=dates, columns=["SPY", "TLT"]
    )
    mock_close = mock_adj * 1.0015

    mock_data = pd.concat(
        {"Adj Close": mock_adj, "Close": mock_close},
        axis=1,
        names=["Price Type", "Ticker"],
    ).sort_index(axis=1)  # NO swaplevel

    mock_download.return_value = mock_data

    returns = load_historical_returns(
        tickers=["SPY", "TLT"],
        start_date="2020-01-01",
        end_date="2021-01-01",
        source="yahoo",
    )

    assert returns.shape == (10, 2)
    assert np.allclose(returns, np.log(mock_adj / mock_adj.shift(1)).dropna())


@patch("pandas_datareader.data.DataReader")
def test_load_historical_returns_fred(mock_datareader):
    """Test FRED source."""
    dates = pd.date_range("2020-01-01", periods=20, freq="B")
    mock_df = pd.DataFrame(
        np.abs(np.random.randn(20, 1)) * 0.01 + 0.02,  # positive values
        index=dates,
        columns=["DGS10"],
    )
    mock_datareader.return_value = mock_df

    returns = load_historical_returns(
        tickers=["DGS10"], start_date="2020-01-01", source="fred"
    )

    expected_rows = len(mock_df) - 1
    assert returns.shape == (expected_rows, 1)
    mock_datareader.assert_called_once_with("DGS10", "fred", "2020-01-01", ANY)


def test_load_historical_returns_unknown_source():
    """Test error for unknown source."""
    with pytest.raises(ValueError, match="Unknown source"):
        load_historical_returns(tickers=["SPY"], source="invalid")
