"""Data loaders for the Market Risk module.

This script provides:
- Historical returns loading from Yahoo Finance
- Dummy portfolio positions for testing
"""

from __future__ import annotations

from typing import Dict
import pandas as pd
import numpy as np
import yfinance as yf

from .config import load_market_yaml, resolve_tickers


def load_real_risk_factors(
    start: str = "2020-01-01",
    end: str = "2025-01-01",
    tickers: Dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Fetch historical returns for multi-asset risk factors from Yahoo Finance.

    Priority order for tickers:
        1. Explicit tickers argument (highest priority)
        2. YAML overrides in config/market_config.yaml
        3. Python DEFAULT_TICKERS (fallback)

    Args:
        start (str): Start date for data download (YYYY-MM-DD).
        end (str): End date for data download (YYYY-MM-DD).
        tickers (Dict[str, str], optional): Explicit mapping of ticker symbols to asset class names.

    Returns:
        pd.DataFrame: Daily returns of each risk factor.
    """
    # Step 1: Explicit tickers override everything
    if tickers is not None:
        selected = tickers
    else:
        # Step 2: YAML overrides, else fallback to defaults
        yaml_cfg = load_market_yaml()
        selected = resolve_tickers(yaml_cfg)

    # Step 3: Fetch data
    data = yf.download(list(selected.keys()), start=start, end=end, progress=False)

    # Use Adjusted Close if available, else fallback to Close
    prices = (
        data["Adj Close"].dropna() if "Adj Close" in data else data["Close"].dropna()
    )
    returns = prices.pct_change().dropna()
    returns.columns = list(selected.keys())
    return returns


def load_dummy_positions(
    positions: Dict[str, Dict[str, float]] | None = None,
) -> pd.DataFrame:
    """
    Construct a dummy portfolio with exposures across equities, credit, commodities, and FX.

    Priority order for positions:
        1. Explicit `positions` argument (highest priority)
        2. YAML overrides in config/market_config.yaml under "positions"
        3. Hardcoded defaults (fallback)

    Args:
        positions (Dict[str, Dict[str, float]], optional):
            Explicit positions, keyed by risk factor and instrument.

    Returns:
        pd.DataFrame: Positions DataFrame with asset exposures.
    """
    # Step 1: Explicit positions override everything
    if positions is not None:
        return pd.DataFrame.from_dict(positions, orient="index").fillna(0.0)

    # Step 2: YAML overrides
    yaml_cfg = load_market_yaml()
    if yaml_cfg and "positions" in yaml_cfg:
        return pd.DataFrame.from_dict(yaml_cfg["positions"], orient="index").fillna(0.0)

    # Step 3: Hardcoded defaults (for demo/testing)
    idx = [
        "Equity_US",
        "Equity_EM",
        "Rates",
        "Credit_IG",
        "Credit_HY",
        "Gold",
        "Oil",
        "FX_EURUSD",
        "FX_GBPUSD",
    ]

    df = pd.DataFrame(index=idx)
    df["SPY"] = [250_000_000, 0, 0, 0, 0, 0, 0, 0, 0]  # US Equity: £250M
    df["EEM"] = [0, 150_000_000, 0, 0, 0, 0, 0, 0, 0]  # EM Equity: £150M
    df["TLT"] = [
        0,
        0,
        -1_000_000_000,
        0,
        0,
        0,
        0,
        0,
        0,
    ]  # Rates (long-duration): £1B equiv DV01 scaled
    df["LQD"] = [0, 0, 0, -250_000_000, 0, 0, 0, 0, 0]  # IG Credit bonds: £250M
    df["HYG"] = [0, 0, 0, 0, -400_000_000, 0, 0, 0, 0]  # HY Credit: £400M
    df["GLD"] = [0, 0, 0, 0, 0, 100_000_000, 0, 0, 0]  # Gold: £100M
    df["USO"] = [0, 0, 0, 0, 0, 0, 50_000_000, 0, 0]  # Oil: £50M
    df["EURUSD=X"] = [0, 0, 0, 0, 0, 0, 0, 250_000_000, 0]  # FX EURUSD: £250M
    df["GBPUSD=X"] = [0, 0, 0, 0, 0, 0, 0, 0, 200_000_000]  # FX GBPUSD: £200M

    return df.fillna(0.0)


def load_historical_returns(
    tickers: list[str],
    start_date: str = "2020-01-01",
    end_date: str = None,
    source: str = "yahoo",
) -> pd.DataFrame:
    """
    Load historical returns for factor universe.

    Parameters
    ----------
    tickers : list[str]
        Risk factor tickers (e.g., ['SPY', 'TLT', 'GLD'])
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date (default: today)
    source : str
        Data source: 'yahoo' or 'fred'

    Returns
    -------
    returns : pd.DataFrame
        Daily log returns, columns = tickers
    """
    import yfinance as yf
    from datetime import datetime

    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    if source == "yahoo":
        # Download adjusted close prices
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        prices = data["Adj Close"]

        # Compute log returns
        returns = np.log(prices / prices.shift(1)).dropna()

        return returns

    elif source == "fred":
        # Use pandas_datareader for FRED data
        from pandas_datareader import data as web

        dfs = []
        for ticker in tickers:
            try:
                df = web.DataReader(ticker, "fred", start_date, end_date)
                dfs.append(df)
            except Exception as e:
                print(f"Failed to load {ticker}: {e}")

        prices = pd.concat(dfs, axis=1)
        prices.columns = tickers
        returns = np.log(prices / prices.shift(1)).dropna()

        return returns

    else:
        raise ValueError(f"Unknown source: {source}")


# Exported symbols
__all__ = ["load_real_risk_factors", "load_dummy_positions", "load_historical_returns"]
