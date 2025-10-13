"""Data loaders for the Market Risk module.

This script provides:
- Historical returns loading from Yahoo Finance
- Dummy portfolio positions for testing
"""

from __future__ import annotations

from typing import Dict
import pandas as pd
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
    df["SPY"] = [5_000_000, 0, 0, 0, 0, 0, 0, 0, 0]
    df["EEM"] = [0, 2_000_000, 0, 0, 0, 0, 0, 0, 0]
    df["TLT"] = [0, 0, -10_000.0, 0, 0, 0, 0, 0, 0]
    df["LQD"] = [0, 0, 0, -3_000.0, 0, 0, 0, 0, 0]
    df["HYG"] = [0, 0, 0, 0, -4_000.0, 0, 0, 0, 0]
    df["GLD"] = [0, 0, 0, 0, 0, 1_000_000, 0, 0, 0]
    df["USO"] = [0, 0, 0, 0, 0, 0, 500_000, 0, 0]
    df["EURUSD=X"] = [0, 0, 0, 0, 0, 0, 0, 2_000_000, 0]
    df["GBPUSD=X"] = [0, 0, 0, 0, 0, 0, 0, 0, 1_500_000]

    return df.fillna(0.0)


# Exported symbols
__all__ = ["load_real_risk_factors", "load_dummy_positions"]
