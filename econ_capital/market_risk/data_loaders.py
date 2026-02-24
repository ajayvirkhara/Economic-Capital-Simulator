"""Data loaders for the Market Risk module.

This script provides:
- Historical returns loading from Yahoo Finance
- Dummy portfolio positions for testing
"""

from __future__ import annotations

from typing import Dict
import pandas as pd
import numpy as np
import fredapi as Fred
import os
from datetime import datetime
from typing import Any, Optional

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
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError(
            "FRED_API_KEY environment variable not set. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    fred = Fred(api_key=api_key)

    # Priority order remains the same
    if tickers is not None:
        selected = tickers
    else:
        yaml_cfg = load_market_yaml()
        selected = resolve_tickers(yaml_cfg)

    data = {}
    for col_name, series_id in selected.items():
        try:
            s = fred.get_series(
                series_id,
                observation_start=start,
                observation_end=end,
            )
            data[col_name] = s
        except Exception as e:
            print(f"Failed to fetch {series_id} ({col_name}): {e}")
            data[col_name] = pd.Series(dtype=float)  # fallback

    df = pd.DataFrame(data)

    # Fill gaps
    df = df.ffill().bfill()

    return df


def load_dummy_positions(
    positions: Dict[str, Dict[str, float]] | None = None,
) -> pd.DataFrame:
    """
    Returns a dictionary of positions in the format expected by the advanced
    _build_pricing_portfolio() method (supports equity, bond, etc.).

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
        -100_000_000,
        0,
        0,
        0,
        0,
        0,
        0,
    ]  # Rates (short-duration): £1B equiv DV01 scaled
    df["LQD"] = [0, 0, 0, -50_000_000, 0, 0, 0, 0, 0]  # IG Credit bonds: £250M
    df["HYG"] = [0, 0, 0, 0, -75_000_000, 0, 0, 0, 0]  # HY Credit bonds: £400M
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
    if source != "fred":
        raise ValueError("Only source='fred' is now supported (using fredapi)")

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError("FRED_API_KEY environment variable not set")

    fred = Fred(api_key=api_key)

    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    dfs = []
    for ticker in tickers:
        try:
            s = fred.get_series(
                ticker,
                observation_start=start_date,
                observation_end=end_date,
            )
            s.name = ticker
            dfs.append(s)
        except Exception as e:
            print(f"Failed to load {ticker}: {e}")

    if not dfs:
        raise ValueError("No FRED series loaded successfully")

    prices = pd.concat(dfs, axis=1).sort_index()
    returns = np.log(prices / prices.shift(1)).dropna()

    return returns


def convert_positions_to_dict(
    exposure_df: pd.DataFrame,
    current_levels: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Convert the exposure DataFrame (delta matrix) into a dict format
    suitable for full revaluation pricing.

    Infers instrument type and adds reasonable defaults for price/duration/etc.
    """
    if current_levels is None:
        current_levels = {
            "SPY": 580.0,
            "EFA": 85.0,
            "EEM": 45.0,
            "TLT": 92.0,
            "LQD": 108.0,
            "HYG": 78.0,
            "GLD": 240.0,
            "USO": 70.0,
            "EURUSD=X": 1.08,
            "GBPUSD=X": 1.27,
        }

    pos_dict = {}

    for pos_name in exposure_df.index:
        row = exposure_df.loc[pos_name]
        non_zero_factors = row[row != 0].index.tolist()

        if not non_zero_factors:
            continue  # skip zero-exposure positions

        # Assume single-factor position
        factor = non_zero_factors[0]
        exposure = row[factor]

        # Infer type and build metadata
        if pos_name in ["Equity_US", "Equity_Developed", "Equity_EM", "Gold", "Oil"]:
            pos_dict[pos_name] = {
                "type": "equity",
                "position_name": pos_name,
                "quantity": abs(exposure) / current_levels.get(factor, 100.0),
                "price": current_levels.get(factor, 100.0),
                "factor": factor,
            }

        elif pos_name in ["Rates", "Credit_IG", "Credit_HY"]:
            duration = (
                16.5 if pos_name == "Rates" else 7.8 if pos_name == "Credit_IG" else 4.2
            )
            convexity = (
                320.0
                if pos_name == "Rates"
                else 85.0
                if pos_name == "Credit_IG"
                else 45.0
            )
            yld = (
                0.042
                if pos_name == "Rates"
                else 0.052
                if pos_name == "Credit_IG"
                else 0.078
            )

            pos_dict[pos_name] = {
                "type": "bond",
                "position_name": pos_name,
                "notional": exposure,
                "price": current_levels.get(factor, 100.0),
                "duration": duration,
                "convexity": convexity,
                "factor": factor,
                "yield": yld,
            }

        elif pos_name in ["FX_EURUSD", "FX_GBPUSD"]:
            # Treat FX as linear for now
            pos_dict[pos_name] = {
                "type": "equity",
                "position_name": pos_name,
                "quantity": abs(exposure) / current_levels.get(factor, 1.0),
                "price": current_levels.get(factor, 1.0),
                "factor": factor,
            }

    return pos_dict


# Exported symbols
__all__ = [
    "load_real_risk_factors",
    "load_dummy_positions",
    "load_historical_returns",
    "convert_positions_to_dict",
]
