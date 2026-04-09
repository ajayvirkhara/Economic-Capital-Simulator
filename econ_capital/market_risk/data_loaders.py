"""Data loaders for the Market Risk module.

This script provides:
- Historical returns loading from Yahoo Finance
- Dummy portfolio positions for testing
"""

from __future__ import annotations

from typing import Dict
import pandas as pd
import numpy as np
from fredapi import Fred
import os
from datetime import datetime
from typing import Any, Optional

from .config import load_market_yaml
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))


def load_real_risk_factors(
    start: str = "2020-01-01",
    end: str = "2025-01-01",
    tickers: Dict[str, str] | None = None,
) -> pd.DataFrame:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError("FRED_API_KEY environment variable not set.")

    fred = Fred(api_key=api_key)

    # Use these specific FRED IDs which actually exist in their database
    fred_map = {
        "SPY": "SP500",  # S&P 500
        "EFA": "MSCI_NQ_W_I_U",  # Placeholder for EFA if available, or use 'SP500'
        "EEM": "MSCI_NQ_E_M_I",  # Placeholder for EEM
        "TLT": "LTGOVTBD",  # Long Term Gov Bonds
        "LQD": "BAA10Y",  # Corporate Bond Spread
        "HYG": "BAMLH0A0HYM2",  # High Yield Master II
        "GLD": "GOLDAMGBD228NLBM",  # Gold Price
        "USO": "DCOILWTICO",  # WTI Oil Price
        "EURUSD=X": "DEXUSEU",  # EUR/USD
        "GBPUSD=X": "DEXUSUK",  # GBP/USD
    }

    # Priority: explicit tickers > YAML > hardcoded defaults
    if tickers is None:
        yaml_cfg = load_market_yaml()
        tickers = yaml_cfg.get("tickers", fred_map) if yaml_cfg else fred_map

    data = {}
    for col_name, fred_id in tickers.items():
        try:
            print(f"Fetching {col_name} using FRED ID: {fred_id}...")
            s = fred.get_series(
                fred_id,
                observation_start=start,
                observation_end=end,
            )
            data[col_name] = s
        except Exception as e:
            print(f"Failed to fetch {fred_id} ({col_name}): {e}")
            data[col_name] = pd.Series(dtype=float)

    df = pd.DataFrame(data)
    df = df.dropna(axis=1, how="all")

    # Ensure DatetimeIndex before resampling
    df.index = pd.to_datetime(df.index)

    # Resample to business daily, forward-fill gaps
    df = df.resample("B").last().ffill()

    # Log returns
    returns = np.log(df / df.shift(1)).dropna()
    returns = returns.loc[:, returns.std() > 1e-8]

    if returns.empty or returns.shape[1] == 0:
        raise ValueError(
            "No valid return series after cleaning. Check FRED data fetch."
        )

    return returns


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
    source: str = "fred",
) -> pd.DataFrame:
    if source != "fred":
        raise ValueError("Only source='fred' is now supported.")

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError("FRED_API_KEY environment variable not set")

    fred = Fred(api_key=api_key)
    fred_id_map = {
        "SPY": "SP500",
        "EFA": "MSCI_NQ_W_I_U",
        "EEM": "MSCI_NQ_E_M_I",
        "TLT": "LTGOVTBD",
        "LQD": "BAA10Y",
        "HYG": "BAMLH0A0HYM2",
        "GLD": "GOLDAMGBD228NLBM",
        "USO": "DCOILWTICO",
        "EURUSD=X": "DEXUSEU",
        "GBPUSD=X": "DEXUSUK",
    }

    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    dfs = []
    for ticker in tickers:
        fred_id = fred_id_map.get(ticker, ticker)
        try:
            s = fred.get_series(
                fred_id, observation_start=start_date, observation_end=end_date
            )
            s.name = ticker
            dfs.append(s)
        except Exception as e:
            print(f"Failed to load {fred_id}: {e}")

    if not dfs:
        raise ValueError("No FRED series loaded successfully")

    # --- SINGLE PASS CLEANING ---
    prices = pd.concat(dfs, axis=1).sort_index()
    prices = prices.ffill().dropna()  # Fill gaps first

    # Ensure all prices are positive to avoid log(0) which creates Inf
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.clip(lower=1e-6)

    # Calculate Log Returns
    returns = np.log(prices / prices.shift(1)).dropna()

    # Hard cap at 50% daily move
    returns = returns.clip(lower=-0.55, upper=0.50)

    print(f"DEBUG: Max daily return: {returns.max().max():.4%}")
    print(f"DEBUG: Data shape: {returns.shape}")

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
