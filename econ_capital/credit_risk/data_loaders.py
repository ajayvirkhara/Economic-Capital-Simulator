"""
Data loaders for the Credit Risk module.

Provides:
- Index-level free data loaders (via FRED API)
- Issuer-level loaders from curated CSV
- Dummy loaders for unit testing
"""

from __future__ import annotations
from datetime import datetime

import pandas as pd
from fredapi import Fred

from econ_capital.utils import setup_logging

logger = setup_logging(__name__)

# Column schema for issuer-level credit CSVs
CSV_SCHEMA = [
    "counterparty",
    "instrument_id",
    "id_type",
    "as_of_date",
    "measure",
    "value",
    "units",
    "currency",
    "pd_annual",
]


# --- A) Free credit indices (macro anchors) ---
def load_credit_indexes(
    start: str = "2015-01-01", end: str | None = None
) -> pd.DataFrame:
    """
    Load key credit spread indices from FRED:
    - Investment Grade OAS (BAMLC0A0CM)
    - High Yield OAS (BAMLH0A0HYM2)
    - Moody's Seasoned Baa Corporate Bond Yield (DBAA)
    """
    end = end or datetime.today().strftime("%Y-%m-%d")

    api_key = "c6fc80debe9ed7ad0ee697eb52a86349"
    fred = Fred(api_key=api_key)

    series = {
        "IG_OAS_bps": "BAMLC0A0CM",
        "HY_OAS_bps": "BAMLH0A0HYM2",
        "BAA_yield_pct": "DBAA",
    }

    data = {}
    for col_name, series_id in series.items():
        try:
            s = fred.get_series(
                series_id,
                observation_start=start,
                observation_end=end,
            )
            data[col_name] = s
            logger.debug(f"Fetched {series_id} → {col_name} ({len(s)} obs)")
        except Exception as e:
            logger.error(f"Failed to fetch {series_id}: {e}")
            raise

    df = pd.DataFrame(data)
    if df.empty:
        logger.warning("No data fetched from FRED series - returning empty DataFrame")
    return df.dropna(how="all")


# --- B) Issuer-level CSV loader ---
REQUIRED = CSV_SCHEMA


def load_issuer_spreads_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["as_of_date"])
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df["measure"] = df["measure"].str.upper().str.strip()
    df["units"] = df["units"].astype(str).str.lower().str.strip()
    allowed_units = {"bps", "%", "per_year", "", "absolute"}
    if not set(df["units"]).issubset(allowed_units):
        raise ValueError(
            "units must be bps, %, per_year, or blank (for absolute values)"
        )

    def convert_value(row):
        if row["units"] == "bps":
            return row["value"] / 10000  # Convert bps to decimal (e.g., 120 -> 0.012)
        elif row["units"] == "%":
            return row["value"] / 100  # Convert % to decimal (e.g., 1.2 -> 0.012)
        elif row["units"] == "per_year":
            return row["value"]  # Assume already in decimal annual form
        else:  # Blank or "absolute"
            return row["value"]  # No conversion

    df["value"] = df.apply(convert_value, axis=1)
    if (df["value"] < 0).any():
        raise ValueError("Negative values found")
    return df


# --- C) Dummy loaders (for tests) ---
def load_dummy_credit_data() -> pd.DataFrame:
    data = {
        "counterparty": ["CPTY_A", "CPTY_B"],
        "instrument_id": ["US1234567890", "US0987654321"],
        "id_type": ["ISIN", "ISIN"],
        "as_of_date": [pd.Timestamp("2024-12-31"), pd.Timestamp("2024-12-31")],
        "measure": ["CDS_SPREAD", "BOND_OAS"],
        "value": [120, 180],
        "units": ["bps", "bps"],
        "currency": ["USD", "USD"],
        "pd_annual": ["0.01", "0.07"],
    }
    return pd.DataFrame(data)
