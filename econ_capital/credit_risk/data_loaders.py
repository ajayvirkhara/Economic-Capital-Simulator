"""
Data loaders for the Credit Risk module.

Provides:
- Index-level free data loaders (via FRED API)
- Issuer-level loaders from curated CSV
- Dummy loaders for unit testing
"""

from __future__ import annotations
import pandas as pd
from pandas_datareader import data as web
from datetime import datetime

# --- A) Free credit indices (macro anchors) ---
def load_credit_indexes(start="2015-01-01", end=None) -> pd.DataFrame:
    end = end or datetime.today().strftime("%Y-%m-%d")
    series = {
        "IG_OAS_bps": "BAMLC0A0CM",
        "HY_OAS_bps": "BAMLH0A0HYM2",
        "BAA_yield_pct": "DBAA"
    }
    df = pd.concat({k: web.DataReader(v, "fred", start, end) for k, v in series.items()}, axis=1)
    return df.dropna()

# --- B) Issuer-level CSV loader ---
REQUIRED = ["counterparty","instrument_id","id_type","as_of_date",
            "measure","value","units","currency"]

def load_issuer_spreads_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["as_of_date"])
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df["measure"] = df["measure"].str.upper().str.strip()
    df["units"] = df["units"].str.lower().str.strip()
    if not set(df["units"]).issubset({"bps","%","per_year"}):
        raise ValueError("units must be bps, %, or per_year")
    if (df["value"] < 0).any():
        raise ValueError("Negative values found")
    return df

# --- C) Dummy loaders (for tests) ---
def load_dummy_credit_data() -> pd.DataFrame:
    data = {
        "counterparty": ["CPTY_A","CPTY_B"],
        "instrument_id": ["US1234567890","US0987654321"],
        "id_type": ["ISIN","ISIN"],
        "as_of_date": [pd.Timestamp("2024-12-31"), pd.Timestamp("2024-12-31")],
        "measure": ["CDS_SPREAD","BOND_OAS"],
        "value": [120, 180],
        "units": ["bps","bps"],
        "currency": ["USD","USD"]
    }
    return pd.DataFrame(data)
