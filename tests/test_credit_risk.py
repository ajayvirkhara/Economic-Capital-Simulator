"""
Unit tests for credit_risk data_loaders
"""

import os
import tempfile
import pandas as pd
import pytest

from econ_capital.credit_risk.data_loaders import (
    load_dummy_credit_data,
    load_issuer_spreads_csv,
    load_credit_indexes,
    CSV_SCHEMA,
)


def test_load_dummy_credit_data():
    """Dummy loader should return a 2x8 DataFrame with expected columns."""
    df = load_dummy_credit_data()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 8)
    assert list(df.columns) == CSV_SCHEMA
    assert (df["units"] == "bps").all()


def test_load_issuer_spreads_csv_valid():
    """CSV loader should parse and normalize a valid file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w") as tmp_file:
        tmp_file.write(
            "counterparty,instrument_id,id_type,as_of_date,measure,value,units,currency\n"
            "CPTY_X,US1111111111,ISIN,2024-12-31, cds_spread ,150, BPS ,USD\n"
        )

    df = load_issuer_spreads_csv(tmp_file.name)
    assert df.loc[0, "measure"] == "CDS_SPREAD"  # normalized uppercase
    assert df.loc[0, "units"] == "bps"  # normalized lowercase

    os.remove(tmp_file.name)


def test_load_issuer_spreads_csv_invalid_units():
    """CSV loader should raise error on invalid units."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w") as tmp_file:
        tmp_file.write(
            "counterparty,instrument_id,id_type,as_of_date,measure,value,units,currency\n"
            "CPTY_X,US1111111111,ISIN,2024-12-31,CDS_SPREAD,150,points,USD\n"
        )

    with pytest.raises(ValueError):
        load_issuer_spreads_csv(tmp_file.name)

    os.remove(tmp_file.name)


def test_load_credit_indexes():
    """FRED loader should return a non-empty DataFrame with expected columns."""
    df = load_credit_indexes(start="2024-01-01")
    assert not df.empty
    assert {"IG_OAS_bps", "HY_OAS_bps", "BAA_yield_pct"}.issubset(df.columns)
