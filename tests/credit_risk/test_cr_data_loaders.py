"""
Unit tests for econ_capital.credit_risk.data_loaders
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
from econ_capital.utils import profile_test


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------
# Tests the loading and structure of the dummy credit data DataFrame.
def test_load_dummy_credit_data():
    df = load_dummy_credit_data()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 8)
    assert list(df.columns) == CSV_SCHEMA
    assert (df["units"] == "bps").all()


# Tests successful loading and data transformation of a valid issuer spreads CSV file.
def test_load_issuer_spreads_csv_valid():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w") as tmp_file:
        tmp_file.write(
            "counterparty,instrument_id,id_type,as_of_date,measure,value,units,currency\n"
            "CPTY_X,US1111111111,ISIN,2024-12-31, cds_spread ,150, BPS ,USD\n"
        )
    df = load_issuer_spreads_csv(tmp_file.name)
    assert df.loc[0, "measure"] == "CDS_SPREAD"
    assert df.loc[0, "units"] == "bps"
    os.remove(tmp_file.name)


# Tests that loading an issuer spreads CSV with invalid units raises a ValueError.
def test_load_issuer_spreads_csv_invalid_units():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w") as tmp_file:
        tmp_file.write(
            "counterparty,instrument_id,id_type,as_of_date,measure,value,units,currency\n"
            "CPTY_X,US1111111111,ISIN,2024-12-31,CDS_SPREAD,150,points,USD\n"
        )
    with pytest.raises(ValueError):
        load_issuer_spreads_csv(tmp_file.name)
    os.remove(tmp_file.name)


@profile_test
# Tests loading credit index data from FRED and checks for expected columns.
def test_load_credit_indexes():
    df = load_credit_indexes(start="2024-01-01")
    assert not df.empty
    assert {"IG_OAS_bps", "HY_OAS_bps", "BAA_yield_pct"}.issubset(df.columns)
