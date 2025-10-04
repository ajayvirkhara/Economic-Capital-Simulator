"""
Unit tests for credit_risk module
"""

import logging
import cProfile
import pstats
from io import StringIO

import os
import tempfile
import pandas as pd
import numpy as np
import pytest

from econ_capital.credit_risk.data_loaders import (
    load_dummy_credit_data,
    load_issuer_spreads_csv,
    load_credit_indexes,
    CSV_SCHEMA,
)
from econ_capital.credit_risk.trade_models import Trade, NettingSet
from econ_capital.credit_risk.exposure_engine import ExposureEngine
from econ_capital.credit_risk.csa import CSA
from econ_capital.credit_risk.exposure_models import _compute_mtm


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)


def profile_test(func):
    """Decorator to profile heavy test functions."""

    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()

        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
        ps.print_stats(5)
        logger.info("Profiling summary for %s:\n%s", func.__name__, s.getvalue())
        return result

    return wrapper


# -----------------------------------------------------------------------
# Data loaders unit tests
# -----------------------------------------------------------------------


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


# -----------------------------------------------------------------------
# Exposure engine unit tests
# -----------------------------------------------------------------------


def simulate_dummy_market(  # pylint: disable=too-many-positional-arguments
    n_paths=5000, n_steps=6, s0=100.0, mu=0.0, sigma=0.2, seed=42
):
    """Simple GBM paths for test."""
    rng = np.random.default_rng(seed)
    times = np.linspace(0, 1.0, n_steps)
    dt = np.diff(np.concatenate([[0.0], times]))
    z = rng.standard_normal((n_paths, n_steps))
    S = np.empty_like(z)
    S[:, 0] = s0
    for k in range(1, n_steps):
        S[:, k] = S[:, k - 1] * np.exp(
            (mu - 0.5 * sigma**2) * dt[k - 1] + sigma * np.sqrt(dt[k - 1]) * z[:, k - 1]
        )
    return times, {"SP500": S}


def test_linear_trade_mtm_increases_with_price():
    """Linear trade should have MTM that increases with S."""
    times, market_paths = simulate_dummy_market(
        n_paths=5000,
        n_steps=252,
        s0=100.0,
        mu=0.5,
        sigma=0.2,
        seed=42,
    )
    tr = Trade(name="IRS_like", factor="SP500", w=1.0)
    ns = NettingSet(counterparty="A", trades=[tr], csa=CSA())
    engine = ExposureEngine(netting_set=ns, market_paths=market_paths, times=times)
    mtm = engine.compute_mtm_only()
    assert (mtm[:, -1] > mtm[:, 0]).mean() > 0.8  # mostly positive slope


def test_gamma_trade_has_convex_mtm():
    """Option-like (γ>0) trade should exhibit convex MTM."""
    _, market_paths = simulate_dummy_market()
    tr = Trade(name="Option_like", factor="SP500", w=0.1, gamma=0.002)
    ns = NettingSet(counterparty="B", trades=[tr], csa=CSA())
    mtm = _compute_mtm(ns.trades, market_paths)
    assert np.var(mtm[:, -1]) > np.var(mtm[:, 0])


@profile_test
def test_vm_and_im_effects():
    """IM adds constant buffer; higher VM frequency reduces exposure volatility."""
    times, market_paths = simulate_dummy_market(n_steps=252)
    tr = Trade(name="IRS_like", factor="SP500", w=0.8)
    ns_daily = NettingSet(
        counterparty="C",
        trades=[tr],
        csa=CSA(vm_calls_per_day=1, im=0.0, threshold=1.0),
    )
    ns_weekly = NettingSet(
        counterparty="D",
        trades=[tr],
        csa=CSA(vm_mode="per_week", vm_calls=1, im=0.0, threshold=1.0),
    )
    ns_im = NettingSet(
        counterparty="E",
        trades=[tr],
        csa=CSA(vm_calls_per_day=1, im=1.0, threshold=1.0),
    )

    engine_d = ExposureEngine(
        netting_set=ns_daily, market_paths=market_paths, times=times
    )
    engine_w = ExposureEngine(
        netting_set=ns_weekly, market_paths=market_paths, times=times
    )
    engine_i = ExposureEngine(netting_set=ns_im, market_paths=market_paths, times=times)

    exp_d, _ = engine_d.compute_exposure_profile()
    exp_i, _ = engine_i.compute_exposure_profile()
    exp_w, _ = engine_w.compute_exposure_profile()

    # VM: higher frequency should stabilize exposure more
    assert exp_d.std() < exp_w.std(), "Daily VM should reduce Exposure volatility"

    # IM: Initial Margin should reduce the overall exposure profile (EPE)
    # Compare the average exposure (e.g., Expectation of Exposure - EPE) for the IM case (exp_i)
    # vs. the daily VM case (exp_d), which has no IM (IM=0.0).
    assert (
        exp_i.mean() < exp_d.mean()
    ), "Initial Margin should reduce mean exposure (EPE)"


def test_summary_dataframe_structure():
    """Exposure summary should have correct columns and monotonic EPE_cum."""
    times, market_paths = simulate_dummy_market()
    tr = Trade(name="IRS_like", factor="SP500", w=1.0)
    ns = NettingSet(counterparty="Z", trades=[tr], csa=CSA())
    engine = ExposureEngine(netting_set=ns, market_paths=market_paths, times=times)
    _, summary = engine.compute_exposure_profile()

    expected_cols = {"time", "EE", "EPE_cum"}
    assert expected_cols.issubset(summary.columns)
    assert np.all(np.diff(summary["EPE_cum"]) >= -1e-6), "EPE_cum should not decrease"
