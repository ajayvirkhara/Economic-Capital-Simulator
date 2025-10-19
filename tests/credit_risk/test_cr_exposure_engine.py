"""
Unit tests for econ_capital.credit_risk.exposure_engine
"""

import numpy as np

from econ_capital.credit_risk.exposure_models import _compute_mtm
from econ_capital.credit_risk import (
    Trade,
    NettingSet,
    ExposureEngine,
    CSA,
)
from econ_capital.utils import profile_test


# pylint: disable=too-many-positional-arguments
def simulate_dummy_market(
    n_paths=5000,
    n_steps=6,
    s0=100.0,
    mu=0.0,
    sigma=0.2,
    seed=42,
):
    # Simulates a dummy market path using a Geometric Brownian Motion model.
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
    # Tests that the Mark-to-Market (MtM) for a simple linear trade increases on average when the underlying price has a positive drift (mu=0.5).
    times, market_paths = simulate_dummy_market(n_steps=252, mu=0.5)
    tr = Trade(name="IRS_like", factor="SP500", w=1.0)
    ns = NettingSet(counterparty="A", trades=[tr], csa=CSA())
    engine = ExposureEngine(netting_set=ns, market_paths=market_paths, times=times)
    mtm = engine.compute_mtm_only()
    assert (mtm[:, -1] > mtm[:, 0]).mean() > 0.8


def test_gamma_trade_has_convex_mtm():
    # Tests that a trade with gamma (like an option) exhibits convex MtM behaviour, which is measured by an increase in MtM variance over time.
    _, market_paths = simulate_dummy_market()
    tr = Trade(name="Option_like", factor="SP500", w=0.1, gamma=0.002)
    ns = NettingSet(counterparty="B", trades=[tr], csa=CSA())
    mtm = _compute_mtm(ns.trades, market_paths)
    assert np.var(mtm[:, -1]) > np.var(mtm[:, 0])


@profile_test
def test_vm_and_im_effects():
    # Tests the impact of Variation Margin (VM) frequency and Initial Margin (IM) on the exposure profile's standard deviation and mean.
    times, market_paths = simulate_dummy_market(n_steps=252)
    tr = Trade(name="IRS_like", factor="SP500", w=0.8)

    ns_daily = NettingSet(counterparty="C", trades=[tr], csa=CSA(vm_calls_per_day=1))
    ns_weekly = NettingSet(
        counterparty="D", trades=[tr], csa=CSA(vm_mode="per_week", vm_calls=1)
    )
    ns_im = NettingSet(
        counterparty="E", trades=[tr], csa=CSA(vm_calls_per_day=1, im=1.0)
    )

    engine_d = ExposureEngine(ns_daily, market_paths, times)
    engine_w = ExposureEngine(ns_weekly, market_paths, times)
    engine_i = ExposureEngine(ns_im, market_paths, times)

    exp_d, _ = engine_d.compute_exposure_profile()
    exp_i, _ = engine_i.compute_exposure_profile()
    exp_w, _ = engine_w.compute_exposure_profile()

    assert exp_d.std() < exp_w.std()
    assert exp_i.mean() < exp_d.mean()


def test_summary_dataframe_structure():
    # Tests that the output summary DataFrame contains the required columns (time, EE, EPE_cum) and that the Cumulative Expected Positive Exposure (EPE_cum) is monotonically non-decreasing.
    times, market_paths = simulate_dummy_market()
    tr = Trade(name="IRS_like", factor="SP500", w=1.0)
    ns = NettingSet(counterparty="Z", trades=[tr], csa=CSA())
    engine = ExposureEngine(ns, market_paths, times)
    _, summary = engine.compute_exposure_profile()
    assert {"time", "EE", "EPE_cum"}.issubset(summary.columns)
    assert np.all(np.diff(summary["EPE_cum"]) >= -1e-6)
