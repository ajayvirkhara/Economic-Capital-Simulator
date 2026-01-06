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
    n_paths = market_paths["SP500"].shape[0]  # Extract number of paths
    tr = Trade(name="IRS_like", factor="SP500", w=1.0)
    ns = NettingSet(counterparty="A", trades=[tr], csa=CSA())
    engine = ExposureEngine(
        netting_set=ns, market_paths=market_paths, times=times, n_paths=n_paths
    )
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
    # 1. High volatility to ensure movement
    times, market_paths = simulate_dummy_market(n_steps=252, mu=0.10, sigma=3.5)
    n_paths = market_paths["SP500"].shape[0]

    # 2. Large trade
    tr = Trade(name="IRS_like", factor="SP500", w=1_000_000.0)

    # 3. Override MTA to 0 just for the test
    ns_daily = NettingSet(
        counterparty="C",
        trades=[tr],
        csa=CSA(threshold=10_000_000, mta=0, im=0, vm_calls_per_day=1),
    )

    ns_weekly = NettingSet(
        counterparty="D",
        trades=[tr],
        csa=CSA(threshold=10_000_000, mta=0, im=0, vm_mode="per_week", vm_calls=1),
    )

    ns_im = NettingSet(
        counterparty="E",
        trades=[tr],
        csa=CSA(threshold=10_000_000, mta=0, im=1_000_000, vm_calls_per_day=1),
    )

    engine_d = ExposureEngine(ns_daily, market_paths, times, n_paths=n_paths)
    engine_w = ExposureEngine(ns_weekly, market_paths, times, n_paths=n_paths)
    engine_i = ExposureEngine(ns_im, market_paths, times, n_paths=n_paths)

    exp_d, _ = engine_d.compute_exposure_profile()
    exp_w, _ = engine_w.compute_exposure_profile()
    exp_i, _ = engine_i.compute_exposure_profile()

    # 4. Calculate means (now that MTA=0, these will be > 0)
    mean_d = exp_d.mean()
    mean_w = exp_w.mean()
    mean_i = exp_i.mean()

    # --- Assertions ---
    # Daily check is more frequent than Weekly, so mean exposure should be lower
    assert mean_d < mean_w, f"Daily {mean_d} should be < Weekly {mean_w}"

    # IM provides 1M extra protection, so it must be lower than Daily
    assert mean_i < mean_d, f"IM {mean_i} should be < Daily {mean_d}"


def test_summary_dataframe_structure():
    # Tests that the output summary DataFrame contains the required columns (EAD, EAD_final) and that the Cumulative Expected Positive Exposure (EPE_cum) is monotonically non-decreasing.
    times, market_paths = simulate_dummy_market()
    n_paths = market_paths["SP500"].shape[0]
    tr = Trade(name="IRS_like", factor="SP500", w=1.0)
    ns = NettingSet(counterparty="Z", trades=[tr], csa=CSA())
    engine = ExposureEngine(ns, market_paths, times, n_paths=n_paths)
    _, summary = engine.compute_exposure_profile()
    assert "EAD" in summary.columns
    assert "EAD_final" in summary.columns
    assert np.allclose(summary["EAD"], engine.alpha_factor * summary["EPE_cum"])
    assert np.all(np.diff(summary["EPE_cum"]) >= -1e-6)
