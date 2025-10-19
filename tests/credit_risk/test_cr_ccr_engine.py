"""
Unit tests for econ_capital.credit_risk.ccr_engine
"""

import numpy as np

from econ_capital.credit_risk import (
    compute_counterparty_risk_profiles,
    aggregate_credit_losses,
)


def test_compute_counterparty_risk_profiles():
    data = [
        {"name": "A", "EAD": 100, "PD": 0.02, "LGD": 0.6},
        {"name": "B", "EAD": 200, "PD": 0.01, "LGD": 0.4},
    ]
    df = compute_counterparty_risk_profiles(data)
    assert "EL" in df.columns and "UL" in df.columns
    assert np.all(df["EL"] > 0)
    assert np.all(df["UL"] > 0)


def test_aggregate_credit_losses():
    el = np.array([1.2, 0.8, 1.0])
    ul = np.array([2.0, 1.5, 1.2])
    corr = np.eye(3) * 0.5 + 0.5  # simple correlated structure
    EL_total, UL_total, EC_total = aggregate_credit_losses(el, ul, corr)
    assert EL_total > 0
    assert UL_total > 0
    assert EC_total > EL_total


def test_portfolio_diversification_effect():
    """
    Tests that portfolio UL decreases as correlation falls (diversification benefit).
    """
    ul = np.array([2.0, 1.5, 1.2])
    el = np.array([1.0, 1.0, 1.0])

    # ρ = 1 -> perfectly correlated (no diversification)
    corr_full = np.ones((3, 3))
    _, ul_full, _ = aggregate_credit_losses(el, ul, corr_full)

    # ρ = 0 -> fully independent (max diversification)
    corr_indep = np.eye(3)
    _, ul_indep, _ = aggregate_credit_losses(el, ul, corr_indep)

    assert ul_indep < ul_full, "UL should be smaller under lower correlations"
    assert ul_full == np.sum(
        ul
    ), "When ρ=1, portfolio UL should equal sum of individual ULs (no diversification)"
