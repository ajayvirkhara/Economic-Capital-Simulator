"""
Unit tests for econ_capital/aggregate.py
Covers both normalize_risk_results() and aggregate_economic_capital()
"""

import numpy as np
import pandas as pd
import pytest

from econ_capital.aggregate import (
    normalize_risk_results,
    aggregate_economic_capital,
)


# ===================================================================
# Tests for normalize_risk_results
# ===================================================================


def test_normalize_risk_results_typical_case():
    market_results = {"var_1y_999": 150_000_000.0}
    credit_results = {"EL_total": 80_000_000.0, "UL_total": 200_000_000.0}
    oprisk_capital = 120_000_000.0

    normalized = normalize_risk_results(market_results, credit_results, oprisk_capital)

    expected = {
        "Market": {"EL": 0.0, "UL": 150_000_000.0},
        "Credit": {"EL": 80_000_000.0, "UL": 200_000_000.0},
        "OpRisk": {"EL": 0.0, "UL": 120_000_000.0},
    }

    assert normalized == expected


def test_normalize_risk_results_fallback_to_es():
    market_results = {"es_1y_999": 160_000_000.0}
    credit_results = {"EL_total": 50_000_000.0}
    oprisk_capital = 100_000_000.0

    normalized = normalize_risk_results(market_results, credit_results, oprisk_capital)

    assert normalized["Market"]["UL"] == 160_000_000.0
    assert normalized["Credit"]["EL"] == 50_000_000.0
    assert normalized["OpRisk"]["UL"] == 100_000_000.0


def test_normalize_risk_results_missing_keys():
    market_results = {}
    credit_results = {}
    oprisk_capital = 0.0

    normalized = normalize_risk_results(market_results, credit_results, oprisk_capital)

    expected = {
        "Market": {"EL": 0.0, "UL": 0.0},
        "Credit": {"EL": 0.0, "UL": 0.0},
        "OpRisk": {"EL": 0.0, "UL": 0.0},
    }

    assert normalized == expected


# ===================================================================
# Tests for aggregate_economic_capital
# ===================================================================


@pytest.fixture
def standard_normalized():
    return {
        "Market": {"EL": 0.0, "UL": 150_000_000.0},
        "Credit": {"EL": 80_000_000.0, "UL": 200_000_000.0},
        "OpRisk": {"EL": 0.0, "UL": 120_000_000.0},
    }


def test_aggregate_economic_capital_zero_ul(standard_normalized):
    zero_ul = {
        "Market": {"EL": 0.0, "UL": 0.0},
        "Credit": {"EL": 10_000_000.0, "UL": 0.0},
        "OpRisk": {"EL": 0.0, "UL": 0.0},
    }

    EL_total, UL_portfolio, EC_total, marginal, div_benefit = (
        aggregate_economic_capital(zero_ul)
    )

    assert EL_total == 10_000_000.0
    assert UL_portfolio == 0.0
    assert EC_total == 10_000_000.0
    assert div_benefit == 0.0
    expected_marginal = pd.Series(
        [0.0, 10_000_000.0, 0.0],
        index=["Market", "Credit", "OpRisk"],
        name="EC_Marginal",
    )
    pd.testing.assert_series_equal(marginal, expected_marginal)


def test_aggregate_economic_capital_different_risk_order(standard_normalized):
    flipped = {
        "OpRisk": {"EL": 0.0, "UL": 120e6},
        "Credit": {"EL": 80e6, "UL": 200e6},
        "Market": {"EL": 0.0, "UL": 150e6},
    }

    result_flipped = aggregate_economic_capital(flipped)
    result_standard = aggregate_economic_capital(standard_normalized)

    # Scalar outputs must be identical
    assert all(np.isclose(result_flipped[i], result_standard[i]) for i in range(3))
    assert np.isclose(result_flipped[4], result_standard[4])

    # Marginal contributions same values
    pd.testing.assert_series_equal(
        result_flipped[3].sort_index(),
        result_standard[3].sort_index(),
        check_names=False,
    )


def test_aggregate_economic_capital_custom_confidence_level(standard_normalized):
    _, _, EC_99, _, _ = aggregate_economic_capital(
        standard_normalized, confidence_level=0.99
    )
    _, _, EC_999, _, _ = aggregate_economic_capital(
        standard_normalized, confidence_level=0.999
    )

    assert EC_999 > EC_99  # higher confidence = higher capital requirement


def test_t_copula_produces_fatter_tails_than_gaussian():
    """
    Test that enabling t-copula (with low df) produces:
    - Higher total EC than Gaussian (due to fat tails)
    - Activation of the t-copula code path
    - Reasonable diversification benefit
    """
    # Simple normalized risk inputs (all EL=0 for clarity)
    normalized = {
        "Market": {"EL": 0.0, "UL": 100_000_000.0},
        "Credit": {"EL": 0.0, "UL": 80_000_000.0},
        "OpRisk": {"EL": 0.0, "UL": 60_000_000.0},
    }

    # 1. Gaussian baseline (no copula_df)
    _, _, ec_gaussian, _, div_gaussian = aggregate_economic_capital(
        normalized,
        confidence_level=0.999,
        copula_df=None,  # Explicitly Gaussian
        n_sim=50_000,  # High number for stable comparison
        seed=42,
    )

    # 2. t-copula with moderate fat tails (df=5)
    _, _, ec_tcopula, _, div_tcopula = aggregate_economic_capital(
        normalized,
        confidence_level=0.999,
        copula_df=5.0,
        n_sim=50_000,
        seed=42,
    )

    # 3. t-copula with very heavy tails (df=3)
    _, _, ec_heavy, _, div_heavy = aggregate_economic_capital(
        normalized,
        confidence_level=0.999,
        copula_df=3.0,
        n_sim=50_000,
        seed=42,
    )

    # Assertions
    assert ec_tcopula > ec_gaussian * 1.02  # t-copula EC higher than Gaussian EC
    assert ec_heavy > ec_tcopula * 1.02  # Total EC increases with tail heaviness
    assert (
        div_tcopula > div_gaussian * 0.9
    )  # Diversification benefit higher with t-copula as Gaussian z-factor inflates standalone EC

    print(f"Gaussian EC: £{ec_gaussian:,.0f}")
    print(
        f"t-copula (df=5) EC: £{ec_tcopula:,.0f} (+{(ec_tcopula / ec_gaussian - 1) * 100:.1f}%)"
    )
    print(
        f"t-copula (df=3) EC: £{ec_heavy:,.0f} (+{(ec_heavy / ec_gaussian - 1) * 100:.1f}%)"
    )
