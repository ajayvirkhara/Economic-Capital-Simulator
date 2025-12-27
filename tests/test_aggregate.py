"""
Unit tests for econ_capital/aggregate.py
Covers both normalize_risk_results() and aggregate_economic_capital()
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

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


def test_aggregate_economic_capital_default_correlations(standard_normalized):
    EL_total, UL_portfolio, EC_total, marginal, div_benefit = (
        aggregate_economic_capital(standard_normalized)
    )

    expected_el = 80_000_000.0
    expected_ul_vec = np.array([150e6, 200e6, 120e6])
    corr_matrix = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])
    expected_portfolio_var = expected_ul_vec @ corr_matrix @ expected_ul_vec
    expected_ul_portfolio = np.sqrt(expected_portfolio_var)

    z = norm.ppf(0.999)
    expected_ec = expected_el + z * expected_ul_portfolio

    standalone = sum(d["EL"] + z * d["UL"] for d in standard_normalized.values())
    expected_div_benefit = standalone - expected_ec

    assert np.isclose(EL_total, expected_el)
    assert np.isclose(UL_portfolio, expected_ul_portfolio)
    assert np.isclose(EC_total, expected_ec)
    assert np.isclose(div_benefit, expected_div_benefit)
    assert np.isclose(marginal.sum(), EC_total)
    assert list(marginal.index) == ["Market", "Credit", "OpRisk"]
    assert marginal.name == "EC_Marginal"


def test_aggregate_economic_capital_custom_correlations(standard_normalized):
    # All risks perfectly correlated → maximum UL (no diversification)
    custom_corr = {
        "Market": {"Credit": 1.0, "OpRisk": 1.0},
        "Credit": {"Market": 1.0, "OpRisk": 1.0},
        "OpRisk": {"Market": 1.0, "Credit": 1.0},
    }

    _, UL_custom, _, _, _ = aggregate_economic_capital(
        standard_normalized, correlations=custom_corr
    )

    _, UL_default, _, _, _ = aggregate_economic_capital(standard_normalized)

    # Perfect correlation → UL = sum of individual ULs
    expected_max_ul = 150_000_000.0 + 200_000_000.0 + 120_000_000.0  # 470e6

    assert np.isclose(UL_custom, expected_max_ul)
    assert UL_custom > UL_default  # Less diversification → higher portfolio UL


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
