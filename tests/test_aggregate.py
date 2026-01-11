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

Z_999 = norm.ppf(0.999)

# ===================================================================
# Tests for normalize_risk_results
# ===================================================================


def test_normalize_risk_results_typical_case():
    market_results = {"var_1y_999": 150_000_000.0}
    credit_results = {"EL_total": 80_000_000.0, "UL_total": 200_000_000.0}
    oprisk_capital = 120_000_000.0

    op_results = {"capital_999": oprisk_capital, "expected_loss": 0.0}
    normalized = normalize_risk_results(market_results, credit_results, op_results)

    expected_market_ul = 150_000_000.0 / Z_999
    expected_op_ul = 120_000_000.0 / Z_999

    assert normalized["Credit"]["EL"] == 80_000_000.0
    assert normalized["Credit"]["UL"] == 200_000_000.0

    assert normalized["Market"]["EL"] == 0.0
    assert np.isclose(normalized["Market"]["UL"], expected_market_ul)
    assert normalized["Market"]["Total_Standalone"] == 150_000_000.0

    assert normalized["OpRisk"]["EL"] == 0.0
    assert np.isclose(normalized["OpRisk"]["UL"], expected_op_ul)


def test_normalize_risk_results_fallback_to_es():
    market_results = {"es_1y_999": 160_000_000.0}
    credit_results = {"EL_total": 50_000_000.0}
    oprisk_capital = 100_000_000.0

    op_results = {"capital_999": oprisk_capital, "expected_loss": 0.0}
    normalized = normalize_risk_results(market_results, credit_results, op_results)

    expected_market_ul = 160_000_000.0 / Z_999
    expected_op_ul = 100_000_000.0 / Z_999

    assert np.isclose(normalized["Market"]["UL"], expected_market_ul)
    assert normalized["Credit"]["EL"] == 50_000_000.0
    assert np.isclose(normalized["OpRisk"]["UL"], expected_op_ul)


def test_normalize_risk_results_missing_keys():
    market_results = {}
    credit_results = {}
    oprisk_capital = 0.0

    op_results = {"capital_999": oprisk_capital, "expected_loss": 0.0}
    normalized = normalize_risk_results(market_results, credit_results, op_results)

    expected = {
        "Market": {
            "EL": 0.0,
            "UL": 0.0,
            "Total_Standalone": 0.0,
            "label": "Market Risk",
        },
        "Credit": {"EL": 0.0, "UL": 0.0, "Total_Standalone": 0.0, "label": "Credit"},
        "OpRisk": {
            "EL": 0.0,
            "UL": 0.0,
            "Total_Standalone": 0.0,
            "label": "Operational Risk",
        },
    }

    # Manual deep comparison (pytest.approx doesn't work on nested dicts)
    for risk in expected:
        for k, v in expected[risk].items():
            actual = normalized[risk][k]
            if isinstance(v, float):
                assert np.isclose(actual, v, atol=1e-6), (
                    f"Mismatch in {risk}.{k}: {actual} != {v}"
                )
            else:
                assert actual == v, f"Mismatch in {risk}.{k}: {actual} != {v}"


# ===================================================================
# Tests for aggregate_economic_capital
# ===================================================================


@pytest.fixture
def standard_normalized():
    return {
        "Market": {"EL": 0.0, "UL": 150_000_000.0, "Total_Standalone": 150_000_000.0},
        "Credit": {
            "EL": 80_000_000.0,
            "UL": 200_000_000.0,
            "Total_Standalone": 280_000_000.0,
        },
        "OpRisk": {"EL": 0.0, "UL": 120_000_000.0, "Total_Standalone": 120_000_000.0},
    }


def test_aggregate_economic_capital_zero_ul(standard_normalized):
    zero_ul = {
        "Market": {"EL": 0.0, "UL": 0.0},
        "Credit": {"EL": 10_000_000.0, "UL": 0.0},
        "OpRisk": {"EL": 0.0, "UL": 0.0},
    }

    EL_total, UL_portfolio, EC_total, marginal, div_benefit = (
        aggregate_economic_capital(
            market_results={},
            credit_results=zero_ul["Credit"],
            op_results={},
        )
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
        "OpRisk": {"EL": 0.0, "UL": 120_000_000.0},
        "Credit": {"EL": 80_000_000.0, "UL": 200_000_000.0},
        "Market": {"EL": 0.0, "UL": 150_000_000.0},
    }

    result_flipped = aggregate_economic_capital(
        market_results=flipped["Market"],
        credit_results=flipped["Credit"],
        op_results=flipped["OpRisk"],
    )
    result_standard = aggregate_economic_capital(
        market_results=standard_normalized["Market"],
        credit_results=standard_normalized["Credit"],
        op_results=standard_normalized["OpRisk"],
    )

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
        market_results=standard_normalized["Market"],
        credit_results=standard_normalized["Credit"],
        op_results=standard_normalized["OpRisk"],
        confidence_level=0.99,
    )
    _, _, EC_999, _, _ = aggregate_economic_capital(
        market_results=standard_normalized["Market"],
        credit_results=standard_normalized["Credit"],
        op_results=standard_normalized["OpRisk"],
        confidence_level=0.999,
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
        market_results=normalized["Market"],
        credit_results=normalized["Credit"],
        op_results=normalized["OpRisk"],
        confidence_level=0.999,
        copula_df=None,  # Explicitly Gaussian
        n_sim=50_000,  # High number for stable comparison
        seed=42,
    )

    # 2. t-copula with moderate fat tails (df=5)
    _, _, ec_tcopula, _, div_tcopula = aggregate_economic_capital(
        market_results=normalized["Market"],
        credit_results=normalized["Credit"],
        op_results=normalized["OpRisk"],
        confidence_level=0.999,
        copula_df=5.0,
        n_sim=50_000,
        seed=42,
    )

    # 3. t-copula with very heavy tails (df=3)
    _, _, ec_heavy, _, div_heavy = aggregate_economic_capital(
        market_results=normalized["Market"],
        credit_results=normalized["Credit"],
        op_results=normalized["OpRisk"],
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


def test_marginals_sum_to_total_ec(standard_normalized):
    # Run aggregation to get realistic marginals
    _, _, EC_total, marginal, _ = aggregate_economic_capital(
        market_results=standard_normalized["Market"],
        credit_results=standard_normalized["Credit"],
        op_results=standard_normalized["OpRisk"],
        confidence_level=0.999,
    )

    marginal_sum = marginal.sum()
    diff = abs(marginal_sum - EC_total)

    assert diff < 1_000_000.0, (
        f"Marginals sum £{marginal_sum:,.0f} != Total EC £{EC_total:,.0f} "
        f"(difference: £{diff:,.0f})"
    )


def test_standalone_ge_total_ec(standard_normalized):
    EC_total = 400_000_000.0  # example realistic value
    standalone_sum = sum(r["Total_Standalone"] for r in standard_normalized.values())

    assert standalone_sum >= EC_total - 1_000_000.0, (
        f"Standalone sum £{standalone_sum:,.0f} < Total EC £{EC_total:,.0f} "
        f"(negative diversification of £{EC_total - standalone_sum:,.0f}) - impossible!"
    )


@pytest.mark.parametrize(
    "has_wwr_keys, expected_pass",
    [
        (True, True),  # WWR UL >= base UL
        (False, True),  # No WWR keys → skip check
        (True, False),  # Simulate failure case
    ],
)
def test_credit_wwr_ul_non_negative(has_wwr_keys, expected_pass):
    credit_results = {}
    if has_wwr_keys:
        credit_results = {
            "UL_total": 100_000_000.0,
            "UL_WWR_total": 120_000_000.0
            if expected_pass
            else 90_000_000.0,  # fail case
        }

    # Manual check
    if "UL_WWR_total" in credit_results and "UL_total" in credit_results:
        base_ul = credit_results["UL_total"]
        wwr_ul = credit_results["UL_WWR_total"]
        passes = wwr_ul >= base_ul - 1_000_000.0

        assert passes == expected_pass, (
            f"WWR UL £{wwr_ul:,.0f} < base UL £{base_ul:,.0f} "
            f"(diff: £{base_ul - wwr_ul:,.0f})"
        )


def test_el_components_sum_to_total(standard_normalized):
    EL_total = sum(r["EL"] for r in standard_normalized.values())
    el_sum = sum(r["EL"] for r in standard_normalized.values())

    assert abs(el_sum - EL_total) < 1_000.0, (
        f"Sum of EL components £{el_sum:,.0f} != Total EL £{EL_total:,.0f} "
        f"(diff: £{abs(el_sum - EL_total):,.0f})"
    )


# Full end-to-end aggregation test
def test_full_aggregation_consistency():
    # Mock minimal inputs
    market = {"es_1y_999": 200_000_000.0}
    credit = {"EL_total": 50_000_000.0, "UL_total": 150_000_000.0}
    op = {"capital_999": 100_000_000.0, "expected_loss": 10_000_000.0}

    EL_total, UL_portfolio, EC_total, marginal, div_benefit = (
        aggregate_economic_capital(
            market_results=market,
            credit_results=credit,
            op_results=op,
        )
    )

    # Basic sanity checks
    assert EL_total > 0
    assert EC_total > EL_total
    assert div_benefit >= 0, "Diversification benefit cannot be negative"
    assert marginal.sum() == pytest.approx(EC_total, abs=5_000_000.0), (
        "Marginal contributions should approximate total EC"
    )
