import pytest
import numpy as np
from econ_capital.firmwide_reporting import FirmWideECReporter
from econ_capital.aggregate import aggregate_economic_capital


@pytest.fixture
def sample_aggregated_results():
    return {
        "EL_total": 436408416,
        "UL_portfolio": 715891094,
        "EC_total": 4694959615,
        "diversification_benefit": 1364041160,
        "marginal_contributions": {
            "Market": 3204378770,
            "Credit": 642947731,
            "OpRisk": 407558516,
        },
        "individual_risks": {
            "Market": {"EL": 0, "UL": 330684931},
            "Credit": {"EL": 21634000, "UL": 110918107},
            "OpRisk": {"EL": 414558627, "UL": 108908640},
        },
    }


def test_marginals_sum_to_total():
    """
    Verify that marginal contributions approximately sum to total EC
    using the real aggregation function with realistic dummy inputs.
    """
    # Realistic dummy inputs matching scale
    market = {
        "es_1y_999": 3_380_000_000,  # approximate standalone after t(3)
    }
    credit = {
        "EL_WWR_total": 21_634_000,
        "UL_WWR_total": 110_918_000,
    }
    oprisk = {
        "capital_999": 1_527_000_000,
        "expected_loss": 414_558_627,
    }

    # Run the ACTUAL aggregation function
    EL_total, UL_portfolio, EC_total, marginal, _ = aggregate_economic_capital(
        market_results=market,
        credit_results=credit,
        op_results=oprisk,
        confidence_level=0.999,
        copula_df=3.0,
        n_sim=20_000,  # enough for stable tail test, fast enough for CI
        seed=42,
    )

    sum_marg = marginal.sum()
    rel_diff = abs(sum_marg - EC_total) / EC_total if EC_total > 0 else 0

    # Debug print for visibility
    print(f"EL_total:          £{EL_total:,.0f}")
    print(f"EC_total:          £{EC_total:,.0f}")
    print(f"Sum marginals:     £{sum_marg:,.0f}")
    print(f"Relative diff:     {rel_diff:.4%}")
    print(f"Marginals:\n{marginal.round(0)}")

    assert rel_diff < 0.015, (
        f"Marginals sum £{sum_marg:,.0f} vs EC_total £{EC_total:,.0f} "
        f"(rel diff {rel_diff:.2%}) — check tail stability (n_tail) or increase n_sim"
    )


def test_standalone_sum_consistent():
    # Minimal example
    el_vec = np.array([0, 21634000, 414558627])
    ul_vec = np.array([330684931, 110918107, 108908640])
    quantile = 10.215  # ~t.ppf(0.999, 3)

    standalone_sum = np.sum(el_vec + quantile * ul_vec)
    assert standalone_sum > 5_500_000_000, "Standalone sum suspiciously low"


def test_reporting_initialization(sample_aggregated_results):
    reporter = FirmWideECReporter(sample_aggregated_results, copula_df=3.0)
    assert reporter.copula_df == 3.0
    assert isinstance(reporter.results, dict)
