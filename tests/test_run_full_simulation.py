"""
Unit tests for econ_capital.run_full_simulation()
Assumes risk runner imports are lazy inside the function.
"""

import json
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t

from econ_capital import run_full_simulation


@pytest.fixture
def fixed_time(monkeypatch):
    """Freeze datetime.now() and isoformat for deterministic timestamps."""
    fixed_dt = datetime(2025, 1, 15, 14, 30, 0)

    class MockDateTime(datetime):
        @classmethod
        def now(cls):
            return fixed_dt

        def isoformat(self, *args, **kwargs):
            return "2025-01-15T14:30:00"

    monkeypatch.setattr("econ_capital.datetime", MockDateTime)

    return fixed_dt


def test_run_full_simulation_happy_path(fixed_time):
    """Test full orchestration with realistic values and verify output + JSON save."""
    market_results = {"var_1y_999": 150_000_000.0}
    credit_results = {"EL_total": 80_000_000.0, "UL_total": 200_000_000.0}
    op_results = {"total_capital": 120_000_000.0, "expected_loss": 0.0}

    normalized = {
        "Market": {"EL": 0.0, "UL": 150_000_000.0},
        "Credit": {"EL": 80_000_000.0, "UL": 200_000_000.0},
        "OpRisk": {"EL": 0.0, "UL": 120_000_000.0},
    }

    EL_total = 80_000_000.0
    UL_portfolio = 250_000_000.0
    z = t.ppf(0.999, 3)
    EC_total = EL_total + z * UL_portfolio
    diversification_benefit = 150_000_000.0
    marginal_series = pd.Series(
        [100_000_000.0, 200_000_000.0, 80_000_000.0],
        index=["Market", "Credit", "OpRisk"],
        name="EC_Marginal",
    )

    with (
        patch(
            "econ_capital.market_risk.run_marketrisk_report.main",
            return_value=market_results,
        ),
        patch(
            "econ_capital.credit_risk.run_creditrisk_report.main",
            return_value=credit_results,
        ),
        patch("econ_capital.op_risk.run_oprisk_report.main", return_value=op_results),
        patch("econ_capital.normalize_risk_results", return_value=normalized),
        patch(
            "econ_capital.aggregate_economic_capital",
            return_value=(
                EL_total,
                UL_portfolio,
                EC_total,
                marginal_series,
                diversification_benefit,
            ),
        ),
    ):
        result = run_full_simulation(config_override={"test": True}, verbose=False)

    assert result["run_timestamp"] == "2025-01-15T14:30:00"
    assert result["EL_total"] == EL_total
    assert np.isclose(result["UL_portfolio"], UL_portfolio)
    assert np.isclose(result["EC_total"], EC_total)
    assert np.isclose(result["diversification_benefit"], diversification_benefit)
    assert result["individual_risks"] == normalized
    assert result["config_override"] == {"test": True}
    assert result["marginal_contributions"] == marginal_series.to_dict()

    # Check JSON report
    json_path = "econ_capital/reports/FirmWide_EC_Summary_20250115_143000.json"
    with open(json_path) as f:
        saved = json.load(f)
    assert saved["run_timestamp"] == "2025-01-15T14:30:00"
    assert saved["EL_total"] == EL_total
    assert saved["config_override"] == {"test": True}


def test_run_full_simulation_handles_zero_values(fixed_time):
    with (
        patch("econ_capital.market_risk.run_marketrisk_report.main", return_value={}),
        patch("econ_capital.credit_risk.run_creditrisk_report.main", return_value={}),
        patch("econ_capital.op_risk.run_oprisk_report.main", return_value={}),
    ):
        result = run_full_simulation(verbose=False)

    assert result["EL_total"] == 0.0
    assert result["UL_portfolio"] == 0.0
    assert result["EC_total"] == 0.0
    assert result["diversification_benefit"] == 0.0
    assert all(v == 0.0 for v in result["marginal_contributions"].values())


def test_run_full_simulation_verbose_prints(fixed_time, capsys):
    with (
        patch(
            "econ_capital.market_risk.run_marketrisk_report.main",
            return_value={"var_1y_999": 1e8},
        ),
        patch(
            "econ_capital.credit_risk.run_creditrisk_report.main",
            return_value={"EL_total": 5e7, "UL_total": 2e8},
        ),
        patch(
            "econ_capital.op_risk.run_oprisk_report.main",
            return_value={"total_capital": 1e8},
        ),
    ):
        run_full_simulation(verbose=True)

    captured = capsys.readouterr().out
    expected_phrases = [
        "ECONOMIC CAPITAL SIMULATOR — FIRM-WIDE RUN",
        "Running Market Risk module...",
        "Running Credit Risk module...",
        "Running Operational Risk module...",
        "All individual risk modules completed successfully.",
        "Normalized Individual Contributions:",
        "FIRM-WIDE ECONOMIC CAPITAL RESULTS",
        "Consolidated summary saved to:",
    ]
    for phrase in expected_phrases:
        assert phrase in captured


def test_run_full_simulation_respects_config_override(fixed_time):
    config = {"scenario": "stress", "calibration_date": "2025-12-31"}

    with (
        patch(
            "econ_capital.market_risk.run_marketrisk_report.main",
            return_value={"var_1y_999": 100.0},
        ),
        patch(
            "econ_capital.credit_risk.run_creditrisk_report.main",
            return_value={"EL_total": 10.0, "UL_total": 50.0},
        ),
        patch(
            "econ_capital.op_risk.run_oprisk_report.main",
            return_value={"total_capital": 20.0},
        ),
    ):
        result = run_full_simulation(config_override=config, verbose=False)

    assert result["config_override"] == config
