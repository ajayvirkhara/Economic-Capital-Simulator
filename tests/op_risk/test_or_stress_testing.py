from __future__ import annotations

from pathlib import Path
import pytest
import numpy as np

from econ_capital.op_risk.stress_tests import (
    OpRiskStressTester,
    StressTestResult,
    _run_single_scenario,
)
from econ_capital.op_risk.scenarios import Scenario, ScenarioSet

# pylint: disable=redefined-outer-name


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------
@pytest.fixture
def config_path(tmp_path: Path) -> str:
    p = tmp_path / "config.yaml"
    p.write_text("op_risk: {}")
    return str(p)


@pytest.fixture
def tester(config_path):
    """Provides a fully initialized tester with loaded baseline"""
    t = OpRiskStressTester(config_path)
    _ = t.baseline  # Force load
    return t


@pytest.fixture
def scenario_set():
    return ScenarioSet(
        base_profile={"UoM": {"lambda": 1.0}},
        scenarios=[
            Scenario(
                name="Mild Stress",
                freq_multiplier={"UoM": 2.0},
                sev_mu_shift={"UoM": np.log(3.0)},
                sev_scale_multiplier={"UoM": 1.0},
                note="Moderate",
            ),
            Scenario(
                name="Extreme Stress",
                freq_multiplier={"UoM": 10.0},
                sev_mu_shift={"UoM": np.log(20.0)},
                sev_scale_multiplier={"UoM": 1.0},
                note="Catastrophic",
            ),
        ],
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_returns_list_of_StressTestResult(tester, scenario_set):
    results = tester.run_scenario_set(scenario_set, parallel=False)
    assert len(results) == 2
    assert all(isinstance(r, StressTestResult) for r in results)


def test_results_are_sorted_by_uplift_descending(tester, scenario_set):
    results = tester.run_scenario_set(scenario_set, parallel=False)
    assert results[0].name == "Extreme Stress"
    assert results[1].name == "Mild Stress"
    assert results[0].uplift_factor == pytest.approx(200.0)
    assert results[1].uplift_factor == pytest.approx(6.0)


def test_fields_are_correctly_populated(tester, scenario_set):
    results = tester.run_scenario_set(scenario_set, parallel=False)
    extreme = results[0]
    assert extreme.capital_base == pytest.approx(1_000_000.0)
    assert extreme.capital_stressed == pytest.approx(200_000_000.0)
    assert extreme.absolute_uplift == pytest.approx(199_000_000.0)
    assert extreme.uplift_factor == pytest.approx(200.0)
    assert extreme.uplift_pct == pytest.approx(199.0)
    assert extreme.runtime_sec >= 0.0


def test_parallel_execution_works_via_worker_unit_test(tester, scenario_set):
    """
    Verifies the parallel worker logic by calling _run_single_scenario directly.
    This tests the exact code path used in ProcessPoolExecutor — without spawning processes.
    """
    scenarios = scenario_set.scenarios
    results = []

    # use public helper instead of protected member
    base_cfg = tester.get_base_config_for_tests()

    for scenario in scenarios:
        _, result_dict, runtime = _run_single_scenario((base_cfg, scenario))
        cap_stressed = float(result_dict["capital_999"])

        # use public helper instead of protected member
        result = tester.make_result_for_tests(scenario, cap_stressed, runtime)
        results.append(result)

    # Sort like real run_scenario_set does
    results = sorted(results, key=lambda r: r.uplift_factor, reverse=True)

    assert len(results) == 2
    assert results[0].name == "Extreme Stress"
    assert results[0].uplift_factor == pytest.approx(200.0)
    assert results[1].uplift_factor == pytest.approx(6.0)


def test_StressTestResult_is_immutable(tester, scenario_set):
    result = tester.run_scenario_set(scenario_set, parallel=False)[0]
    with pytest.raises(AttributeError):
        result.name = "error"


def test_empty_note_becomes_empty_string(config_path):
    empty_set = ScenarioSet(
        base_profile={"TestUoM": {"lambda": 1.0}},
        scenarios=[
            Scenario(
                name="NoNote",
                freq_multiplier={"TestUoM": 2.0},
                sev_mu_shift={"TestUoM": 0.0},
                sev_scale_multiplier={"TestUoM": 1.0},
            )
        ],
    )
    tester = OpRiskStressTester(config_path)
    result = tester.run_scenario_set(empty_set, parallel=False)[0]
    assert result.description == ""
