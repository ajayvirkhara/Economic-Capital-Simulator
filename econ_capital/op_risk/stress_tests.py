"""
Pure stress-testing engine for OpRisk EC (LDA)

All tabular/display/export logic belongs in reporting.py
"""

from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from .scenarios import (
    Scenario,
    ScenarioSet,
    apply_scenario_to_config,
)
from .lda_engine import lda_run_engine
from .config import OpRiskConfig


@dataclass(frozen=True)
class StressTestResult:
    """Immutable result container"""

    name: str
    description: str
    capital_base: float
    capital_stressed: float
    absolute_uplift: float
    uplift_factor: float
    uplift_pct: float
    runtime_sec: float


def _run_single_scenario(
    args: Tuple[Dict[str, Any], Scenario],
) -> Tuple[str, Dict[str, Any], float]:
    """Internal worker used by parallel executor"""
    base_config, scenario = args
    start = time.perf_counter()

    stressed_config = apply_scenario_to_config(base_config, scenario)
    result: Dict[str, Any] = lda_run_engine(stressed_config)
    runtime = time.perf_counter() - start

    return scenario.name, result, runtime


class OpRiskStressTester:
    """
    Core stress-testing engine.
    """

    def __init__(self, config_path: str = "config/op_config.yaml"):
        self.config_path = Path(config_path)
        self._baseline_result: Optional[Dict[str, Any]] = None
        self._base_config: Optional[Dict[str, Any]] = None

    @property
    def baseline(self) -> Dict[str, Any]:
        if self._baseline_result is None:
            cfg = OpRiskConfig(self.config_path)
            cfg.validate()
            self._base_config = cfg.as_dict()
            self._baseline_result = lda_run_engine(self._base_config)
        return self._baseline_result

    @property
    def baseline_capital(self) -> float:
        return float(self.baseline.get("capital_999", np.nan))

    def run_scenario_set(
        self,
        scenario_set: ScenarioSet,
        parallel: bool = True,
        max_workers: Optional[int] = None,
    ) -> List[StressTestResult]:
        """Run all scenarios in a ScenarioSet"""
        scenarios = scenario_set.scenarios
        _ = self.baseline  # Force load

        tasks = [(self._base_config, scen) for scen in scenarios]
        results: List[StressTestResult] = []

        if parallel and len(scenarios) > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_run_single_scenario, task) for task in tasks
                ]
                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Running scenarios"
                ):
                    raw_result = future.result()
                    name: str = raw_result[0]
                    stressed_result: Dict[str, Any] = raw_result[1]
                    runtime: float = raw_result[2]
                    scenario = next(s for s in scenarios if s.name == name)
                    cap_stressed = float(stressed_result.get("capital_999", np.nan))
                    results.append(self._make_result(scenario, cap_stressed, runtime))
        else:
            for scenario in tqdm(scenarios, desc="Running scenarios"):
                raw_result = _run_single_scenario((self._base_config, scenario))
                name: str = raw_result[0]
                stressed_result: Dict[str, Any] = raw_result[1]
                runtime: float = raw_result[2]
                cap_stressed = float(stressed_result.get("capital_999", np.nan))
                results.append(self._make_result(scenario, cap_stressed, runtime))

        return sorted(results, key=lambda r: r.uplift_factor, reverse=True)

    def _make_result(
        self, scenario: Scenario, cap_stressed: float, runtime: float
    ) -> StressTestResult:
        base = self.baseline_capital
        return StressTestResult(
            name=scenario.name,
            description=scenario.note or "",
            capital_base=base,
            capital_stressed=cap_stressed,
            absolute_uplift=cap_stressed - base,
            uplift_factor=cap_stressed / base if base > 0 else np.nan,
            uplift_pct=(cap_stressed - base) / base if base > 0 else np.nan,
            runtime_sec=runtime,
        )

    def get_base_config_for_tests(self) -> dict:
        return self._base_config.copy()

    def make_result_for_tests(self, scenario, cap_stressed, runtime):
        return self._make_result(scenario, cap_stressed, runtime)
