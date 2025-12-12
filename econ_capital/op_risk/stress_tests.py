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
from copy import deepcopy

import numpy as np
import logging
from tqdm.auto import tqdm

from .scenarios import (
    Scenario,
    ScenarioSet,
)
from .lda_engine import lda_run_engine
from .config import OpRiskConfig

logger = logging.getLogger(__name__)


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

    config = deepcopy(base_config)

    config.setdefault("uom_overrides", {})
    config["uom_overrides"]["freq_multiplier"] = scenario.freq_multiplier
    config["uom_overrides"]["sev_mu_shift"] = scenario.sev_mu_shift
    config["uom_overrides"]["sev_scale_multiplier"] = scenario.sev_scale_multiplier

    result = lda_run_engine(config)
    runtime = time.perf_counter() - start

    # --- Normalise return: tuple or dict ---
    if isinstance(result, dict):
        metrics = result
    elif isinstance(result, tuple) and len(result) >= 3:
        metrics = result[2]
    else:
        raise ValueError(f"Unexpected lda_run_engine return format: {type(result)}")

    return scenario.name, metrics, runtime


class OpRiskStressTester:
    """
    Core stress-testing engine.
    """

    def __init__(self, config_path: str = "config/op_config.yaml"):
        self.config_path = Path(config_path)
        self._baseline_result: Optional[
            Tuple[np.ndarray, Dict[str, Dict[str, Any]], Dict[str, float]]
        ] = None
        self._base_config: Optional[Dict[str, Any]] = None

    @property
    def baseline(self) -> Dict[str, Any]:
        """
        Lazily compute and return baseline metrics dictionary.
        Accepts lda_run_engine returning either:
        - (dist, models, metrics) tuple  OR
        - metrics dict directly
        """
        if self._baseline_result is None:
            cfg = OpRiskConfig(self.config_path)
            cfg.validate()
            # store base config without running engine
            self._base_config = cfg.as_dict()
            # run engine and cache whatever it returns
            self._baseline_result = lda_run_engine(self._base_config)

        # If engine returned a 3-tuple, metrics are in index 2
        if isinstance(self._baseline_result, tuple) and len(self._baseline_result) >= 3:
            metrics = self._baseline_result[2]
            if isinstance(metrics, dict):
                return metrics
            # defensive fallback
            logger.error(
                "Baseline engine tuple did not contain metrics dict: %s", type(metrics)
            )
            return {}

        # If engine returned a dict (metrics) directly, return it
        if isinstance(self._baseline_result, dict):
            return self._baseline_result

        # last-resort fallback
        logger.error("Baseline result has unexpected format: %s", self._baseline_result)
        return {}

    @property
    def baseline_capital(self) -> float:
        metrics = self.baseline
        return float(metrics.get("capital_999", np.nan))

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
                    name, metrics, runtime = future.result()
                    cap_stressed = float(metrics.get("capital_999", np.nan))
                    scenario = next(s for s in scenarios if s.name == name)
                    results.append(self._make_result(scenario, cap_stressed, runtime))
        else:
            for scenario in tqdm(scenarios, desc="Running scenarios"):
                name, metrics, runtime = _run_single_scenario(
                    (self._base_config, scenario)
                )

                cap_stressed = float(metrics.get("capital_999", np.nan))
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
        """
        Return a shallow copy of the base config used for runs.
        If not yet loaded, build the config from OpRiskConfig without running the engine
        (useful for tests that monkeypatch lda_run_engine before baseline is called)
        """
        if self._base_config is None:
            cfg = OpRiskConfig(self.config_path)
            cfg.validate()
            self._base_config = cfg.as_dict()
        return dict(self._base_config)  # shallow copy

    def make_result_for_tests(self, scenario, cap_stressed, runtime):
        return self._make_result(scenario, cap_stressed, runtime)
