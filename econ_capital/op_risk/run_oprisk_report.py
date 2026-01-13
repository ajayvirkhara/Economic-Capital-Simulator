"""
OpRisk Stress Test + Regulatory Excel Report
"""

from __future__ import annotations

from pathlib import Path

import sys

import numpy as np
import yaml
import time
import logging

from econ_capital.op_risk.scenarios import (
    Scenario,
    build_scenario_set_from_data,
)
from econ_capital.op_risk.oprisk_reporting import generate_oprisk_report
from econ_capital.op_risk.stress_tests import OpRiskStressTester, StressTestResult
from econ_capital.op_risk.lda_engine import lda_run_engine

# Silence the noisy LDA engine logger
lda_logger = logging.getLogger("econ_capital.op_risk.lda_engine")
lda_logger.setLevel(logging.WARNING)
lda_logger.propagate = False


def main() -> float:
    # ──────────────────────────────────────────────────────────────
    # PATH & DEBUG SETUP
    # ──────────────────────────────────────────────────────────────
    start = time.perf_counter()

    print("\nSTARTING OPRISK REPORT GENERATION")
    print(f"Python: {sys.version}")
    print(f"Current working directory: {Path.cwd()}")
    print(f"Script location: {Path(__file__).resolve()}\n")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    print(f"Detected project root: {PROJECT_ROOT}")

    CONFIG_PATH = PROJECT_ROOT / "config" / "op_config.yaml"
    REPORT_DIR = PROJECT_ROOT / "econ_capital" / "op_risk" / "reports"
    REPORT_DIR.mkdir(exist_ok=True)

    print(f"Looking for config: {CONFIG_PATH}")
    if not CONFIG_PATH.exists():
        print("CONFIG FILE NOT FOUND!")
        sys.exit(1)
    else:
        print("Config file found!\n")

    # ──────────────────────────────────────────────────────────────
    # LOADING CONFIG
    # ──────────────────────────────────────────────────────────────

    print("Loading config...")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)["op_risk"]

    print("Config loaded successfully")
    print(f"Frequency data: {config['frequency']['data_path']}")
    print(f"Severity data:  {config['severity']['data_path']}\n")

    # ──────────────────────────────────────────────────────────────
    # EXPERT JUDGMENT SCENARIO CAPITAL (from config.scenarios)
    # ──────────────────────────────────────────────────────────────
    scenario_el = 0.0
    scenario_details = []

    yaml_expert_scenarios = config.get("scenarios", {})
    for name, s in yaml_expert_scenarios.items():
        prob = float(s.get("probability", 0.0))
        impact = float(s.get("impact", 0.0))
        annual_el = prob * impact
        scenario_el += annual_el
        scenario_details.append(
            {
                "name": name,
                "probability": prob,
                "impact": impact,
                "annual_el": annual_el,
            }
        )

    SCENARIO_CAPITAL_MULTIPLIER = 20
    scenario_capital = scenario_el * SCENARIO_CAPITAL_MULTIPLIER

    print("\n" + "=" * 70)
    print("EXPERT JUDGMENT SCENARIO CAPITAL")
    print("=" * 70)
    if scenario_details:
        for d in scenario_details:
            print(
                f"  • {d['name']:<15} {d['probability']:>7.1%} × £{d['impact']:>12,.0f} → £{d['annual_el']:>12,.0f}"
            )
        print(f"  {'TOTAL EL':<15} {'':>30} → £{scenario_el:>12,.0f}")
        print(
            f"  Scenario Capital (×{SCENARIO_CAPITAL_MULTIPLIER}): £{scenario_capital:>12,.0f}"
        )
    else:
        print("  No expert judgment scenarios defined")

    # Pass to reporting
    config["expert_scenario_capital"] = scenario_capital
    config["expert_scenario_el"] = scenario_el
    config["expert_scenario_details"] = scenario_details

    # ──────────────────────────────────────────────────────────────
    # DATA-DRIVEN SCENARIOS + YAML STRESS SCENARIOS
    # ──────────────────────────────────────────────────────────────
    data_set = build_scenario_set_from_data(
        freq_data_path=config["frequency"]["data_path"],
        sev_data_path=config["severity"]["data_path"],
        n_random=10,
        seed=42,
    )

    # Rename random scenarios to cleaner names
    for scen in data_set.scenarios:
        if scen.name.startswith("rand_"):
            num = scen.name.split("_")[1]
            scen.name = f"Random Scenario {num}"

    # Add YAML-defined stress scenarios (extreme_event etc.)
    yaml_scenarios = config.get("stress_tests", {})
    uom_keys = list(data_set.base_profile.keys())  # e.g. ["UoM1", "UoM2"]

    for name, sdef in yaml_scenarios.items():
        freq_mult = float(sdef.get("frequency_multiplier", 1.0))
        sev_mult = float(sdef.get("severity_multiplier", 1.0))

        data_set.scenarios.append(
            Scenario(
                name=name,
                freq_multiplier={u: freq_mult for u in uom_keys},
                sev_mu_shift={u: np.log(sev_mult) for u in uom_keys},
                sev_scale_multiplier={u: 1.0 for u in uom_keys},
                note=f"YAML-driven scenario: {name}",
            )
        )

    final_set = data_set
    print(f"Total scenarios (data + YAML): {len(final_set.scenarios)}")

    # ──────────────────────────────────────────────────────────────
    # RUN STRESS TESTS
    # ──────────────────────────────────────────────────────────────
    print("\nStarting stress testing and report generation...\n")
    tester = OpRiskStressTester(config_path=str(CONFIG_PATH))
    _ = tester.baseline  # Trigger baseline run to cache capital

    results = tester.run_scenario_set(final_set, parallel=True)

    print("\n=== DEBUG: STRESS TEST RESULTS ===")
    try:
        print(
            f"{'Scenario':<35} {'Base Capital':>15} {'Stressed Capital':>18} {'Uplift':>12} {'Time (s)':>10}"
        )
        print("-" * 95)
        for r in results:
            print(
                f"{r.name:<35} "
                f"{r.capital_base:>15,.0f} "
                f"{r.capital_stressed:>18,.0f} "
                f"{r.uplift_factor:>10.2f}x "
                f"{r.runtime_sec:>9.2f}s"
            )
    except Exception as e:
        print("Failed during stress testing:", e)
        raise

    print("=== END DEBUG ===\n")

    # ──────────────────────────────────────────────────────────────
    # GENERATE FINAL REPORT
    # ──────────────────────────────────────────────────────────────
    # Explicitly get full baseline metrics for firm-wide aggregation
    base_config = tester.get_base_config_for_tests()
    _, _, baseline_metrics = lda_run_engine(base_config)

    max_stressed = max(
        (r.capital_stressed for r in results), default=tester.baseline_capital
    )
    total_oprisk_capital = max(scenario_capital, max_stressed)
    results_dict = {
        "total_capital": total_oprisk_capital,
        "stress_test_results": results,
        "expert_scenario_details": config.get("expert_scenario_details", []),
        "expert_scenario_capital": scenario_capital,
        "baseline_metrics": baseline_metrics,
        "expected_loss": baseline_metrics.get("expected_loss", 0.0),
    }
    runtime = time.perf_counter() - start
    # ── Make stressed capitals sum to standalone OpRisk EC ────────────────────────── #

    standalone_oprisk_ec = results_dict.get(
        "standalone_oprisk_ec", total_oprisk_capital
    )
    if results:
        current_sum = sum(r.capital_stressed for r in results)
        if current_sum != 0 and abs(current_sum - standalone_oprisk_ec) > 1e5:
            scale = standalone_oprisk_ec / current_sum
            # Create new list with scaled instances (immutable → new objects)
            cap_stressed = r.capital_stressed * scale
            cap_base = r.capital_base
            scaled_results = []
            for r in results:
                new_result = StressTestResult(
                    name=r.name,
                    description=r.description,
                    capital_base=cap_base,
                    capital_stressed=cap_stressed,
                    absolute_uplift=cap_stressed - cap_base,
                    uplift_factor=r.uplift_factor * scale,
                    uplift_pct=(cap_stressed - cap_base) / cap_base
                    if cap_base > 0
                    else np.nan,
                    runtime_sec=runtime,
                )
                scaled_results.append(new_result)
            results = scaled_results  # replace original list
            print(
                f"Scaled OpRisk stressed capitals to sum to standalone £{standalone_oprisk_ec:,.0f} "
                f"(factor: {scale:.3f})"
            )

    # Generate Excel report
    generate_oprisk_report(
        tester=tester,
        results=results,
        config=config,
        output_dir=str(REPORT_DIR),
    )

    print("\nREPORT SUCCESSFULLY GENERATED!")

    # Print results summary
    print("\n=== Operational Risk Economic Capital Summary ===")
    print(f"Total OpRisk Capital (incl. stress): £{total_oprisk_capital:,.0f}")
    print(f"Baseline Expected Loss: £{baseline_metrics.get('expected_loss', 0):,.0f}")
    print(
        f"Max Stressed Capital Uplift: £{max_stressed - tester.baseline_capital:,.0f}"
    )

    return results_dict


if __name__ == "__main__":
    results = main()
