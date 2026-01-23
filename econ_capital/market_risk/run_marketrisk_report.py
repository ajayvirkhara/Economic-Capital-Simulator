"""
Market Risk EC Report Driver
"""

from __future__ import annotations

from pathlib import Path
import sys
import yaml
import pandas as pd

from econ_capital.market_risk.data_loaders import (
    load_real_risk_factors,
    load_dummy_positions,
)
from econ_capital.market_risk.engine import MarketRiskEconomicCapital
from econ_capital.market_risk.marketrisk_reporting import generate_market_risk_report
from econ_capital.utils import setup_logging

logger = setup_logging(__name__)


def main() -> dict:
    # ──────────────────────────────────────────────────────────────
    # DEBUGGING
    # ──────────────────────────────────────────────────────────────
    print("\nSTARTING MARKET RISK REPORT GENERATION")
    print(f"Python: {sys.version}")
    print(f"Current working directory: {Path.cwd()}")
    print(f"Script location: {Path(__file__).resolve()}\n")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    print(f"Detected project root: {PROJECT_ROOT}")

    CONFIG_PATH = PROJECT_ROOT / "config" / "market_config.yaml"
    REPORT_DIR = PROJECT_ROOT / "econ_capital" / "market_risk" / "reports"
    REPORT_DIR.mkdir(exist_ok=True)

    print(f"Looking for config: {CONFIG_PATH}")
    if not CONFIG_PATH.exists():
        print("CONFIG FILE NOT FOUND!")
        sys.exit(1)
    else:
        print("Config file found!\n")

    # ──────────────────────────────────────────────────────────────
    # LOAD CONFIG
    # ──────────────────────────────────────────────────────────────

    print("Loading config...")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)["market_risk"]

    print("Config loaded successfully\n")

    # ──────────────────────────────────────────────────────────────
    # LOAD DATA & RUN ENGINE
    # ──────────────────────────────────────────────────────────────
    print("\nLoading data...")
    risk_factors = load_real_risk_factors(
        start=config.get("start_date", "2021-01-01"),
        end=config.get("end_date", "2026-01-01"),
    )
    positions = load_dummy_positions()

    print("Data loaded successfully\n")

    print("\nStarting Market Risk EC simulation...\n")
    engine = MarketRiskEconomicCapital(
        risk_factors=risk_factors,
        positions=positions,
        config=config,
    )

    print("\n=== DEBUG: MARKET RISK RESULTS ===")
    try:
        results = engine.run()

        print(f"{'Metric':<15} {'Value':>20}")
        print("-" * 40)
        for k, v in results.items():
            if isinstance(v, (int, float)):
                print(f"{k:<25} £{v:>18,.0f}")
            else:
                print(f"{k:<25} (DataFrame - see breakdown below)")

        print("\nCapital Breakdown (Top 10):")
        print(results["capital_breakdown"].head(10).to_string())

        # Printing CoVaR metrics
        if "covar_metrics" in results and results["covar_metrics"]:
            print("\n" + "=" * 60)
            print("CoVaR SYSTEMIC RISK METRICS")
            print("=" * 60)
            covar_df = pd.DataFrame(results["covar_metrics"]).T
            print(covar_df.to_string())
            print("\nInterpretation:")
            print("- CoVaR: Portfolio VaR conditional on position being stressed")
            print("- ΔCoVaR: Incremental systemic contribution")
            print("- Systemic %: ΔCoVaR as % of portfolio VaR")

    except Exception as e:
        print("Failed during simulation:", e)
        raise

    logger.debug("=== END DEBUG ===")

    # ──────────────────────────────────────────────────────────────
    # GENERATE FINAL REPORT
    # ──────────────────────────────────────────────────────────────
    generate_market_risk_report(
        config=config,
        engine=engine,
        results=results,
        output_dir=str(REPORT_DIR),
    )

    print("\nREPORT SUCCESSFULLY GENERATED!")

    return results


if __name__ == "__main__":
    main()
