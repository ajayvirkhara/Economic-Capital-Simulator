"""
Market Risk EC Report Driver
"""

from __future__ import annotations

from pathlib import Path
import sys
import yaml

from econ_capital.market_risk.data_loaders import (
    load_real_risk_factors,
    load_dummy_positions,
)
from econ_capital.market_risk.engine import MarketRiskEconomicCapital
from econ_capital.market_risk.reporting import generate_market_risk_report

# ──────────────────────────────────────────────────────────────
# DEBUGGING
# ──────────────────────────────────────────────────────────────
print("\nSTARTING MARKET RISK REPORT GENERATION")
print(f"Python: {sys.version}")
print(f"Current working directory: {Path.cwd()}")
print(f"Script location: {Path(__file__).resolve()}\n")

if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
else:
    SCRIPT_DIR = Path.cwd()

PROJECT_ROOT = (
    SCRIPT_DIR if (SCRIPT_DIR / "econ_capital").exists() else SCRIPT_DIR.parent
)
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
# REPORT GENERATION
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
    start=config.get("start_date", "2020-01-01"),
    end=config.get("end_date", "2025-01-01"),
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
            print(f"{k:<15} £{v:>18,.0f}")
        else:
            print(f"{k:<15} (DataFrame - see breakdown below)")

    print("\nCapital Breakdown (Top 10):")
    print(results["capital_breakdown"].head(10).to_string())
except Exception as e:
    print("Failed during simulation:", e)
    raise

print("=== END DEBUG ===\n")

# ──────────────────────────────────────────────────────────────
# GENERATE FINAL REPORT
# ──────────────────────────────────────────────────────────────
report_path = generate_market_risk_report(
    config=config,
    engine=engine,
    results=results,
    output_dir=str(REPORT_DIR),
)

print("\nREPORT SUCCESSFULLY GENERATED!")
print(f"Location: {report_path}")
