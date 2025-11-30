"""
Demo run for the Market Risk engine.

This script can be executed directly with:
    python -m econ_capital.market_risk
"""

from __future__ import annotations

from econ_capital.utils import setup_logging
from .engine import MarketRiskEconomicCapital
from .data_loaders import load_real_risk_factors, load_dummy_positions

logger = setup_logging(__name__)


def main() -> None:
    """Run a demo simulation of market risk economic capital."""
    # --- Load data ---
    risk_factors = load_real_risk_factors(start="2020-01-01", end="2025-01-01")
    positions = load_dummy_positions()

    # --- Configure engine ---
    engine = MarketRiskEconomicCapital(
        risk_factors=risk_factors,
        positions=positions,
        config={
            "n_paths": 500_000,
            "seed": 123,
            "cov_method": "GARCH",
        },
    )

    # --- Run simulation ---
    results = engine.run()

    # --- Print results ---
    print("=== Market Risk Economic Capital Results ===")
    print(f"10D VaR (99.9%): {results['var_10d_999']:,.0f}")
    print(f"10D  ES (99.9%): {results['es_10d_999']:,.0f}")
    print(f"1Y  VaR (99.9%): {results['var_1y_999']:,.0f}")
    print(f"1Y   ES (99.9%): {results['es_1y_999']:,.0f}\n")

    print("=== Capital Breakdown (Top 10) ===")
    print(results["capital_breakdown"].head(10).to_string())


if __name__ == "__main__":
    logger.info("Running Market Risk demo")
    main()

# Mark successful completing of the simulation for audit trail
logger.info("Simulation finished successfully.")
