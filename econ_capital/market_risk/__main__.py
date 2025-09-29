"""Demo run for the Market Risk engine."""

from __future__ import annotations

from .market_risk import MarketRiskEconomicCapital
from .data_loaders import load_real_risk_factors, load_dummy_positions

if __name__ == "__main__":
    # Load data
    rf = load_real_risk_factors(start="2020-01-01", end="2025-01-01")
    pos = load_dummy_positions()

    # Configure & run
    engine = MarketRiskEconomicCapital(
        risk_factors=rf,
        positions=pos,
        config={"n_paths": 500_000, "seed": 123, "cov_method": "GARCH"},
    )
    results = engine.run()

    # Pretty print
    print(f"10D VaR (99.9%): {results['var_10d_999']:,.0f}")
    print(f"10D  ES (99.9%): {results['es_10d_999']:,.0f}")
    print(f"1Y  VaR (99.9%): {results['var_1y_999']:,.0f}")
    print(f"1Y   ES (99.9%): {results['es_1y_999']:,.0f}")
    print("\nCapital breakdown (top 10):")
    print(results["capital_breakdown"].head(10).to_string())
