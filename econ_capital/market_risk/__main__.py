# econ_capital/market_risk/__main__.py
from .market_risk import MarketRiskEconomicCapital
from .data_loaders import load_real_risk_factors, load_dummy_positions_real

if __name__ == "__main__":
    rf = load_real_risk_factors()
    pos = load_dummy_positions_real()
    engine = MarketRiskEconomicCapital(rf, pos, config={"n_paths": 500_000, "seed": 123, "cov_method": "GARCH"})
    results = engine.run()
    print(results)