"""
Standalone demo for ExposureEngine using stylised GBM paths.
"""

import numpy as np
from econ_capital.credit_risk.trade_models import Trade, NettingSet
from econ_capital.credit_risk.csa import CSA
from econ_capital.credit_risk.exposure_engine import ExposureEngine

if __name__ == "__main__":

    # Simple test
    _times = np.linspace(0, 1.0, 6)
    _n_paths = 5000
    _rng = np.random.default_rng(42)

    # Simulate single factor (SP500)
    S0, mu, sigma = 100.0, 0.0, 0.2
    dt = np.diff(np.concatenate([[0.0], _times]))
    z = _rng.standard_normal((_n_paths, len(_times)))
    S = np.empty_like(z)
    S[:, 0] = S0
    for k in range(1, len(_times)):
        S[:, k] = S[:, k - 1] * np.exp(
            (mu - 0.5 * sigma**2) * dt[k - 1] + sigma * np.sqrt(dt[k - 1]) * z[:, k - 1]
        )
    _market_paths = {"SP500": S}

    # Define trades
    _trades = [
        Trade(name="IRS_like", factor="SP500", w=0.8, gamma=0.0),
        Trade(name="Option_like", factor="SP500", w=0.1, gamma=0.002),
    ]
    _csa = CSA(threshold=1.0, mta=0.1, im=0.5, vm_calls_per_day=1)

    # Create netting set and engine
    _ns = NettingSet(counterparty="CPTY_A", trades=_trades, csa=_csa)
    engine = ExposureEngine(netting_set=_ns, market_paths=_market_paths, times=_times)
    _, summary = engine.compute_exposure_profile()

    print(summary.head())
