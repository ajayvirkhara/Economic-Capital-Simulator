"""
Public API façade for the Market Risk module.

Re-exports the key classes and functions from submodules so that users can
access the full market risk economic capital engine from a single namespace.

Exposed
-------
- MarketRiskEconomicCapital : Monte Carlo engine for market risk EC (VaR/ES)
- ewma_cov, sample_cov, garch_cov : covariance estimators
- mv_t_draws : multivariate Student-t shock generator
- left_tail_var, left_tail_es : tail risk statistics (VaR/ES)

Usage
-----
>>> from econ_capital.market_risk import MarketRiskEconomicCapital
>>> engine = MarketRiskEconomicCapital(risk_factors, positions)
>>> results = engine.run()
"""

from .engine import MarketRiskEconomicCapital
from .covariance import ewma_cov, sample_cov, garch_cov
from .shocks import mv_t_draws
from .stats import left_tail_var, left_tail_es

__all__ = [
    "MarketRiskEconomicCapital",
    "ewma_cov",
    "sample_cov",
    "garch_cov",
    "mv_t_draws",
    "left_tail_var",
    "left_tail_es",
]
