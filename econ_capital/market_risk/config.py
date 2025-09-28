from typing import Dict, Any

DEFAULT_CONFIG: Dict[str, Any] = {
    "n_paths": 500000,                  # number of Monte Carlo simulations
    "horizon_days": 10,                 # horizon for base VaR
    "var_q": 0.999,                     # confidence level for VaR/ES
    "scaling_days_year": 252,           # trading days in a year
    "df_t": 7.0,                        # degrees of freedom for Student-t (smaller = heavier tail)
    "cov_method": "EWMA",               # EWMA, SAMPLE, or GARCH covariance method
    "ewma_lambda": 0.97,                # decay factor for EWMA (smaller = more reactive to recent observations)
    "fix_mean": True,                   # assume zero mean shocks
    "seed": 42,                         # random seed (fixed for reproducibility)
    "allocation_method": "Euler-ES",    # allocation method to distribute portfolio ES amongst positions
}