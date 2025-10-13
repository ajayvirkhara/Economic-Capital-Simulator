"""
Covariance estimators for risk factor returns.

Functions
---------
- ewma_cov    : exponentially weighted covariance matrix
- sample_cov  : unbiased sample covariance
- garch_vols  : univariate GARCH(1,1) volatility estimates per factor
- garch_cov   : GARCH vols × sample correlation → covariance snapshot

Notes
-----
These estimators are used to calibrate the factor covariance matrix
for the MarketRiskEconomicCapital engine.
"""

from __future__ import annotations
from typing import Dict
import time

import numpy as np
import pandas as pd
from econ_capital.utils import setup_logging

logger = setup_logging(__name__)


def ewma_cov(returns: pd.DataFrame, lamb: float) -> pd.DataFrame:
    """Exponentially weighted covariance."""

    # Log EWMA covariance computation (decay factor + runtime)
    t0 = time.perf_counter()

    x = returns.fillna(0.0).to_numpy()
    k = x.shape[1]
    s = np.zeros((k, k))
    for t in range(x.shape[0]):
        s = lamb * s + (1.0 - lamb) * np.outer(x[t], x[t])
    s = s / (1.0 - lamb ** x.shape[0])  # unbias for finite sample length
    elapsed = time.perf_counter() - t0
    logger.debug("Computed EWMA covariance: shape=%s, elapsed=%.3fs", s.shape, elapsed)
    return pd.DataFrame(s, index=returns.columns, columns=returns.columns)


def sample_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Sample covariance (unbiased)."""

    # Log sample covariance computation (baseline estimator, usually very fast)
    t0 = time.perf_counter()

    cov = returns.cov()
    elapsed = time.perf_counter() - t0
    logger.debug(
        "Computed sample covariance: shape=%s, elapsed=%.3fs", cov.shape, elapsed
    )
    return cov


def garch_vols(returns: pd.DataFrame) -> pd.Series:
    """Estimate last conditional vol per factor via univariate GARCH(1,1).

    Notes
    -----
    - Scaling to percent for numerical stability in `arch`, then rescale back.
    - Lazy import of `arch` to avoid hard dependency when not used.
    """
    try:
        from arch import arch_model  # pylint: disable=import-outside-toplevel
    except Exception as exc:
        raise ImportError(
            "GARCH selected but 'arch' is not installed. "
            "Install with: pip install arch"
        ) from exc

    vols: Dict[str, float] = {}

    # Log GARCH volatility estimation per factor (vol level + runtime)
    t0 = time.perf_counter()

    for col in returns.columns:
        # returns in %, arch expects relatively larger magnitudes
        am = arch_model(returns[col] * 100.0, vol="Garch", p=1, q=1)
        res = am.fit(disp="off")
        vols[col] = float(res.conditional_volatility.iloc[-1]) / 100.0
        logger.debug("Fitted GARCH for %s: vol=%.5f", col, vols[col])

    elapsed = time.perf_counter() - t0
    logger.info("Computed GARCH vols for %d factors in %.2fs", len(vols), elapsed)
    return pd.Series(vols)


def garch_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """GARCH vols x sample correlation → covariance snapshot."""

    # Log GARCH covariance snapshot (correlation x volatilities + runtime)
    t0 = time.perf_counter()
    vols = garch_vols(returns)
    corr = returns.corr()
    cov = np.outer(vols, vols) * corr.to_numpy()
    elapsed = time.perf_counter() - t0
    logger.info(
        "Computed GARCH covariance: shape=%s, elapsed=%.3fs", cov.shape, elapsed
    )
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
