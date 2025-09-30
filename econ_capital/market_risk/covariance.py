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

from typing import Dict

import numpy as np
import pandas as pd


def ewma_cov(returns: pd.DataFrame, lamb: float) -> pd.DataFrame:
    """Exponentially weighted covariance.

    Parameters
    ----------
    returns : pd.DataFrame
        T x K matrix of factor returns.
    lamb : float
        Decay factor in (0,1). Higher => longer memory.

    Returns
    -------
    pd.DataFrame
        K x K covariance matrix.
    """
    x = returns.fillna(0.0).to_numpy()
    k = x.shape[1]
    s = np.zeros((k, k))
    for t in range(x.shape[0]):
        s = lamb * s + (1.0 - lamb) * np.outer(x[t], x[t])
    # Unbias for finite sample length
    s = s / (1.0 - lamb ** x.shape[0])
    return pd.DataFrame(s, index=returns.columns, columns=returns.columns)


def sample_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Sample covariance (unbiased)."""
    return returns.cov()


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
    for col in returns.columns:
        # returns in %, arch expects relatively larger magnitudes
        am = arch_model(returns[col] * 100.0, vol="Garch", p=1, q=1)
        res = am.fit(disp="off")
        vols[col] = float(res.conditional_volatility.iloc[-1]) / 100.0
    return pd.Series(vols)


def garch_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """GARCH vols x sample correlation → covariance snapshot."""
    vols = garch_vols(returns)
    corr = returns.corr()
    cov = np.outer(vols, vols) * corr.to_numpy()
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
