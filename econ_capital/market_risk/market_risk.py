"""Market Risk Economic Capital engine (Monte Carlo, 99.9% VaR/ES).

Features
--------
- Covariance options: EWMA, SAMPLE, GARCH(1,1) vols + sample correlation
- Multivariate Student-t shocks (df configurable)
- 10-day simulation with square-root-of-time scaling to 1-year
- Euler-ES allocation across positions (mean tail P&L)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG

# pylint: disable=import-outside-toplevel


# ---------------------------------------------------------------------
# Utility: covariance estimators
# ---------------------------------------------------------------------
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
        from arch import arch_model
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


# ---------------------------------------------------------------------
# Utility: multivariate Student-t draws & tail risk stats
# ---------------------------------------------------------------------
def mv_t_draws(
    n: int,
    mu: np.ndarray,
    cov: np.ndarray,
    df: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draws from a multivariate Student-t via normal/chi-square mixture."""
    g = rng.chisquare(df, size=n) / df  # shape (n,)
    z = rng.multivariate_normal(mean=np.zeros(cov.shape[0]), cov=cov, size=n)
    return mu + z / np.sqrt(g)[:, None]


def left_tail_var(pnl: np.ndarray, q: float) -> float:
    """Left-tail VaR at confidence q."""
    return -np.quantile(pnl, 1.0 - q)


def left_tail_es(pnl: np.ndarray, q: float) -> float:
    """Left-tail Expected Shortfall at confidence q."""
    cutoff = np.quantile(pnl, 1.0 - q)
    return -pnl[pnl <= cutoff].mean()


# ---------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------
@dataclass
class MarketRiskEconomicCapital:
    """Monte Carlo engine for market risk EC (VaR/ES) and Euler allocation.

    Parameters
    ----------
    risk_factors : pd.DataFrame
        T x K return series of the risk factors (columns = factor names).
    positions : pd.DataFrame
        Position exposures with rows as positions and columns:
        - linear deltas per factor:       col = factor name (e.g., "SPY")
        - optional quadratic gammas:      col = f"gamma_{factor}"
        - optional vega sensitivities:    col = f"vega_{factor}"
    config : Dict[str, Any], optional
        Overrides DEFAULT_CONFIG.
    """

    risk_factors: pd.DataFrame
    positions: pd.DataFrame
    config: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        # Merge config with defaults
        cfg = DEFAULT_CONFIG.copy()
        if self.config:
            cfg.update(self.config)
        self.config = cfg

        self.rng = np.random.default_rng(self.config["seed"])

        # Cache factor names and number of factors
        self.factor_names = list(self.risk_factors.columns)
        self.n_factors = len(self.factor_names)

        # Build exposures matrices aligned to factor order
        self.delta, self.gamma, self.vega = self._build_exposures()

    def _build_exposures(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Align delta/gamma/vega columns to factor order (missing → 0)."""
        # Linear (delta)
        delta = self.positions.reindex(columns=self.factor_names).fillna(0.0)

        # Quadratic (gamma)
        gamma_cols = [f"gamma_{f}" for f in self.factor_names]
        gamma = self.positions.reindex(columns=gamma_cols).fillna(0.0).copy()
        gamma.columns = self.factor_names  # rename back to factor names

        # Vega
        vega_cols = [f"vega_{f}" for f in self.factor_names]
        vega = self.positions.reindex(columns=vega_cols).fillna(0.0).copy()
        vega.columns = self.factor_names

        return delta, gamma, vega

    def _estimate_mu_cov(self) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate factor mean vector and covariance according to config."""
        rf = self.risk_factors.copy().dropna()

        mu = (
            np.zeros(self.n_factors)
            if self.config["fix_mean"]
            else rf.mean().to_numpy()
        )

        method = str(self.config["cov_method"]).upper()
        if method == "EWMA":
            cov_df = ewma_cov(rf, float(self.config["ewma_lambda"]))
        elif method == "GARCH":
            cov_df = garch_cov(rf)
        elif method == "SAMPLE":
            cov_df = sample_cov(rf)
        else:
            raise ValueError(f"Unsupported cov_method: {self.config['cov_method']}")

        cov = cov_df.to_numpy()
        # Numerical jitter for Cholesky stability
        cov += 1e-8 * np.eye(self.n_factors)
        return mu, cov

    def _simulate_shocks(self, n_paths: int) -> np.ndarray:
        """Accumulate daily multivariate t-shocks across the horizon."""
        mu, cov = self._estimate_mu_cov()
        df = float(self.config["df_t"])
        horizon = int(self.config["horizon_days"])

        shocks = np.zeros((n_paths, self.n_factors))
        for _ in range(horizon):
            shocks += mv_t_draws(n_paths, mu, cov, df, self.rng)
        return shocks

    def _pnl_from_shocks(self, shocks: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
        """Map factor shocks to position P&L and aggregate to portfolio."""
        pnl_positions = shocks @ self.delta.to_numpy().T
        pnl_positions += 0.5 * (shocks**2) @ self.gamma.to_numpy().T
        pnl_positions += shocks @ self.vega.to_numpy().T

        pnl_portfolio = pnl_positions.sum(axis=1)
        pnl_by_position = pd.DataFrame(pnl_positions, columns=self.positions.index)
        return pnl_portfolio, pnl_by_position

    def _allocate_euler_es(
        self, pnl_positions: pd.DataFrame, tail_mask: np.ndarray
    ) -> pd.Series:
        """Euler-ES: contribution = mean tail loss per position (positive = capital)."""
        tail_pnl = pnl_positions[tail_mask]
        contrib = -tail_pnl.mean(axis=0)
        return contrib

    def run(self) -> Dict[str, Any]:
        """Run simulation, return VaR/ES (10D and scaled 1Y) + Euler allocation."""
        shocks = self._simulate_shocks(int(self.config["n_paths"]))
        pnl_port, pnl_by_pos = self._pnl_from_shocks(shocks)

        q = float(self.config["var_q"])
        var_10d = left_tail_var(pnl_port, q)
        es_10d = left_tail_es(pnl_port, q)

        # 10D → 1Y scaling (sqrt time)
        scale = np.sqrt(self.config["scaling_days_year"] / self.config["horizon_days"])
        var_1y = var_10d * scale
        es_1y = es_10d * scale

        # Tail set for Euler-ES
        cutoff = np.quantile(pnl_port, 1.0 - q)
        tail_mask = pnl_port <= cutoff
        contrib = self._allocate_euler_es(pnl_by_pos, tail_mask)

        return {
            "var_10d_999": float(var_10d),
            "es_10d_999": float(es_10d),
            "var_1y_999": float(var_1y),
            "es_1y_999": float(es_1y),
            "capital_breakdown": contrib.sort_values(ascending=False),
        }
