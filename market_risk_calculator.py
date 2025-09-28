"""
Market Risk Economic Capital (EC) Module with EWMA, Sample, and GARCH covariance options
Coverage: Equities, Fixed Income, FX, Derivatives (delta–gamma–vega)
Confidence: 99.9% (left-tail)
Method: Monte Carlo simulations on risk factors, 10-day horizon, scaled to 1-year
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any
import yfinance as yf
from arch import arch_model

# ---------------------------------------------------
# Default configuration parameters
# ---------------------------------------------------
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

# ---------------------------------------------------
# Utility functions
# ---------------------------------------------------
def ewma_cov(returns: pd.DataFrame, lamb: float = 0.97) -> pd.DataFrame:
    """Exponentially Weighted Moving Average covariance."""
    x = returns.fillna(0.0).to_numpy()
    s = np.zeros((x.shape[1], x.shape[1]))
    for t in range(x.shape[0]):
        s = lamb * s + (1 - lamb) * np.outer(x[t], x[t])
    return pd.DataFrame(s / (1 - lamb ** x.shape[0]), index=returns.columns, columns=returns.columns)

def sample_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Sample covariance matrix."""
    return returns.cov()

def garch_vols(returns: pd.DataFrame) -> pd.Series:
    """Estimate conditional volatility for each factor using univariate GARCH(1,1)."""
    vols = {}
    for col in returns.columns:
        am = arch_model(returns[col] * 100, vol='Garch', p=1, q=1)  # scale returns to % for stability
        res = am.fit(disp="off")
        cond_vol = res.conditional_volatility
        vols[col] = cond_vol.iloc[-1] / 100.0   # rescale back
    return pd.Series(vols)

def garch_cov(returns: pd.DataFrame) -> pd.DataFrame:
    # Build covariance matrix: vol from GARCH, correlation from sample
    vols = garch_vols(returns)
    corr = returns.corr()
    cov = np.outer(vols, vols) * corr.to_numpy()
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)

def mv_t_draws(n: int, mu: np.ndarray, cov: np.ndarray, df: float, rng: np.random.Generator) -> np.ndarray:
    g = rng.chisquare(df, size=n) / df
    z = rng.multivariate_normal(np.zeros(cov.shape[0]), cov, size=n)
    return mu + z / np.sqrt(g)[:, None]

# Defining VaR function
def left_tail_var(pnl: np.ndarray, q: float) -> float:
    return -np.quantile(pnl, 1 - q)

# Defining ES function
def left_tail_es(pnl: np.ndarray, q: float) -> float:
    cutoff = np.quantile(pnl, 1 - q)
    return -pnl[pnl <= cutoff].mean()

# ---------------------------------------------------
# Main Engine
# ---------------------------------------------------
@dataclass
class MarketRiskEconomicCapital:
    risk_factors: pd.DataFrame
    positions: pd.DataFrame
    config: Dict[str, Any] = None

    def __post_init__(self):
        if self.config is None:
            self.config = DEFAULT_CONFIG.copy()
        else:
            cfg = DEFAULT_CONFIG.copy()
            cfg.update(self.config)
            self.config = cfg
        self.rng = np.random.default_rng(self.config["seed"])
        self.factor_names = list(self.risk_factors.columns)
        self.K = len(self.factor_names)
        self.delta, self.gamma, self.vega = self._build_exposures()

    def _build_exposures(self):
        delta = self.positions.reindex(columns=self.factor_names).fillna(0.0)
        gamma = self.positions.reindex(columns=[f"gamma_{f}" for f in self.factor_names]).fillna(0.0)
        gamma.columns = self.factor_names
        vega = self.positions.reindex(columns=[f"vega_{f}" for f in self.factor_names]).fillna(0.0)
        vega.columns = self.factor_names
        return delta, gamma, vega

    def _estimate_mu_cov(self):
        rf = self.risk_factors.copy().dropna()
        mu = np.zeros(self.K) if self.config["fix_mean"] else rf.mean().to_numpy()
        if self.config["cov_method"] == "EWMA":
            cov = ewma_cov(rf, self.config["ewma_lambda"]).to_numpy()
        elif self.config["cov_method"] == "GARCH":
            cov = garch_cov(rf).to_numpy()
        else:
            cov = sample_cov(rf).to_numpy()
        return mu, cov + 1e-8 * np.eye(self.K)

    def _simulate_shocks(self, n_paths: int) -> np.ndarray:
        mu, cov = self._estimate_mu_cov()
        df = float(self.config["df_t"])
        H = int(self.config["horizon_days"])
        shocks = np.zeros((n_paths, self.K))
        for d in range(H):
            shocks += mv_t_draws(n_paths, mu, cov, df, self.rng)
        return shocks

    def _pnl_from_shocks(self, shocks: np.ndarray):
        dF = shocks
        pnl_pos = dF @ self.delta.to_numpy().T
        pnl_pos += 0.5 * (dF ** 2) @ self.gamma.to_numpy().T
        pnl_pos += dF @ self.vega.to_numpy().T
        pnl_port = pnl_pos.sum(axis=1)
        return pnl_port, pd.DataFrame(pnl_pos, columns=self.positions.index)

    def _allocate_euler_es(self, pnl_pos: pd.DataFrame, tail_mask: np.ndarray) -> pd.Series:
        tail_pnl = pnl_pos[tail_mask]
        contrib = -tail_pnl.mean(axis=0)
        return contrib

    def run(self) -> Dict[str, Any]:
        shocks = self._simulate_shocks(int(self.config["n_paths"]))
        pnl_port, pnl_by_pos = self._pnl_from_shocks(shocks)
        q = float(self.config["var_q"])
        var_10d = left_tail_var(pnl_port, q)
        es_10d = left_tail_es(pnl_port, q)
        scale = np.sqrt(self.config["scaling_days_year"] / self.config["horizon_days"])
        var_1y = var_10d * scale
        es_1y = es_10d * scale
        cutoff = np.quantile(pnl_port, 1 - q)
        tail_mask = pnl_port <= cutoff
        contrib = self._allocate_euler_es(pnl_by_pos, tail_mask)
        return {
            "var_10d_999": float(var_10d),
            "es_10d_999": float(es_10d),
            "var_1y_999": float(var_1y),
            "es_1y_999": float(es_1y),
            "capital_breakdown": contrib.sort_values(ascending=False),
        }

# ---------------------------------------------------
# Realistic multi-asset loader using yfinance
# ---------------------------------------------------
def load_real_risk_factors(start="2020-01-01", end="2025-01-01") -> pd.DataFrame:
    tickers = {
        "SPY": "Equities (US)",
        "EFA": "Equities (Developed ex-US)",
        "EEM": "Equities (EM)",
        "TLT": "US Treasuries (long duration)",
        "LQD": "IG Credit ETF",
        "HYG": "HY Credit ETF",
        "GLD": "Gold",
        "USO": "Oil",
        "EURUSD=X": "EURUSD",
        "GBPUSD=X": "GBPUSD",
    }
    data = yf.download(list(tickers.keys()), start=start, end=end)
    if "Adj Close" in data:
        prices = data["Adj Close"].dropna()
    else:
        prices = data["Close"].dropna()
    returns = prices.pct_change().dropna()
    returns.columns = list(tickers.keys())
    return returns

def load_dummy_positions_real() -> pd.DataFrame:
    idx = ["Equity_US", "Equity_EM", "Rates", "Credit_IG", "Credit_HY", "Gold", "Oil", "FX_EURUSD", "FX_GBPUSD"]
    df = pd.DataFrame(index=idx)
    df["SPY"] = [5_000_000, 0, 0, 0, 0, 0, 0, 0, 0]
    df["EEM"] = [0, 2_000_000, 0, 0, 0, 0, 0, 0, 0]
    df["TLT"] = [0, 0, -10_000.0, 0, 0, 0, 0, 0, 0]
    df["LQD"] = [0, 0, 0, -3_000.0, 0, 0, 0, 0, 0]
    df["HYG"] = [0, 0, 0, 0, -4_000.0, 0, 0, 0, 0]
    df["GLD"] = [0, 0, 0, 0, 0, 1_000_000, 0, 0, 0]
    df["USO"] = [0, 0, 0, 0, 0, 0, 500_000, 0, 0]
    df["EURUSD=X"] = [0, 0, 0, 0, 0, 0, 0, 2_000_000, 0]
    df["GBPUSD=X"] = [0, 0, 0, 0, 0, 0, 0, 0, 1_500_000]
    return df.fillna(0.0)

# ---------------------------------------------------
# Example run
# ---------------------------------------------------
if __name__ == "__main__":
    rf = load_real_risk_factors()
    pos = load_dummy_positions_real()
    engine = MarketRiskEconomicCapital(rf, pos, config={"n_paths": 500_000, "seed": 123, "cov_method": "GARCH"})
    results = engine.run()
    print(results)