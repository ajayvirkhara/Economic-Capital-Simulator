"""
Market Risk Economic Capital engine.

Provides the MarketRiskEconomicCapital class, which:
- Estimates factor covariance using configurable methods (EWMA, sample, GARCH)
- Simulates multivariate Student-t shocks over a chosen horizon
- Maps factor shocks into position and portfolio P&L
- Computes 10-day and 1-year VaR/ES
- Allocates ES to positions using Euler’s principle (mean tail P&L)

This is the main entry point for running a full market risk capital simulation.
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional
import numpy as np
import pandas as pd

from econ_capital.utils import setup_logging
from econ_capital.config_loader import merge_with_global
from .config import DEFAULT_CONFIG
from .covariance import ewma_cov, sample_cov, garch_cov
from .shocks import mv_t_draws
from .stats import left_tail_var, left_tail_es

logger = setup_logging(__name__)


@dataclass
class MarketRiskEconomicCapital:
    """Monte Carlo engine for market risk EC (VaR/ES) and Euler allocation."""

    risk_factors: pd.DataFrame
    positions: pd.DataFrame
    config: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        # Start profiling init to measure object setup overhead
        t0 = time.perf_counter()

        # Start with DEFAULT_CONFIG (module defaults)
        cfg = DEFAULT_CONFIG.copy()

        # If user passed config dict, merge it
        if self.config:
            cfg.update(self.config)

        # Else merge with global defaults from default.yaml
        self.config = merge_with_global(cfg)

        # Extract key simulation parameters with fallbacks
        sim = self.config.get("simulation", {})
        self.n_paths = sim.get("default_n_paths", 500_000)
        self.seed = self.config.get("seed", 42)
        self.horizon_days = sim.get("horizon_days", 10)
        self.var_q = sim.get("var_q", 0.999)

        # Create RNG using the final seed
        self.rng = np.random.default_rng(self.seed)

        # Cache factor names and number of factors
        self.factor_names = list(self.risk_factors.columns)
        self.n_factors = len(self.factor_names)

        # Build exposures matrices aligned to factor order
        self.delta, self.gamma, self.vega = self._build_exposures()

        # Log initialization timing for reproducibility and performance tracking
        elapsed = time.perf_counter() - t0
        logger.info(
            "Initialized MarketRiskEconomicCapital with %d factors, %d paths, seed=%d in %.3fs",
            self.n_factors,
            self.n_paths,
            self.seed,
            elapsed,
        )

    def _build_exposures(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Align delta/gamma/vega columns to factor order (missing → 0)."""

        # Log/debug matrix alignment to ensure deltas, gammas, vegas are correctly reshaped
        t0 = time.perf_counter()

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

        elapsed = time.perf_counter() - t0
        logger.debug(
            "Built exposures: delta=%s gamma=%s vega=%s elapsed=%.3fs",
            delta.shape,
            gamma.shape,
            vega.shape,
            elapsed,
        )
        return delta, gamma, vega

    def _estimate_mu_cov(self) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate factor mean vector and covariance according to config."""

        # Profile covariance estimation; critical for performance in large simulations
        t0 = time.perf_counter()

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

        elapsed = time.perf_counter() - t0
        # Log chosen covariance method and timing to validate config + track cost
        logger.info("Estimated mu/cov using %s in %.3fs", method, elapsed)
        return mu, cov

    def _simulate_shocks(self, n_paths: int) -> np.ndarray:
        """Accumulate daily multivariate t-shocks across the horizon."""

        # Measure simulation runtime (MC typically bottleneck)
        t0 = time.perf_counter()

        mu, cov = self._estimate_mu_cov()
        df = float(self.config["df_t"])
        horizon = int(self.config["horizon_days"])

        shocks = np.zeros((n_paths, self.n_factors))
        for _ in range(horizon):
            shocks += mv_t_draws(n_paths, mu, cov, df, self.rng)
        elapsed = time.perf_counter() - t0
        logger.info(
            "Simulated %d shocks over horizon=%d in %.3fs", n_paths, horizon, elapsed
        )
        return shocks

    def _pnl_from_shocks(self, shocks: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
        """Map factor shocks to position P&L and aggregate to portfolio."""

        # Debug  P&L generation
        t0 = time.perf_counter()

        pnl_positions = shocks @ self.delta.to_numpy().T
        pnl_positions += 0.5 * (shocks**2) @ self.gamma.to_numpy().T
        pnl_positions += shocks @ self.vega.to_numpy().T

        pnl_portfolio = pnl_positions.sum(axis=1)
        pnl_by_position = pd.DataFrame(pnl_positions, columns=self.positions.index)
        elapsed = time.perf_counter() - t0
        logger.debug(
            "Computed PnL from shocks: n=%d positions=%d elapsed=%.3fs",
            len(pnl_portfolio),
            pnl_by_position.shape[1],
            elapsed,
        )
        return pnl_portfolio, pnl_by_position

    def _allocate_euler_es(
        self, pnl_positions: pd.DataFrame, tail_mask: np.ndarray
    ) -> pd.Series:
        """Euler-ES: contribution = mean tail loss per position (positive = capital)."""

        # Profile Euler allocation to positions
        t0 = time.perf_counter()

        tail_pnl = pnl_positions[tail_mask]
        contrib = -tail_pnl.mean(axis=0)
        elapsed = time.perf_counter() - t0
        logger.debug(
            "Allocated Euler-ES to %d positions in %.3fs", len(contrib), elapsed
        )
        return contrib

    def run(self) -> Dict[str, Any]:
        """Run simulation, return VaR/ES (10D and scaled 1Y) + Euler allocation."""

        # Reproducibility fingerprint (log seed + config snapshot)
        logger.info(
            "Starting MarketRiskEconomicCapital run with seed=%s, n_paths=%s, horizon=%s, cov_method=%s",
            self.seed,
            self.n_paths,
            self.horizon_days,
            self.config.get("cov_method"),
        )

        # Profile entire run duration for high-level benchmarking
        t0 = time.perf_counter()
        shocks = self._simulate_shocks(self.n_paths)
        pnl_port, pnl_by_pos = self._pnl_from_shocks(shocks)

        q = float(self.config["var_q"])
        var_10d = left_tail_var(pnl_port, q)
        es_10d = left_tail_es(pnl_port, q)

        # 10D → 1Y scaling (sqrt time)
        scale = np.sqrt(self.config.get("scaling_days_year", 252) / self.horizon_days)
        var_1y = var_10d * scale
        es_1y = es_10d * scale

        # Tail quantile for tail statistics
        cutoff = np.quantile(pnl_port, 1.0 - q)
        tail_mask = pnl_port <= cutoff

        # ──────────────────────────────────────────────────────────────
        # Euler-style component ES — linear (delta) approximation
        # Average contribution of each position in the tail scenarios
        # ──────────────────────────────────────────────────────────────

        # Factor-level contribution (10-day average tail impact per factor)
        tail_shocks = shocks[tail_mask, :]  # (n_tail, n_factors)

        # Compute position-level P&L in tail scenarios (linear term only)
        tail_position_pnl = np.dot(
            tail_shocks, self.delta.T
        )  # shape (n_tail, n_positions)

        # Average loss contribution per position in the tail
        component_es = -tail_position_pnl.mean(axis=0)  # shape (n_positions,)

        # Create series with correct index
        capital_breakdown = pd.Series(
            component_es,
            index=self.positions.columns,
            name="Component ES 1Y (linear approx)",
        )

        # Final numerical cleanup (small adjustment)
        total_component_sum = capital_breakdown.sum()
        if abs(total_component_sum - es_1y) > 1e-6 * es_1y:
            scale = es_1y / total_component_sum if total_component_sum != 0 else 1.0
            capital_breakdown *= scale
            logger.info(
                "Component ES normalized to match portfolio ES (adjustment factor %.6f)",
                scale,
            )
        else:
            logger.info("Component ES already sums to portfolio ES (within tolerance)")

        # Sort descending for reporting (largest contributors first)
        capital_breakdown = capital_breakdown.sort_values(ascending=False)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Run completed: VaR10d=%.3f ES10d=%.3f VaR1y=%.3f ES1y=%.3f elapsed=%.3fs",
            var_10d,
            es_10d,
            var_1y,
            es_1y,
            elapsed,
        )

        # --- Stress Testing ---
        stressed_var_1y = stressed_es_1y = None
        stress_shocks = self.config.get("stress_shocks")
        stress_enabled = self.config.get("stress_enabled")
        if not isinstance(stress_shocks, dict) or not stress_shocks:
            logger.warning(
                "Stress testing enabled but no valid shocks defined; skipping."
            )
            stressed_var_1y = stressed_es_1y = 0.0
        if stress_enabled:
            logger.info("Applying predefined stress shocks for stress testing")

            # 1. Build a stressed Mean Vector aligned with factor names
            stressed_mu = np.zeros(self.n_factors)
            for i, name in enumerate(self.factor_names):
                # Look up the ticker in stress_shocks; divide by horizon for daily mean
                if name in stress_shocks:
                    stressed_mu[i] = stress_shocks[name] / self.horizon_days

            # 2. Get current covariance
            _, cov = self._estimate_mu_cov()

            # 3. Simulate new shocks centered around the stressed mean
            n_s = 50_000
            s_shocks = np.zeros((n_s, self.n_factors))
            for _ in range(self.horizon_days):
                s_shocks += mv_t_draws(
                    n_s, stressed_mu, cov, float(self.config["df_t"]), self.rng
                )

            # 4. Map to P&L and compute Tail Stats
            s_pnl, _ = self._pnl_from_shocks(s_shocks)

            # Calculate 1Y scaled results
            stressed_var_1y = left_tail_var(s_pnl, q) * scale
            stressed_es_1y = left_tail_es(s_pnl, q) * scale

            logger.info(
                "Stress Test Result: 1Y Stressed Capital = £%.0f", stressed_var_1y
            )

        return {
            "var_10d_999": float(var_10d),
            "es_10d_999": float(es_10d),
            "var_1y_999": float(var_1y),
            "es_1y_999": float(es_1y),
            "stressed_var_1y_999": float(stressed_var_1y)
            if stressed_var_1y is not None
            else None,
            "stressed_es_1y_999": float(stressed_es_1y)
            if stressed_es_1y is not None
            else None,
            "baseline_capital": 0.0,
            "capital_breakdown": capital_breakdown.sort_values(ascending=False),
        }
