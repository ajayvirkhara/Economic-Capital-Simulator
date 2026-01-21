"""
Market Risk Economic Capital engine.

Provides the MarketRiskEconomicCapital class, which:
- Estimates factor covariance using configurable methods (EWMA, sample, GARCH)
- Simulates multivariate Student-t shocks over a chosen horizon
- Maps factor shocks into position and portfolio P&L
- Computes 10-day and 1-year VaR/ES
- Allocates ES to positions using Euler's principle (mean tail P&L)

This is the main entry point for running a full market risk capital simulation.
"""

from __future__ import annotations
import time
from typing import Any, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import yfinance as yf

from econ_capital.utils import setup_logging
from econ_capital.config_loader import merge_with_global
from .config import DEFAULT_CONFIG, load_market_yaml
from .covariance import ewma_cov, sample_cov, garch_cov
from .shocks import mv_t_draws
from .stats import left_tail_var, left_tail_es, compute_covar

logger = setup_logging(__name__)


class MarketRiskEconomicCapital:
    """Monte Carlo engine for market risk EC (VaR/ES) and Euler allocation."""

    def __init__(
        self,
        risk_factors: pd.DataFrame,
        positions: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None,
        pricing_portfolio=None,
    ):
        """
        Initialize Market Risk Engine.

        Parameters
        ----------
        risk_factors : pd.DataFrame
            Historical factor returns
        positions : pd.DataFrame
            Position data with delta/gamma/vega exposures
        config : dict, optional
            Configuration parameters
        pricing_portfolio : Portfolio, optional
            Full revaluation pricing portfolio (if use_full_revaluation=True)
        """
        # Start profiling init to measure object setup overhead
        t0 = time.perf_counter()

        # Start with DEFAULT_CONFIG (module defaults)
        cfg = DEFAULT_CONFIG.copy()

        # Load and merge Market Risk YAML parameters
        yaml_settings = load_market_yaml("config/market_config.yaml")
        cfg.update(yaml_settings)

        # If user passed config dict, merge it
        if config:
            cfg.update(config)

        # Merge with global defaults from default.yaml
        self.config = merge_with_global(cfg)

        # Store inputs
        self.risk_factors = risk_factors
        self.positions = positions

        # Extract key simulation parameters with fallbacks
        sim = self.config.get("simulation", {})
        global_config = self.config.get("global", {})
        self.n_paths = sim.get("default_n_paths", 500_000)
        self.seed = global_config.get("seed", 42)
        self.horizon_days = sim.get("default_horizon_days", 10)
        self.var_q = sim.get("var_q", 0.999)

        # Optional features
        self.use_full_revaluation = self.config.get("use_full_revaluation", True)
        self.use_historical_var = self.config.get("use_historical_var", False)
        self.compute_covar_flag = self.config.get("compute_covar", True)

        # Create RNG using the final seed
        self.rng = np.random.default_rng(self.seed)

        # Cache factor names and number of factors
        self.factor_names = list(self.risk_factors.columns) 
        self.n_factors = len(self.factor_names)

        # Build exposures matrices aligned to factor order
        self.delta, self.gamma, self.vega = self._build_exposures()

        # Handle pricing portfolio
        if self.use_full_revaluation:
            if pricing_portfolio is None:
                # Auto-build from positions
                self.pricing_portfolio = self._build_pricing_portfolio()
            else:
                self.pricing_portfolio = pricing_portfolio
        else:
            self.pricing_portfolio = pricing_portfolio

        # Log initialization timing for reproducibility and performance tracking
        elapsed = time.perf_counter() - t0
        logger.info(
            "Initialized MarketRiskEconomicCapital with %d factors, %d paths, seed=%d in %.3fs",
            self.n_factors,
            self.n_paths,
            self.seed,
            elapsed,
        )

    def _build_pricing_portfolio(self):
        """Convert positions dict to Portfolio with pricing classes from script."""
        from econ_capital.market_risk.marketrisk_pricing import (
            Portfolio,
            EquityPosition,
            BondPosition,
            EuropeanOption,
            FXForward,
            InterestRateSwap,
        )

        portfolio = Portfolio()

        for name, pos_data in self.positions.items():
            p_type = pos_data.get("type", "").lower()

            # 1. Equities
            if p_type == "equity":
                portfolio.add_position(
                    EquityPosition(
                        quantity=pos_data["quantity"],
                        current_price=pos_data["price"],
                        underlying_factor=pos_data["factor"],
                    )
                )

            # 2. Bonds (Fixed Income)
            elif p_type == "bond":
                portfolio.add_position(
                    BondPosition(
                        notional=pos_data["notional"],
                        current_price=pos_data["price"],
                        modified_duration=pos_data["duration"],
                        convexity=pos_data.get("convexity", 0.0),
                        yield_factor=pos_data["factor"],
                        current_yield=pos_data["yield"],
                    )
                )

            # 3. FX Forwards
            elif p_type == "fxforward":
                portfolio.add_position(
                    FXForward(
                        notional=pos_data["notional"],
                        strike=pos_data["strike"],
                        maturity=pos_data["maturity"],
                        fx_spot_factor=pos_data["factor"],
                        domestic_rate=pos_data.get("r_dom", 0.02),
                        foreign_rate=pos_data.get("r_for", 0.01),
                        current_spot=pos_data["spot"],
                    )
                )

            # 4. European Options
            elif p_type == "option":
                opt = EuropeanOption(
                    strike=pos_data["strike"],
                    maturity=pos_data["maturity"],
                    option_type=pos_data["option_type"],
                    volatility=pos_data["vol"],
                    quantity=pos_data["quantity"],
                    underlying_factor=pos_data["factor"],
                    risk_free_rate=pos_data.get("rf", 0.02),
                )
                opt.current_spot = pos_data["spot"]  # Setting required attribute
                portfolio.add_position(opt)

            # 5. Interest Rate Swaps
            elif p_type == "swap":
                portfolio.add_position(
                    InterestRateSwap(
                        notional=pos_data["notional"],
                        fixed_rate=pos_data["fixed_rate"],
                        tenor_years=pos_data["tenor"],
                        rate_factor=pos_data["factor"],
                        current_rate=pos_data["rate"],
                    )
                )

            # 6. Fallback to Linear Revaluation
            else:
                # Fallback uses EquityPosition for a simple linear ΔP = Q * ΔS
                portfolio.add_position(
                    EquityPosition(
                        quantity=pos_data.get("quantity", 1.0),
                        current_price=pos_data.get("price", 100.0),
                        underlying_factor=pos_data.get("factor", "UNKNOWN"),
                    )
                )

        return portfolio

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

    def _pnl_from_shocks(
        self, shocks: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Map factor shocks to position P&L and aggregate to portfolio."""

        # Debug  P&L generation
        t0 = time.perf_counter()

        pnl_positions = shocks @ self.delta.to_numpy().T
        pnl_positions += 0.5 * (shocks**2) @ self.gamma.to_numpy().T
        pnl_positions += shocks @ self.vega.to_numpy().T

        pnl_portfolio = pnl_positions.sum(axis=1)

        # Convert to dict for consistency with full revaluation path
        pnl_by_position = {
            pos_name: pnl_positions[:, i]
            for i, pos_name in enumerate(self.positions.index)
        }

        elapsed = time.perf_counter() - t0
        logger.debug(
            "Computed PnL from shocks: n=%d positions=%d elapsed=%.3fs",
            len(pnl_portfolio),
            len(pnl_by_position),
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
        """
        Run simulation, return VaR/ES (10D and scaled 1Y) + Euler allocation.

        Returns
        -------
        dict with keys:
            - var_10d_999, es_10d_999 (10-day metrics)
            - var_1y_999, es_1y_999 (1-year scaled)
            - stressed_var_1y_999, stressed_es_1y_999 (if stress enabled)
            - capital_breakdown (Euler allocation)
            - historical_var_1y_999, historical_es_1y_999 (if enabled)
            - covar_metrics (if enabled)
        """
        logger.info(
            "Starting MarketRiskEconomicCapital run with seed=%s, n_paths=%s, horizon=%s, cov_method=%s",
            self.seed,
            self.n_paths,
            self.horizon_days,
            self.config.get("cov_method"),
        )

        t0 = time.perf_counter()

        # ================================================================
        # MAIN PARAMETRIC VAR/ES
        # ================================================================
        parametric_results = self._run_parametric()

        # ================================================================
        # OPTIONAL: Historical VaR/ES
        # ================================================================
        if self.use_historical_var:
            logger.info("Computing Historical VaR/ES...")
            historical_results = self.compute_historical_var()
            parametric_results.update(historical_results)

        # ================================================================
        # OPTIONAL: CoVaR Systemic Risk Metrics
        # ================================================================
        if self.compute_covar_flag and hasattr(self, "pnl_by_pos"):
            logger.info("Computing CoVaR systemic risk metrics...")
            covar_results = self._compute_covar_metrics()
            parametric_results["covar_metrics"] = covar_results

        elapsed = time.perf_counter() - t0
        logger.info("Total run completed in %.3fs", elapsed)

        return parametric_results

    def _run_parametric(self) -> Dict[str, Any]:
        """Run simulation, return VaR/ES (10D and scaled 1Y) + Euler allocation."""

        shocks = self._simulate_shocks(self.n_paths)

        # Compute P&L - route to full revaluation if enabled
        if self.use_full_revaluation:
            logger.info("Using full revaluation pricing")
            pnl_port, pnl_by_pos = self._pnl_from_shocks_full_revaluation(shocks)
        else:
            logger.info("Using delta-gamma approximation")
            pnl_port, pnl_by_pos = self._pnl_from_shocks(shocks)

        # Store for CoVaR computation
        self.pnl_port = pnl_port
        self.pnl_by_pos = pnl_by_pos

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
            tail_shocks, self.delta.to_numpy().T
        )  # shape (n_tail, n_positions)

        # Average loss contribution per position in the tail
        component_es = -tail_position_pnl.mean(axis=0)  # shape (n_positions,)

        # Create series with correct index
        capital_breakdown = pd.Series(
            component_es,
            index=self.delta.index,
            name="Component ES 1Y (linear approx)",
        )

        # Final numerical cleanup (small adjustment)
        total_component_sum = capital_breakdown.sum()
        if abs(total_component_sum - es_1y) > 1e-6 * es_1y:
            scale_factor = (
                es_1y / total_component_sum if total_component_sum != 0 else 1.0
            )
            capital_breakdown *= scale_factor
            logger.info(
                "Component ES normalized to match portfolio ES (adjustment factor %.6f)",
                scale_factor,
            )
        else:
            logger.info("Component ES already sums to portfolio ES (within tolerance)")

        # Sort descending for reporting (largest contributors first)
        capital_breakdown = capital_breakdown.sort_values(ascending=False)

        logger.info(
            "Run completed: VaR10d=%.3f ES10d=%.3f VaR1y=%.3f ES1y=%.3f",
            var_10d,
            es_10d,
            var_1y,
            es_1y,
        )

        # --- Stress Testing ---
        stressed_var_1y = stressed_es_1y = None
        stress_shocks = self.config.get("stress_shocks")
        stress_enabled = self.config.get("stress_enabled")

        if stress_enabled:
            if not isinstance(stress_shocks, dict) or not stress_shocks:
                logger.warning(
                    "Stress testing enabled but no valid shocks defined; skipping."
                )
                stressed_var_1y = stressed_es_1y = 0.0
            else:
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
                if self.use_full_revaluation:
                    s_pnl, _ = self._pnl_from_shocks_full_revaluation(s_shocks)
                else:
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
            "used_full_revaluation": self.use_full_revaluation,
        }

    def _pnl_from_shocks_full_revaluation(
        self, shocks: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute P&L using full revaluation pricing.

        Parameters
        ----------
        shocks : np.ndarray, shape (n_scenarios, n_factors)
            Factor shocks (cumulative over horizon)

        Returns
        -------
        pnl_port : np.ndarray, shape (n_scenarios,)
            Portfolio P&L
        pnl_by_pos : dict
            Position-level P&L for each instrument
        """
        if self.pricing_portfolio is None:
            raise ValueError("pricing_portfolio required for full revaluation")

        # Get current market levels
        current_market = self._get_current_market_levels()

        # Apply shocks to get shocked market levels
        market_shocks = {}
        for i, factor in enumerate(self.factor_names):
            current_level = current_market[factor]
            shocked_levels = current_level * (1 + shocks[:, i])
            market_shocks[factor] = shocked_levels

        # Revalue portfolio
        pnl_port = self.pricing_portfolio.revalue_all(market_shocks)

        # For position-level breakdown, revalue each position individually
        pnl_by_pos = {}
        for pos in self.pricing_portfolio.positions:
            # Use underlying_factor or a position name attribute
            pos_name = getattr(pos, "underlying_factor", "UNKNOWN")

            # Create single-position portfolio
            if hasattr(pos, "underlying_factor"):
                factor = pos.underlying_factor
                if factor in market_shocks:
                    pos_pnl = pos.revalue(market_shocks[factor])
                    pnl_by_pos[pos_name] = pos_pnl

        return pnl_port, pnl_by_pos

    def compute_historical_var(
        self,
        lookback_days: int = 252,
        confidence: float = 0.999,
    ) -> dict:
        """
        Historical simulation VaR using actual market return history.

        Parameters
        ----------
        lookback_days : int
            Historical window (default 252 = 1 year)
        confidence : float
            Confidence level (default 99.9%)

        Returns
        -------
        dict with keys: 'var_10d', 'var_1y', 'es_10d', 'es_1y', 'method'
        """
        from econ_capital.market_risk.data_loaders import load_historical_returns

        # Load historical returns
        lookback_days = self.config.get("historical_lookback_days", 252)

        try:
            historical_returns = load_historical_returns(
                tickers=self.factor_names,
                start_date="2020-01-01",  # Or compute from lookback_days
            )

            # Use most recent window
            recent_returns = historical_returns.tail(lookback_days)

            # Bootstrap resampling
            n_sims = 10_000
            sampled_indices = self.rng.choice(
                len(recent_returns), size=n_sims, replace=True
            )
            shocks = recent_returns.iloc[sampled_indices].values

            # Cumulate over horizon
            horizon_shocks = shocks * np.sqrt(self.horizon_days)

            # Apply to portfolio
            if self.use_full_revaluation:
                pnl, _ = self._pnl_from_shocks_full_revaluation(horizon_shocks)
            else:
                pnl, _ = self._pnl_from_shocks(horizon_shocks)

            # Compute VaR/ES
            q = float(self.config["var_q"])
            var_10d = left_tail_var(pnl, q)
            es_10d = left_tail_es(pnl, q)

            # Scale to 1 year
            scale = np.sqrt(
                self.config.get("scaling_days_year", 252) / self.horizon_days
            )
            var_1y = var_10d * scale
            es_1y = es_10d * scale

            logger.info(
                "Historical VaR/ES: VaR1y=%.3f ES1y=%.3f (lookback=%d days)",
                var_1y,
                es_1y,
                lookback_days,
            )

            return {
                "historical_var_1y_999": float(var_1y),
                "historical_es_1y_999": float(es_1y),
                "historical_lookback_days": lookback_days,
            }

        except Exception as e:
            logger.error(f"Historical VaR computation failed: {e}")
            return {
                "historical_var_1y_999": None,
                "historical_es_1y_999": None,
                "historical_error": str(e),
            }

    def _compute_covar_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute CoVaR (Conditional VaR) for systemic risk measurement.

        Returns
        -------
        dict
            Keys = position names
            Values = dict with covar, delta_covar, systemic_contribution_pct
        """
        covar_results = {}

        # Get top 10 positions by absolute component ES
        if not hasattr(self, "pnl_by_pos"):
            logger.warning("pnl_by_pos not available for CoVaR computation")
            return {}

        # Sort positions by contribution
        position_contributions = {
            name: np.abs(pnl).mean() for name, pnl in self.pnl_by_pos.items()
        }
        top_positions = sorted(
            position_contributions.items(), key=lambda x: x[1], reverse=True
        )[:10]

        # Compute CoVaR for each top position
        for pos_name, _ in top_positions:
            pos_pnl = self.pnl_by_pos[pos_name]

            try:
                covar, delta_covar = compute_covar(
                    portfolio_losses=-self.pnl_port,  # Negative for loss convention
                    position_losses=-pos_pnl,
                    alpha=0.99,
                )

                # Systemic contribution as % of total VaR
                var_portfolio = left_tail_var(self.pnl_port, 0.99)
                systemic_pct = (
                    (delta_covar / var_portfolio * 100) if var_portfolio > 0 else 0
                )

                covar_results[pos_name] = {
                    "covar": float(covar),
                    "delta_covar": float(delta_covar),
                    "systemic_contribution_pct": float(systemic_pct),
                }

            except Exception as e:
                logger.warning(f"CoVaR computation failed for {pos_name}: {e}")
                continue

        logger.info(f"Computed CoVaR for {len(covar_results)} positions")
        return covar_results

    def _get_current_market_levels(self) -> Dict[str, float]:
        """
        Fetch current market prices for each factor.
        """

        current_levels = {}
        for factor in self.factor_names:
            try:
                ticker = yf.Ticker(factor)
                current_levels[factor] = ticker.history(period="1d")["Close"].iloc[-1]
            except Exception:
                logger.warning(f"Could not fetch price for {factor}, using default")
                current_levels[factor] = 100.0  # Fallback

        return current_levels
