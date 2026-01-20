"""
Full revaluation pricing for market risk positions.

Supports:
- Equities (spot)
- Fixed Income (DV01 + convexity)
- FX forwards
- European options (Black-Scholes)
- Interest rate swaps
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Dict

# ============================================================================
# EQUITIES
# ============================================================================


@dataclass
class EquityPosition:
    """Simple equity position with full revaluation."""

    quantity: float  # Number of shares
    current_price: float  # Current spot price
    underlying_factor: str  # e.g., 'SPX', 'AAPL'

    def revalue(self, shocked_prices: np.ndarray) -> np.ndarray:
        """
        Linear revaluation: P&L = quantity × (S_shocked - S_0)

        Parameters
        ----------
        shocked_prices : np.ndarray, shape (n_scenarios,)
            Shocked spot prices

        Returns
        -------
        pnl : np.ndarray, shape (n_scenarios,)
        """
        return self.quantity * (shocked_prices - self.current_price)


# ============================================================================
# FIXED INCOME
# ============================================================================


@dataclass
class BondPosition:
    """
    Fixed income position with DV01 and convexity.

    Uses duration-convexity approximation:
    ΔP/P ≈ -D·Δy + 0.5·C·(Δy)²
    """

    notional: float  # Face value
    current_price: float  # Clean price (e.g., 98.5)
    modified_duration: float  # Years (e.g., 5.2)
    convexity: float  # Years² (e.g., 35.0)
    yield_factor: str  # e.g., 'USD10Y'
    current_yield: float  # Current yield in decimal (e.g., 0.03 = 3%)

    def revalue(self, shocked_yields: np.ndarray) -> np.ndarray:
        """
        Duration-convexity approximation.

        Parameters
        ----------
        shocked_yields : np.ndarray, shape (n_scenarios,)
            Shocked yield levels (decimal, e.g., 0.035 = 3.5%)

        Returns
        -------
        pnl : np.ndarray
        """
        dy = shocked_yields - self.current_yield  # Yield change

        # Price change as % of notional
        pct_change = -self.modified_duration * dy + 0.5 * self.convexity * (dy**2)

        # P&L in currency units
        current_value = self.notional * (self.current_price / 100)
        return current_value * pct_change


# ============================================================================
# FX FORWARDS
# ============================================================================


@dataclass
class FXForward:
    """
    FX Forward contract.

    MTM = Notional × (F_market - F_strike) × DF
    """

    notional: float  # Notional in base currency
    strike: float  # Contracted forward rate (e.g., 1.25 USD/EUR)
    maturity: float  # Years to maturity
    fx_spot_factor: str  # e.g., 'EURUSD'
    domestic_rate: float  # Risk-free rate in quote currency
    foreign_rate: float  # Risk-free rate in base currency
    current_spot: float  # Current FX spot rate

    def revalue(self, shocked_spots: np.ndarray) -> np.ndarray:
        """
        Revalue forward contract under shocked FX spot.

        F_market = S × exp((r_d - r_f) × T)
        MTM = Notional × (F_market - F_strike) × exp(-r_d × T)
        """
        # Shocked forward rates
        F_market = shocked_spots * np.exp(
            (self.domestic_rate - self.foreign_rate) * self.maturity
        )

        # Discount factor
        DF = np.exp(-self.domestic_rate * self.maturity)

        # P&L
        return self.notional * (F_market - self.strike) * DF


# ============================================================================
# EUROPEAN OPTIONS
# ============================================================================


@dataclass
class EuropeanOption:
    """European call/put with Black-Scholes pricing."""

    strike: float
    maturity: float  # Years
    option_type: str  # 'call' or 'put'
    volatility: float  # Implied vol (decimal)
    quantity: float  # Number of contracts
    underlying_factor: str
    risk_free_rate: float = 0.02
    current_spot: float = None  # Will be set during initialization

    def revalue(
        self, shocked_spots: np.ndarray, time_to_mat: float = None
    ) -> np.ndarray:
        """
        Full Black-Scholes revaluation.

        Parameters
        ----------
        shocked_spots : np.ndarray
            Shocked underlying spot prices
        time_to_mat : float, optional
            Remaining time to maturity (if None, uses self.maturity)

        Returns
        -------
        pnl : np.ndarray
            P&L from current spot to shocked spots
        """
        if time_to_mat is None:
            time_to_mat = self.maturity

        if self.current_spot is None:
            raise ValueError("current_spot must be set before revaluation")

        # Current price
        current_pnl = self._bs_price(self.current_spot, time_to_mat)

        # Shocked prices (vectorized)
        shocked_pnl = self._bs_price(shocked_spots, time_to_mat)

        return (shocked_pnl - current_pnl) * self.quantity

    def _bs_price(self, S: np.ndarray, T: float) -> np.ndarray:
        """Black-Scholes closed-form price (vectorized)."""
        K, sigma, r = self.strike, self.volatility, self.risk_free_rate

        # Handle zero time to maturity
        if T <= 1e-6:
            if self.option_type == "call":
                return np.maximum(S - K, 0)
            else:
                return np.maximum(K - S, 0)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if self.option_type == "call":
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ============================================================================
# INTEREST RATE SWAPS
# ============================================================================


@dataclass
class InterestRateSwap:
    """
    Plain vanilla interest rate swap.

    Simplified DV01-based pricing for small rate changes.
    For large shocks, should use full curve pricing.
    """

    notional: float
    fixed_rate: float  # Contracted fixed rate (decimal)
    tenor_years: float  # Swap maturity
    payment_freq: int = 2  # Semi-annual = 2
    rate_factor: str = "USD10Y"  # Reference rate factor
    current_rate: float = None  # Current market rate

    def revalue(self, shocked_rates: np.ndarray) -> np.ndarray:
        """
        DV01-based swap revaluation.

        MTM ≈ DV01 × Δrate × 10000
        """
        if self.current_rate is None:
            raise ValueError("current_rate must be set")

        # DV01 (dollar value of 01 basis point)
        dv01 = self.notional * self.tenor_years / 10000

        # Rate difference in basis points
        rate_diff_bps = (shocked_rates - self.current_rate) * 10000

        return dv01 * rate_diff_bps


# ============================================================================
# PORTFOLIO CONTAINER
# ============================================================================


class Portfolio:
    """Container for mixed position types with unified revaluation."""

    def __init__(self):
        self.positions = []

    def add_position(self, position):
        """Add any position type (Equity, Bond, Option, etc.)"""
        self.positions.append(position)

    def revalue_all(self, market_shocks: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Revalue entire portfolio under market shocks.

        Parameters
        ----------
        market_shocks : dict
            Keys = factor names (e.g., 'SPX', 'USD10Y')
            Values = shocked levels, shape (n_scenarios,)

        Returns
        -------
        total_pnl : np.ndarray, shape (n_scenarios,)
            Aggregate portfolio P&L across all positions
        """
        n_scenarios = next(iter(market_shocks.values())).shape[0]
        total_pnl = np.zeros(n_scenarios)

        for pos in self.positions:
            if isinstance(pos, EquityPosition):
                shocked = market_shocks.get(pos.underlying_factor)
                if shocked is not None:
                    total_pnl += pos.revalue(shocked)

            elif isinstance(pos, BondPosition):
                shocked = market_shocks.get(pos.yield_factor)
                if shocked is not None:
                    total_pnl += pos.revalue(shocked)

            elif isinstance(pos, FXForward):
                shocked = market_shocks.get(pos.fx_spot_factor)
                if shocked is not None:
                    total_pnl += pos.revalue(shocked)

            elif isinstance(pos, EuropeanOption):
                shocked = market_shocks.get(pos.underlying_factor)
                if shocked is not None:
                    total_pnl += pos.revalue(shocked)

            elif isinstance(pos, InterestRateSwap):
                shocked = market_shocks.get(pos.rate_factor)
                if shocked is not None:
                    total_pnl += pos.revalue(shocked)

        return total_pnl
