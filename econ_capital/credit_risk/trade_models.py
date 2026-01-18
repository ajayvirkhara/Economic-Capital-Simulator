"""
Trade and NettingSet data structures for the Credit Risk module.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from .csa import CSA
from typing import Protocol
import numpy as np


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------
@dataclass
class Trade:
    """
    Stylised trade exposure model relative to a specific market factor.

    MTM_tr(t) = w * (S_t - S_0) + 0.5 * gamma * (S_t - S_0)^2 + add

    Parameters
    ----------
    name : str
        Trade identifier.
    factor : str
        Risk factor name (e.g., 'SP500', 'USD10Y', 'EURUSD').
    w : float
        Linear exposure (delta, DV01, etc.).
    gamma : float
        Convexity term (gamma).
    add : float
        Constant offset, e.g., fixed leg or upfront payment.
    """

    name: str
    factor: str
    w: float
    gamma: float = 0.0
    add: float = 0.0


@dataclass
class NettingSet:
    """Grouping of trades under a single CSA agreement."""

    counterparty: str
    trades: list[Trade]
    csa: CSA = field(default_factory=CSA)


class PricingModel(Protocol):
    """Protocol for trade pricing models."""

    def price(self, market_data: dict[str, np.ndarray], time_idx: int) -> np.ndarray:
        """Price the trade given market data at specific time index."""
        ...


@dataclass
class VanillaSwap(Trade):
    """Interest rate swap with proper discounting."""

    notional: float = 1_000_000.0
    fixed_rate: float = 0.03
    tenor_years: float = 5.0
    payment_freq: int = 2  # semi-annual

    def price(self, market_data: dict[str, np.ndarray], time_idx: int) -> np.ndarray:
        """
        Price using forward curve and discount factors.

        Simplified: MTM = Notional × DV01 × (Market_Rate - Fixed_Rate) × Tenor
        """
        if self.factor not in market_data:
            raise ValueError(f"Missing factor {self.factor}")

        rates = market_data[self.factor][:, time_idx]

        # Simplified DV01-based pricing
        dv01 = self.notional * self.tenor_years / 10000  # per bp
        rate_diff_bps = (rates - self.fixed_rate) * 10000

        mtm = dv01 * rate_diff_bps
        return mtm


@dataclass
class EuropeanOption(Trade):
    """European call/put option with Black-Scholes pricing."""

    strike: float = 100.0
    maturity: float = 1.0
    option_type: str = "call"  # "call" or "put"
    volatility: float = 0.25
    risk_free_rate: float = 0.02

    def price(self, market_data: dict[str, np.ndarray], time_idx: int) -> np.ndarray:
        """Black-Scholes pricing."""
        from scipy.stats import norm

        S = market_data[self.factor][:, time_idx]
        K = self.strike
        T = max(self.maturity - (time_idx * 0.1), 0.01)  # Time to maturity
        sigma = self.volatility
        r = self.risk_free_rate

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if self.option_type == "call":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return price * self.w  # Scale by notional weight
