"""
Trade and NettingSet data structures for the Credit Risk module.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from .csa import CSA


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
