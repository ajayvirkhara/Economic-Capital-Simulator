"""
Exposure model functions for Counterparty Credit Risk (CCR).

Contains helper methods for:
- Margin call frequency calculation
- Mark-to-market (MTM) computation
- Collateral path generation
"""

import numpy as np
from econ_capital.utils import setup_logging

from .csa import CSA

logger = setup_logging(__name__)


# -----------------------------------------------------------------------
# Private helpers
# -----------------------------------------------------------------------
def _calls_per_year(csa: CSA) -> int:
    """Determine VM call frequency per year based on CSA configuration."""
    if getattr(csa, "vm_mode", None) and getattr(csa, "vm_calls", None):
        mode = getattr(csa, "vm_mode")
        calls = max(1, int(getattr(csa, "vm_calls") or 1))
        if mode == "per_day":
            return int(getattr(csa, "business_days_per_year", 252)) * calls
        if mode == "per_week":
            return 52 * calls
        if mode == "per_year":
            return calls
        raise ValueError("CSA.vm_mode must be one of: per_day | per_week | per_year")

    if getattr(csa, "vm_calls_per_day", None):
        return int(getattr(csa, "business_days_per_year", 252)) * max(
            1, int(getattr(csa, "vm_calls_per_day"))
        )

    return 252  # fallback: daily calls


def _compute_mtm(trades, market_paths) -> np.ndarray:
    """
    Compute total MTM across all trades and market factors.

    Formula (stylised):
        MTM_t = Σ [ w * (ΔS/S0) + 0.5 * γ * (ΔS/S0)^2 + add ]
    """
    logger.debug("Computing MTM for %d trades", len(trades))
    first_shape = next(iter(market_paths.values())).shape
    n_paths, n_steps = first_shape
    mtm = np.zeros((n_paths, n_steps))

    for tr in trades:
        if tr.factor not in market_paths:
            raise ValueError(f"Trade {tr.name} refers to unknown factor {tr.factor}")

        S = market_paths[tr.factor]
        if S.shape != first_shape:
            raise ValueError(f"Market path for {tr.factor} has inconsistent shape")

        # Use relative price changes for better dynamics
        rel_dS = (S - S[:, [0]]) / S[:, [0]]

        # Stylised linear + convexity MTM model
        trade_mtm = tr.w * rel_dS + 0.5 * tr.gamma * (rel_dS**2) + tr.add

        # Optional: scaling to bring magnitudes into realistic ranges
        trade_mtm *= 100  # scale by notional if needed

        mtm += trade_mtm

    logger.debug(
        "MTM computed successfully with mean=%.4f, std=%.4f",
        float(np.mean(mtm)),
        float(np.std(mtm)),
    )
    return mtm


def _build_collateral_path(mtm: np.ndarray, times: np.ndarray, csa: CSA) -> np.ndarray:
    """
    Construct collateral path under CSA (VM + IM).

    Mechanics
    ----------
    - VM calls occur according to CSA schedule
    - Call = (MTM - Collateral - Threshold) if |Call| > MTA
    - IM added as static buffer across all paths
    """
    logger.debug("Building collateral path with VM/IM mechanics")

    _, n_steps = mtm.shape
    collat = np.zeros_like(mtm)
    calls_py = _calls_per_year(csa)

    total_calls = max(1, int(calls_py * times[-1]))
    call_steps = max(1, int(n_steps / total_calls))

    im = getattr(csa, "im", 0.0)
    threshold = getattr(csa, "threshold", 0.0)

    for t in range(n_steps):
        if t % call_steps == 0:
            collat[:, t] = np.clip(mtm[:, t] - threshold, a_min=0, a_max=None) + im
        else:
            collat[:, t] = collat[:, t - 1]

    logger.debug("Collateral path completed: shape=%s", str(collat.shape))
    return collat
