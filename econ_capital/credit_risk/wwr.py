"""Wrong-Way Risk (WWR) helper adjustments."""

from __future__ import annotations
import numpy as np


def adjust_for_wwr(
    exposures: np.ndarray,
    credit_factors: np.ndarray,
    sensitivity: float = 0.5,
    apply_to_volatility: bool = False,
    min_factor: float = 0.0,
) -> np.ndarray:
    """
    Enhanced WWR adjustment – can be applied to mean (EL) or volatility (UL).

    Parameters
    ----------
    exposures : np.ndarray
        Base exposures (EL or UL values)
    credit_factors : np.ndarray
        Credit factor shocks (n_paths, n_counterparties) or averaged
    sensitivity : float
        Strength of WWR effect (0.0 = no effect, 0.3–0.7 = realistic range)
    apply_to_volatility : bool
        If True: returns scaling factor for UL (tail risk)
        If False: directly scales EL (mean)
    min_factor : float
        Minimum scaling factor (prevents unrealistically low values)

    Returns
    -------
    np.ndarray
        Adjusted exposures or scaling factors
    """
    exp = np.asarray(exposures, dtype=float)
    cf = np.asarray(credit_factors, dtype=float)

    # Use only adverse (positive) shocks
    adverse_cf = np.maximum(cf, 0.0)

    if apply_to_volatility:
        # Stronger effect on tail / volatility – more realistic for capital
        scale = 1.0 + sensitivity * 4.5 * adverse_cf
        return np.maximum(scale, min_factor)
    else:
        # Moderate effect on expected loss
        scale = 1.0 + sensitivity * 2.8 * adverse_cf
        return exp * np.maximum(scale, min_factor)
