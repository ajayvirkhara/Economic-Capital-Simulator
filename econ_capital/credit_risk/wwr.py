"""Wrong-Way Risk (WWR) helper adjustments."""

from __future__ import annotations
import numpy as np


def adjust_for_wwr(
    exposures: np.ndarray, credit_factors: np.ndarray, sensitivity: float = 0.5
) -> np.ndarray:
    """
    Apply a simple multiplicative WWR adjustment:
        exposures_adj = exposures * (1 + sensitivity * credit_factors)

    exposures shape: (n_paths, n_steps)
    credit_factors shape must broadcast to exposures (n_paths, n_steps) or (n_paths, 1)
    """
    exp = np.asarray(exposures, dtype=float)
    cf = np.asarray(credit_factors, dtype=float)
    if cf.shape not in (exp.shape, (exp.shape[0], 1)):
        raise ValueError(
            "credit_factors must be shape (n_paths,n_steps) or (n_paths,1)"
        )
    return exp * (1.0 + float(sensitivity) * cf)
