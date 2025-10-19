"""
Credit Capital Engine (CCR)
===========================

Purpose
-------
Aggregates counterparty-level credit losses into portfolio-level
Expected Loss (EL), Unexpected Loss (UL), and Economic Capital (EC).

Integration
-----------
- Upstream: default_model (for EL, CVA, hazard, etc.)
- Downstream: aggregate capital reporting

Formulae
--------
EL_total = Σ_i EAD_i × PD_i × LGD_i
Var(L) = Σ_i Σ_j ρ_ij × σ_i × σ_j
EC = Φ⁻¹(α) × sqrt(Var(L))
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm

from econ_capital.utils import setup_logging, validate_shape, timed_section

logger = setup_logging(__name__)


# ----------------------------------------------------------------------
# Portfolio-level aggregation
# ----------------------------------------------------------------------
def aggregate_credit_losses(
    el: np.ndarray,
    ul: np.ndarray,
    corr: np.ndarray,
    confidence: float = 0.999,
) -> tuple[float, float, float]:
    """
    Aggregate counterparty-level EL and UL into total portfolio capital.

    Parameters
    ----------
    el : np.ndarray
        Expected losses per counterparty
    ul : np.ndarray
        Unexpected losses per counterparty (std deviation)
    corr : np.ndarray
        Correlation matrix across counterparties
    confidence : float
        Confidence level for Economic Capital (default=99.9%)

    Returns
    -------
    EL_total : float
    UL_total : float
    EC_total : float
    """

    el = np.asarray(el, dtype=float)
    ul = np.asarray(ul, dtype=float)
    corr = np.asarray(corr, dtype=float)

    validate_shape(corr, (len(el), len(el)), name="corr")

    with timed_section("aggregate_credit_losses"):
        var_portfolio = np.dot(ul, np.dot(corr, ul))
        ul_total = np.sqrt(var_portfolio)
        el_total = el.sum()
        z = norm.ppf(confidence)
        ec_total = el_total + z * ul_total

    logger.info(
        "Portfolio Credit Capital computed | EL=%.3f | UL=%.3f | EC=%.3f | z=%.3f",
        el_total,
        ul_total,
        ec_total,
        z,
    )
    return el_total, ul_total, ec_total


# ----------------------------------------------------------------------
# Counterparty-level helper (for demo)
# ----------------------------------------------------------------------
def compute_counterparty_risk_profiles(counterparties: list[dict]) -> pd.DataFrame:
    """
    Compute simple EL and UL per counterparty given their EAD, PD, and LGD.

    Parameters
    ----------
    counterparties : list of dicts
        Each dict = {"name": str, "EAD": float, "PD": float, "LGD": float}

    Returns
    -------
    pd.DataFrame
        Columns = [counterparty, EAD, PD, LGD, EL, UL]
    """
    df = pd.DataFrame(counterparties)
    df["EL"] = df["EAD"] * df["PD"] * df["LGD"]
    df["UL"] = df["EAD"] * np.sqrt(df["PD"] * (1 - df["PD"])) * df["LGD"]

    logger.debug("Computed counterparty EL/UL table:\n%s", df)
    return df
