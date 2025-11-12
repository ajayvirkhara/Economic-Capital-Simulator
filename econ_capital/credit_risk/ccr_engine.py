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
from econ_capital.credit_risk.config import DEFAULT_CONFIG
from econ_capital.credit_risk.market_model import simulate_credit_factors
from econ_capital.credit_risk.wwr import adjust_for_wwr

logger = setup_logging(__name__)


# ----------------------------------------------------------------------
# Portfolio-level aggregation
# ----------------------------------------------------------------------
def aggregate_credit_losses(
    el: np.ndarray,
    ul: np.ndarray,
    corr: np.ndarray,
    confidence: float | None = None,
) -> tuple[float, float, float, np.ndarray]:
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
    confidence : float, optional
        Confidence level for Economic Capital (default=99.9%)

    Returns
    -------
    EL_total : float
    UL_total : float
    EC_total : float
    alloc : np.ndarray
        Economic capital allocation per counterparty
    """

    params = DEFAULT_CONFIG.copy()
    confidence = confidence or params.get("confidence", 0.999)

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
        alloc = ec_total * (el / el.sum())

    logger.info(
        "Portfolio Credit Capital computed | EL=%.3f | UL=%.3f | EC=%.3f | z=%.3f",
        el_total,
        ul_total,
        ec_total,
        z,
    )
    logger.info("Allocated EC per counterparty: %s", alloc)
    return el_total, ul_total, ec_total, alloc


# ----------------------------------------------------------------------
# Counterparty-level helper (for demo)
# ----------------------------------------------------------------------
def compute_counterparty_risk_profiles(counterparties: list[dict]) -> pd.DataFrame:
    """
    Compute EL and UL per counterparty, simulate correlated factors,
    and apply WWR adjustment.

    Parameters
    ----------
    counterparties : list of dicts
        Each dict = {"name": str, "EAD": float, "PD": float, "LGD": float}

    Returns
    -------
    pd.DataFrame
        Columns = [name, EAD, PD, LGD, EL, UL, EL_adj]
    """

    params = DEFAULT_CONFIG.copy()
    df = pd.DataFrame(counterparties)

    # Base expected & unexpected losses
    df["EL"] = df["EAD"] * df["PD"] * df["LGD"]
    df["UL"] = df["EAD"] * np.sqrt(df["PD"] * (1 - df["PD"])) * df["LGD"]

    # Simulate correlated credit factor shocks
    factors = simulate_credit_factors(
        n_paths=params["n_paths"],
        n_steps=len(df),
        corr=params["corr"],
        seed=params["seed"],
    )

    # Map factors into loss shocks (factor ↑ → higher loss)
    base_losses = df["EL"].values[None, :]
    shocked_losses = base_losses * (
        1 + 0.1 * factors
    )  # 10% sensitivity to factor movement
    simulated_mean_losses = shocked_losses.mean(axis=0)

    # Reduce factors to a per-counterparty metric (mean over paths)
    factor_means = factors.mean(axis=0).reshape(1, -1)

    # Use average factor correlation for WWR adjustment
    wwr_corr = params.get("wwr_corr", 0.2)

    df["EL_adj"] = adjust_for_wwr(
        simulated_mean_losses.reshape(1, -1),
        credit_factors=factor_means,
        sensitivity=wwr_corr,
    ).ravel()

    # Representative simulated loss metric (mean across paths)
    df["Simulated_Loss"] = simulated_mean_losses

    logger.debug("Computed counterparty EL/UL/EL_adj table:\n%s", df)
    return df
