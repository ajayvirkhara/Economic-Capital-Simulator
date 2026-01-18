"""
Default & Credit Capital Engine
===============================

Purpose
-------
Converts exposure profiles into credit loss metrics:
    - Exposure at Default (EAD)
    - Expected Loss (EL)
    - Credit Value Adjustment (CVA)
    - Default Probability (PD) term structure

This module extends the exposure engine by layering on
counterparty default and recovery assumptions.

Integration
-----------
- Upstream: econ_capital.credit_risk.exposure_engine
- Downstream: econ_capital.aggregate / allocation modules
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pds
import warnings
import os

from econ_capital.utils import setup_logging, profile_test

logger = setup_logging(__name__)


# --------------------------------------------------------------------
# 1. Credit Input Container
# --------------------------------------------------------------------
@dataclass
class CreditInputs:
    """
    Credit parameters for a given counterparty or netting set.

    Attributes
    ----------
    counterparty : str
        Identifier for the counterparty.
    lgd : float, optional
        Loss Given Default (fractional, e.g. 0.6 = 60%).
    recovery : float, optional
        Recovery rate (1 - LGD). Only one of LGD or recovery is required.
    pd_annual : float
        Annual probability of default (required).
    """

    counterparty: str
    pd_annual: float
    lgd: Optional[float] = None
    recovery: Optional[float] = None

    def __post_init__(self):
        """Clip and validate inputs at initialization."""
        if self.lgd is not None:
            self.lgd = float(np.clip(self.lgd, 0.0, 1.0))
        if self.recovery is not None:
            self.recovery = float(np.clip(self.recovery, 0.0, 1.0))

        # Warn if both provided and inconsistent
        if self.lgd is not None and self.recovery is not None:
            derived_from_recovery = 1.0 - self.recovery
            if not np.isclose(self.lgd, derived_from_recovery, atol=1e-6):
                if "PYTEST_CURRENT_TEST" not in os.environ:
                    warnings.warn(
                        f"Inconsistent LGD ({self.lgd}) and recovery ({self.recovery}); "
                        "using LGD directly."
                    )

    def effective_lgd(self) -> float:
        """Return LGD, deriving from recovery if needed."""
        if self.lgd is not None:
            return self.lgd
        if self.recovery is not None:
            return 1.0 - self.recovery
        return 0.6  # default LGD 60%

    def get_hazard_curve(self, times: np.ndarray) -> np.ndarray:
        """Convert annual PD to flat hazard rate."""
        pd = np.clip(self.pd_annual, 0.0, 0.999999)
        lam = -np.log(1.0 - pd)
        return np.full_like(times, lam, dtype=float)


def simulate_stochastic_lgd(
    base_lgd: float,
    n_paths: int,
    lgd_volatility: float = 0.15,
    min_lgd: float = 0.10,
    max_lgd: float = 0.90,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate stochastic LGD paths using Beta distribution.

    Parameters
    ----------
    base_lgd : float
        Mean/expected LGD
    n_paths : int
        Number of simulation paths
    lgd_volatility : float
        Volatility of LGD (controls Beta distribution spread)

    Returns
    -------
    lgd_paths : np.ndarray, shape (n_paths,)
        Simulated LGD values
    """
    rng = np.random.default_rng(seed)

    # Parametrize Beta distribution to match mean and volatility
    mean = np.clip(base_lgd, 0.01, 0.99)
    var = (lgd_volatility * mean) ** 2

    # Beta parameters
    alpha = mean * (mean * (1 - mean) / var - 1)
    beta = (1 - mean) * (mean * (1 - mean) / var - 1)

    # Ensure valid parameters
    alpha = max(alpha, 0.5)
    beta = max(beta, 0.5)

    lgd_paths = rng.beta(alpha, beta, size=n_paths)
    return np.clip(lgd_paths, min_lgd, max_lgd)


def simulate_stochastic_pd(
    base_pd: float,
    n_paths: int,
    credit_cycle_factor: np.ndarray | None = None,
    pd_volatility: float = 0.30,
    shock_scale: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate stochastic PD paths with credit cycle sensitivity.

    Parameters
    ----------
    base_pd : float
        Base annual PD
    n_paths : int
        Number of paths
    credit_cycle_factor : np.ndarray, optional
        Systematic credit factor (n_paths,), e.g., from simulate_credit_factors
    pd_volatility : float
        Idiosyncratic PD volatility

    Returns
    -------
    pd_paths : np.ndarray, shape (n_paths,)
    """
    rng = np.random.default_rng(seed)

    # Log-normal model for PD
    log_pd_mean = np.log(base_pd + 1e-6)

    if credit_cycle_factor is not None:
        # Systematic component from credit cycle
        systematic_shock = 1.0 + shock_scale * pd_volatility * credit_cycle_factor
    else:
        systematic_shock = 1.0

    # Idiosyncratic component
    idio_shock = pd_volatility * rng.standard_normal(n_paths)

    # Combined
    log_pd = log_pd_mean + systematic_shock + idio_shock
    pd_paths = np.exp(log_pd)

    return np.clip(pd_paths, 0.0001, 0.50)


# --------------------------------------------------------------------
# 2. Default Probability Computation
# --------------------------------------------------------------------
def incremental_default_prob(times: np.ndarray, hazard: np.ndarray) -> np.ndarray:
    """
    Compute bucket-level default probabilities ΔPD_k from hazard rates λ_k.

    Vectorized implementation:
        S(t) = exp(-∫ λ dt)
        ΔPD = S(t_{k-1}) - S(t_k)
    """
    times = np.asarray(times, dtype=float)
    hazard = np.asarray(hazard, dtype=float)

    if hazard.shape != times.shape:
        raise ValueError("hazard and times must have same shape")

    dt = np.diff(np.concatenate([[0.0], times]))
    cum_hazard = np.cumsum(hazard * dt)
    survival = np.exp(-cum_hazard)
    dpd = np.diff(np.concatenate([[0.0], 1 - survival]))
    return dpd


# --------------------------------------------------------------------
# 3. Exposure at Default (EAD)
# --------------------------------------------------------------------
def ead_from_exposure(
    exposures: np.ndarray,
    method: str = "EE",
    quantile: float = 0.975,
) -> np.ndarray:
    """
    Derive an Exposure-at-Default (EAD) profile from pathwise exposures.

    Parameters
    ----------
    exposures : np.ndarray
        Pathwise positive exposures (n_paths, n_steps)
    method : str
        'EE' for mean exposure (default), or 'quantile' for tail exposure
    quantile : float
        Quantile level if method == 'quantile'
    """
    if exposures.ndim != 2:
        raise ValueError("Exposure array must be 2D (n_paths, n_steps)")

    if method == "EE":
        return exposures.mean(axis=0)

    if method == "quantile":
        if not 0 < quantile < 1:
            raise ValueError("Quantile must be between 0 and 1")
        return np.quantile(exposures, quantile, axis=0)

    raise ValueError("Invalid method: choose 'EE' or 'quantile'")


# --------------------------------------------------------------------
# 4. Core Loss Engine
# --------------------------------------------------------------------
def _loss_profile(
    times: np.ndarray,
    ead: np.ndarray,
    credit: CreditInputs,
    discount: Optional[np.ndarray] = None,
    discounted: bool = True,
    lgd_paths: Optional[np.ndarray] = None,
    pd_paths: Optional[np.ndarray] = None,
) -> Tuple[float, pds.DataFrame]:
    """Internal engine for CVA / EL computation with stochastic LGD/PD support."""
    times = np.asarray(times, dtype=float)
    ead = np.asarray(ead, dtype=float)

    # Use stochastic LGD if provided
    if lgd_paths is not None:
        lgd = lgd_paths.mean()  # Use mean for expected loss
    else:
        lgd = credit.effective_lgd()

    # Use stochastic PD if provided
    if pd_paths is not None:
        # Average across paths for expected default probability
        avg_pd = pd_paths.mean()
        hz = -np.log(1.0 - np.clip(avg_pd, 0.0, 0.999999))
        hz = np.full_like(times, hz)
    else:
        hz = credit.get_hazard_curve(times)

    dpd = incremental_default_prob(times, hz)

    if ead.shape != times.shape:
        raise ValueError("EAD and time grid must have the same length")

    # discount curve or flat rate
    if discount is None:
        r = 0.03 if discounted else 0.0
        DF = np.exp(-r * times)
    else:
        DF = np.asarray(discount, dtype=float)
        if DF.shape != times.shape:
            raise ValueError("Discount curve shape mismatch")

    bucket = DF * ead * lgd * dpd
    cumulative = np.cumsum(bucket)

    profile = pds.DataFrame(
        {
            "time": times,
            "EAD": ead,
            "LGD": lgd,
            "hazard": hz,
            "dPD": dpd,
            "DF": DF,
            "Loss_bucket": bucket,
            "Loss_cum": cumulative,
        }
    )
    total = float(bucket.sum())
    return total, profile


# --------------------------------------------------------------------
# 5. CVA and Expected Loss Wrappers
# --------------------------------------------------------------------
@profile_test
def compute_cva(
    times: np.ndarray,
    ead: np.ndarray,
    credit: CreditInputs,
    discount: Optional[np.ndarray] = None,
) -> Tuple[float, pds.DataFrame]:
    """Compute discounted Credit Valuation Adjustment (CVA)."""
    total, profile = _loss_profile(times, ead, credit, discount, discounted=True)
    logger.info(
        "CVA computed for %s | Total: %.6f | Mean hazard: %.6f | LGD: %.2f",
        credit.counterparty,
        total,
        profile["hazard"].mean(),
        credit.effective_lgd(),
    )
    return total, profile


@profile_test
def compute_expected_loss(
    times: np.ndarray,
    ead: np.ndarray,
    credit: CreditInputs,
) -> Tuple[float, pds.DataFrame]:
    """Compute undiscounted Expected Loss (EL)."""
    total, profile = _loss_profile(times, ead, credit, discounted=False)
    logger.info(
        "Expected Loss computed for %s | Total: %.6f | LGD: %.2f",
        credit.counterparty,
        total,
        credit.effective_lgd(),
    )
    return total, profile


# --------------------------------------------------------------------
# 6. Flat Hazard Utility (optional standalone use)
# --------------------------------------------------------------------
@profile_test
def compute_flat_hazard(times: np.ndarray, flat_annual_pd: float) -> np.ndarray:
    """Compute a flat hazard rate from an annual PD."""
    pd = np.clip(flat_annual_pd, 0.0, 0.999999)
    lam = -np.log(1.0 - pd)
    return np.full_like(times, lam, dtype=float)
