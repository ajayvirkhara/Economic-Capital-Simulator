"""Structural Wrong-Way Risk (WWR) modeling with joint simulation."""

from __future__ import annotations
import numpy as np
from scipy.stats import norm


def structural_wwr_adjustment(
    exposures: np.ndarray,
    default_thresholds: np.ndarray,
    systematic_factors: np.ndarray,
    asset_values: np.ndarray,
    correlation_expo_asset: float = -0.40,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Structural model for WWR based on Merton-style joint simulation.

    Models the joint distribution of:
    - Counterparty creditworthiness (asset value process)
    - Exposure levels (correlated with credit quality)

    Parameters
    ----------
    exposures : np.ndarray, shape (n_paths, n_counterparties)
        Base exposure levels
    default_thresholds : np.ndarray, shape (n_counterparties,)
        Default barriers for each counterparty (from PD)
    systematic_factors : np.ndarray, shape (n_paths,)
        Common systematic risk factor
    asset_values : np.ndarray, shape (n_paths, n_counterparties)
        Simulated counterparty asset values
    correlation_expo_asset : float
        Correlation between exposure and asset value (negative = WWR)

    Returns
    -------
    wwr_exposures : np.ndarray
        WWR-adjusted exposures
    conditional_pd : np.ndarray
        Path-dependent default probabilities
    """
    n_paths, n_cpty = exposures.shape
    rng = np.random.default_rng(seed)

    # Joint simulation: exposure shocks correlated with asset value shocks
    rho_ea = correlation_expo_asset

    # Idiosyncratic shocks
    z_expo = rng.standard_normal((n_paths, n_cpty))

    # Correlated exposure movements
    # When asset value drops (credit deteriorates), exposure increases (if rho < 0)
    asset_shocks = (asset_values - asset_values.mean(axis=0)) / (
        asset_values.std(axis=0) + 1e-8
    )

    correlated_expo_shocks = rho_ea * asset_shocks + np.sqrt(1 - rho_ea**2) * z_expo

    # Adjust exposures based on credit deterioration
    # Higher exposure when counterparty is closer to default
    distance_to_default = (asset_values - default_thresholds) / (
        asset_values.std(axis=0) + 1e-8
    )
    wwr_multiplier = 1.0 + 0.5 * np.maximum(-distance_to_default, 0) ** 2

    wwr_exposures = exposures * wwr_multiplier * (1 + 0.15 * correlated_expo_shocks)
    wwr_exposures = np.maximum(wwr_exposures, 0)

    # Compute conditional PD based on systematic factor
    # Using Vasicek single-factor model
    conditional_pd = norm.cdf(
        (norm.ppf(default_thresholds) - np.sqrt(0.25) * systematic_factors[:, None])
        / np.sqrt(1 - 0.25)
    )

    return wwr_exposures, conditional_pd


def simulate_asset_value_process(
    n_paths: int,
    n_counterparties: int,
    initial_value: float = 100.0,
    drift: float = 0.05,
    volatility: float = 0.30,
    correlation: float = 0.25,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate counterparty asset values using Merton model framework.

    Returns
    -------
    asset_values : np.ndarray, shape (n_paths, n_counterparties)
    """
    rng = np.random.default_rng(seed)

    # Systematic factor
    z_sys = rng.standard_normal(n_paths)[:, None]

    # Idiosyncratic factors
    z_idio = rng.standard_normal((n_paths, n_counterparties))

    # Combined shocks
    sqrt_rho = np.sqrt(correlation)
    sqrt_1mrho = np.sqrt(1 - correlation)

    shocks = sqrt_rho * z_sys + sqrt_1mrho * z_idio

    # Geometric Brownian motion (simplified, single period)
    asset_values = initial_value * np.exp(
        (drift - 0.5 * volatility**2) + volatility * shocks
    )

    return asset_values


def compute_default_threshold_from_pd(
    pd_annual: float, time_horizon: float = 1.0
) -> float:
    """
    Convert annual PD to default threshold in standard normal space (Merton model).

    In Merton model: PD = Φ(threshold), so threshold = Φ^(-1)(PD)
    """
    pd_clipped = np.clip(pd_annual * time_horizon, 0.0001, 0.9999)
    return norm.ppf(pd_clipped)


# Backward compatibility wrapper
def adjust_for_wwr(
    exposures: np.ndarray,
    credit_factors: np.ndarray,
    sensitivity: float = 0.5,
    apply_to_volatility: bool = False,
    min_factor: float = 0.0,
) -> np.ndarray:
    """
    DEPRECATED: Use structural_wwr_adjustment for new code.

    Simple heuristic WWR scaling (kept for backward compatibility).
    """
    exp = np.asarray(exposures, dtype=float)
    cf = np.asarray(credit_factors, dtype=float)

    adverse_cf = np.maximum(cf, 0.0)

    if apply_to_volatility:
        scale = 1.0 + sensitivity * 4.5 * adverse_cf
        return np.maximum(scale, min_factor)
    else:
        scale = 1.0 + sensitivity * 2.8 * adverse_cf
        return exp * np.maximum(scale, min_factor)
