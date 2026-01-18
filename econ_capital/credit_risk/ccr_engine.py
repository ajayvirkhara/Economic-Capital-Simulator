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
from scipy.stats import t

from econ_capital.credit_risk.allocation import allocate_ec
from econ_capital.utils import setup_logging, validate_shape, timed_section
from econ_capital.credit_risk.config import DEFAULT_CONFIG
from econ_capital.credit_risk.market_model import simulate_credit_factors
from econ_capital.credit_risk.wwr import (
    simulate_asset_value_process,
    compute_default_threshold_from_pd,
    structural_wwr_adjustment,
    adjust_for_wwr,
)
from econ_capital.config_loader import merge_with_global
from econ_capital.credit_risk.default_model import (
    simulate_stochastic_lgd,
    simulate_stochastic_pd,
)

logger = setup_logging(__name__)


# ----------------------------------------------------------------------
# Portfolio-level aggregation
# ----------------------------------------------------------------------
def aggregate_credit_losses(
    el: np.ndarray,
    ul: np.ndarray,
    corr: np.ndarray,
    confidence: float | None = None,
    config: dict | None = None,
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

    # Start with module defaults
    params = DEFAULT_CONFIG.copy()

    # If caller passed config, merge it
    if config:
        params.update(config)

    # Merge with global defaults (seed, default_n_paths, etc.)
    params = merge_with_global(params)

    confidence = confidence or params.get("confidence_level", 0.999)

    # Extract market model parameters
    credit_config = params.get("credit_risk", {})
    market_config = credit_config.get("market_model", {})

    el = np.asarray(el, dtype=float)
    ul = np.asarray(ul, dtype=float)
    corr = np.asarray(corr, dtype=float)

    validate_shape(corr, (len(el), len(el)), name="corr")

    with timed_section("aggregate_credit_losses"):
        var_portfolio = np.dot(ul, np.dot(corr, ul))
        ul_total = np.sqrt(var_portfolio)
        el_total = el.sum()
        z = t.ppf(confidence, 3)
        ec_total = el_total + z * ul_total
        alloc = ec_total * (el / el.sum())

    logger.info(
        "Portfolio Credit Capital | EL=%.0f | UL=%.0f | EC=%.0f | Conf=%.3f%%",
        el_total,
        ul_total,
        ec_total,
        confidence * 100,
    )
    logger.info("Allocated EC per counterparty: %s", alloc)

    # Simulate loss paths for Euler allocation (n_sims x n_counterparties)
    sim_cfg = params.get("simulation", {})
    n_sims = sim_cfg.get("default_n_paths", 10_000)
    seed = params.get("seed", 42)

    factors = simulate_credit_factors(
        n_paths=n_sims,
        n_steps=len(el),
        corr=params.get("credit_factor_correlation", 0.2),
        vol=market_config.get("vol", 0.20),
        seed=seed,
    )

    # Simple loss model: L_i = EL_i * (1 + sensitivity * factor_i)
    sensitivity = params.get("sensitivity", 1.0)
    simulated_losses = el[None, :] * (1 + sensitivity * factors)  # shape

    # Clip negative losses
    simulated_losses = np.maximum(simulated_losses, 0)

    # Total portfolio losses per sim
    port_losses = simulated_losses.sum(axis=1)

    # Compute VaR/EC on portfolio
    port_ec = np.quantile(port_losses, confidence)

    # Euler allocation
    alloc = allocate_ec(port_ec, simulated_losses)

    return el_total, ul_total, ec_total, alloc


# ----------------------------------------------------------------------
# Counterparty-level helper (for demo)
# ----------------------------------------------------------------------
def compute_counterparty_risk_profiles(
    counterparties: list[dict],
    config: dict | None = None,
    use_structural_wwr: bool = True,
) -> pd.DataFrame:
    """
    Compute EL and UL per counterparty, simulate correlated factors,
    and apply optional WWR adjustment.

    Parameters
    ----------
    counterparties : list of dicts
        Each dict = {"name": str, "EAD": float, "PD": float, "LGD": float}

    Returns
    -------
    pd.DataFrame
        Columns = [name, EAD, PD, LGD, EL, UL, EL_adj]
    """

    # ========== DEBUG BLOCK ==========
    print("\n" + "=" * 60)
    print("DEBUG: compute_counterparty_risk_profiles() called")
    print(f"Config passed in: {config is not None}")
    print(f"use_structural_wwr parameter: {use_structural_wwr}")
    # ========== DEBUG BLOCK ==========

    params = DEFAULT_CONFIG.copy()
    if config:
        params.update(config)
    params = merge_with_global(params)  # ← Merge global defaults

    # Extract credit-specific config flags
    print(f"\nMerged params keys: {list(params.keys())}")
    print(f"'credit_risk' in params: {'credit_risk' in params}")
    credit_config = params.get("credit_risk", {})
    print(f"credit_config: {credit_config}")

    # Stochastic parameter flags
    stochastic_config = credit_config.get("stochastic_params", {})
    print(f"\nStochastic config: {stochastic_config}")
    use_stochastic_lgd = stochastic_config.get("use_stochastic_lgd", True)
    use_stochastic_pd = stochastic_config.get("use_stochastic_pd", True)
    print(f"use_stochastic_lgd: {use_stochastic_lgd}")
    print(f"use_stochastic_pd: {use_stochastic_pd}")

    lgd_volatility = stochastic_config.get("lgd_volatility", 0.20)
    pd_volatility = stochastic_config.get("pd_volatility", 0.35)

    # WWR flags
    wwr_config = credit_config.get("wwr", {})
    use_structural_wwr = wwr_config.get(
        "use_structural_model", use_structural_wwr
    )  # Use config or parameter
    print(f"use_structural_wwr (from config): {use_structural_wwr}")
    print("=" * 60 + "\n")

    correlation_expo_asset = wwr_config.get("correlation_expo_asset", -0.40)
    asset_volatility = wwr_config.get("asset_volatility", 0.35)

    # Extract market model parameters
    market_config = credit_config.get("market_model", {})

    df = pd.DataFrame(counterparties)

    # Define EAD column name
    EAD_col = "EAD" if "EAD" in df.columns else "EAD_final"

    sim_cfg = params.get("simulation", {})
    n_paths = sim_cfg.get("default_n_paths", 5000)
    seed = params.get("seed", 42)

    # Simulate correlated credit factor shocks
    factors = simulate_credit_factors(
        n_paths=n_paths,
        n_steps=len(df),
        corr=params.get("corr", 0.2),
        vol=market_config.get("vol", 0.20),
        seed=seed,
    )

    # Stochastic LGD and PD
    EL_stochastic = []
    UL_stochastic = []
    rng = np.random.default_rng(seed)

    for idx, row in df.iterrows():
        lgd_floor = stochastic_config.get("lgd_floor", 0.10)
        lgd_ceiling = stochastic_config.get("lgd_ceiling", 0.75)
        base_lgd = rng.uniform(lgd_floor, lgd_ceiling)

        # Generate stochastic LGD
        if use_stochastic_lgd:
            lgd_paths = simulate_stochastic_lgd(
                base_lgd=base_lgd,
                n_paths=n_paths,
                lgd_volatility=lgd_volatility,
                seed=seed + idx,
            )
        else:
            lgd_paths = np.full(n_paths, base_lgd)  # Fixed LGD

        base_pd = row.get("PD", 0.01)
        pd_min = params.get("pd_min", 1e-6)
        pd_max = params.get("pd_max", 0.30)
        base_pd = np.clip(base_pd, pd_min, pd_max)
        ead = row.get("EAD", row.get("EAD_final", 0))

        # Generate stochastic PD with credit cycle
        if use_stochastic_pd:
            pd_paths = simulate_stochastic_pd(
                base_pd=base_pd,
                n_paths=n_paths,
                credit_cycle_factor=factors[:, idx] if idx < factors.shape[1] else None,
                pd_volatility=pd_volatility,
                shock_scale=market_config.get("shock_scale", 0.10),
                seed=seed + idx + 1000,
            )
        else:
            pd_paths = np.full(n_paths, base_pd)  # Fixed PD

        # Compute path-wise losses
        losses = ead * lgd_paths * pd_paths

        EL_stochastic.append(losses.mean())
        UL_stochastic.append(losses.std())

    df["EL"] = EL_stochastic
    df["UL"] = UL_stochastic

    # Reduce factors to a per-counterparty metric (mean over paths)
    factor_means = factors.mean(axis=0).reshape(1, -1)

    # Use average factor correlation for WWR adjustment
    wwr_corr = params.get("wwr_corr", 0.2)

    if use_structural_wwr:
        # Simulate asset values for all counterparties
        asset_values = simulate_asset_value_process(
            n_paths=n_paths,
            n_counterparties=len(df),
            volatility=asset_volatility,
            correlation=params.get("asset_correlation", 0.25),
            seed=seed + 999,
        )

        # Compute default thresholds from PDs
        default_thresholds = np.array(
            [compute_default_threshold_from_pd(pd) for pd in df["PD"].values]
        )

        # Get systematic factors
        sys_factors = factors.mean(axis=1)  # Average across counterparties

        # Base exposures (n_paths x n_counterparties)
        base_exposures = np.tile(df[EAD_col].values, (n_paths, 1))

        # Apply structural WWR
        wwr_exposures, conditional_pds = structural_wwr_adjustment(
            exposures=base_exposures,
            default_thresholds=default_thresholds,
            systematic_factors=sys_factors,
            asset_values=asset_values,
            correlation_expo_asset=correlation_expo_asset,  # Negative = wrong-way
            seed=seed + 888,
        )

        # Update EL with structural WWR
        df["EL_adj"] = (wwr_exposures * conditional_pds * df["LGD"].values).mean(axis=0)
        df["Simulated_Loss"] = (wwr_exposures * df["LGD"].values).mean(axis=0)
    else:
        df["EL_adj"] = adjust_for_wwr(
            np.array(EL_stochastic).reshape(1, -1),
            credit_factors=factor_means,
            sensitivity=wwr_corr,
        ).ravel()
    pass

    # Standardize WWR column names for reporting
    df["EL_WWR"] = df.get("EL_adj", df["EL"])
    df["UL_WWR"] = df["UL"]
    logger.debug("Added WWR columns to df")

    logger.debug("Computed stochastic EL/UL table:\n%s", df)
    return df
