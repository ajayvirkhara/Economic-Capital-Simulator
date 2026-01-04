"""
Firm-wide Economic Capital Aggregation across Market, Credit, and Operational Risk.
Now supports optional t-copula for fat-tailed joint loss simulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from scipy.stats import norm

from econ_capital.market_risk.shocks import mv_t_draws


def normalize_risk_results(
    market_results: Dict[str, Any],
    credit_results: Dict[str, Any],
    oprisk_results: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """
    Normalize the three risk outputs into a common {risk_type: {"EL": ..., "UL": ...}} format.
    """
    confidence_level = 0.999
    z = norm.ppf(confidence_level)

    # Market: Full measure = ES_1Y_999, EL ≈ 0, UL ≈ ES / z
    market_full = market_results.get("es_1y_999", market_results.get("var_1y_999", 0.0))
    market_el = 0.0
    market_ul = market_full / z if z > 0 else market_full

    # Credit Risk
    credit_el = credit_results.get("EL_total", 0.0)
    credit_ul = credit_results.get("UL_total", 0.0)

    # OpRisk: Full capital is VaR-like → treat as UL; EL assumed embedded or zero
    oprisk_full = oprisk_results.get("capital_999", 0.0)
    oprisk_el = oprisk_results.get("expected_loss", 0.0)
    oprisk_ul = (oprisk_full - oprisk_el) / z if z > 0 else (oprisk_full - oprisk_el)

    return {
        "Market": {"EL": market_el, "UL": market_ul, "Total_Standalone": market_full},
        "Credit": {
            "EL": credit_el,
            "UL": credit_ul,
            "Total_Standalone": credit_el + (credit_ul * z),
        },
        "OpRisk": {"EL": oprisk_el, "UL": oprisk_ul, "Total_Standalone": oprisk_full},
    }


def aggregate_economic_capital(
    risk_results: Dict[str, Dict[str, float]],
    confidence_level: float = 0.999,
    copula_df: Optional[float] = None,  # If provided, use t-copula simulation
    n_sim: int = 200_000,
    seed: int = 42,
) -> Tuple[float, float, float, pd.Series, float]:
    """
    Aggregate risks with diversification using either:
      - Gaussian copula (default, fast analytic)
      - Student-t copula (fat-tailed, Monte Carlo)

    Returns
    -------
    EL_total, UL_portfolio, EC_total, marginal_contributions, diversification_benefit
    """
    risk_types = list(risk_results.keys())
    n_risks = len(risk_types)  # Number of risk types (Market, Credit, OpRisk)

    el_vec = np.array([risk_results[rt]["EL"] for rt in risk_types])
    ul_vec = np.array([risk_results[rt]["UL"] for rt in risk_types])

    EL_total = el_vec.sum()

    # Fixed correlation matrix (Market-Credit-OpRisk order)
    corr_matrix = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])
    # Reorder to match actual risk_types order
    order = ["Market", "Credit", "OpRisk"]
    idx = [order.index(rt) for rt in risk_types]
    corr_matrix = corr_matrix[np.ix_(idx, idx)]

    if copula_df is not None and copula_df > 2:
        # --- t-Copula Monte Carlo (fat-tailed joint simulation) ---
        rng = np.random.default_rng(seed)
        t_shocks = mv_t_draws(
            n=n_sim, mu=np.zeros(n_risks), cov=corr_matrix, df=copula_df, rng=rng
        )
        # Scale shocks by individual ULs
        simulated_ul = t_shocks * ul_vec[None, :]
        total_losses = EL_total + simulated_ul.sum(axis=1)

        EC_total = float(np.quantile(total_losses, confidence_level))

        # Standalone EC using t-distribution (consistent with copula)
        from scipy.stats import t

        t_quantile = t.ppf(confidence_level, copula_df)
        standalone_ec = sum(
            risk_results[rt]["EL"] + t_quantile * risk_results[rt]["UL"]
            for rt in risk_types
        )
        diversification_benefit = standalone_ec - EC_total

        # Marginal via simulation: average contribution in tail scenarios
        cutoff = np.quantile(total_losses, confidence_level)
        tail_mask = total_losses >= cutoff  # Upper tail for losses (positive capital)

        tail_contrib = np.zeros(n_risks)
        for i in range(n_risks):
            # Individual contribution in tail: total tail loss minus others
            indiv_tail = (
                total_losses[tail_mask]
                - (EL_total - el_vec[i])
                - simulated_ul[tail_mask].sum(axis=1)
                + simulated_ul[tail_mask, i]
            )
            tail_contrib[i] = indiv_tail.mean()

        marginal = pd.Series(tail_contrib, index=risk_types, name="EC_Marginal")
        UL_portfolio = np.std(simulated_ul.sum(axis=1))

    else:
        # --- Gaussian Analytic Method ---
        portfolio_var = ul_vec @ corr_matrix @ ul_vec
        UL_portfolio = np.sqrt(max(portfolio_var, 0.0))

        z = norm.ppf(confidence_level)
        EC_total = EL_total + z * UL_portfolio

        standalone_components = []
        for i in range(n_risks):
            val = el_vec[i] + (ul_vec[i] * z)
            standalone_components.append(val)

        standalone_ec = sum(standalone_components)

        diversification_benefit = standalone_ec - EC_total

        if UL_portfolio > 1e-8:
            marginal_ul = z * (corr_matrix @ ul_vec) * ul_vec / UL_portfolio
        else:
            marginal_ul = np.zeros(n_risks)

        marginal = pd.Series(marginal_ul + el_vec, index=risk_types, name="EC_Marginal")

    return EL_total, UL_portfolio, EC_total, marginal, diversification_benefit
