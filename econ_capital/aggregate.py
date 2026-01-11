"""
Firm-wide Economic Capital Aggregation across Market, Credit, and Operational Risk.
Now supports optional t-copula for fat-tailed joint loss simulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from scipy.stats import norm

from econ_capital.utils import setup_logging
from econ_capital.market_risk.shocks import mv_t_draws

logger = setup_logging(__name__)


def normalize_risk_results(
    market_results: Dict[str, Any],
    credit_results: Dict[str, Any],
    op_results: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """
    Normalize the three risk outputs into a common format.
    Prefers WWR-adjusted values when available.
    """
    confidence_level = 0.999
    z = norm.ppf(confidence_level)

    normalized = {}

    # Market: prefer ES → VaR → 0
    market_full = market_results.get(
        "es_1y_999", market_results.get("var_1y_999", market_results.get("UL", 0.0))
    )
    market_el = 0.0
    market_ul = market_full / z if z > 0 else market_full

    normalized["Market"] = {
        "EL": market_el,
        "UL": market_ul,
        "Total_Standalone": market_full,
        "label": "Market Risk",
    }

    # Credit: prefer WWR-adjusted if present
    credit = credit_results
    if isinstance(credit, dict) and "credit_details" in credit:
        credit = credit["credit_details"]

    # Prefer WWR-adjusted if present
    credit_el = credit.get(
        "EL_WWR_total", credit.get("EL_total", credit.get("EL", 0.0))
    )
    credit_ul = credit.get(
        "UL_WWR_total", credit.get("UL_total", credit.get("UL", 0.0))
    )
    credit_ec = credit_el + z * credit_ul

    normalized["Credit"] = {
        "EL": credit_el,
        "UL": credit_ul,
        "Total_Standalone": credit_ec,
        "label": "Credit (WWR-adjusted)" if "EL_WWR_total" in credit else "Credit",
    }

    # OpRisk: capital_999 is full VaR-like → split into EL + UL
    oprisk_full = op_results.get("capital_999", 0.0)
    oprisk_el = op_results.get("expected_loss", 0.0)
    oprisk_ul = (oprisk_full - oprisk_el) / z if z > 0 else (oprisk_full - oprisk_el)

    normalized["OpRisk"] = {
        "EL": oprisk_el,
        "UL": oprisk_ul,
        "Total_Standalone": oprisk_full,
        "label": "Operational Risk",
    }

    return normalized


def aggregate_economic_capital(
    market_results: Dict[str, Any],
    credit_results: Dict[str, Any],
    op_results: Dict[str, Any],
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

    # Step 1: Normalize the raw inputs
    normalized = normalize_risk_results(
        market_results=market_results,
        credit_results=credit_results,
        op_results=op_results,
    )

    # Step 2: Extract vectors from the normalized data
    risk_types = list(normalized.keys())
    el_vec = np.array([normalized[rt]["EL"] for rt in risk_types])
    ul_vec = np.array([normalized[rt]["UL"] for rt in risk_types])

    EL_total = el_vec.sum()

    corr_matrix = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])

    order = ["Market", "Credit", "OpRisk"]
    idx = [order.index(rt) for rt in risk_types]
    corr_matrix = corr_matrix[np.ix_(idx, idx)]

    if copula_df is not None and copula_df > 2:
        # --- t-Copula Monte Carlo (fat-tailed joint simulation) ---
        rng = np.random.default_rng(seed)
        t_shocks = mv_t_draws(
            n=n_sim,
            mu=np.zeros(len(risk_types)),
            cov=corr_matrix,
            df=copula_df,
            rng=rng,
        )
        # Scale shocks by individual ULs
        simulated_ul = t_shocks * ul_vec[None, :]
        total_losses = EL_total + simulated_ul.sum(axis=1)

        EC_total = float(np.quantile(total_losses, confidence_level))

        # Standalone EC using t-distribution (consistent with copula)
        from scipy.stats import t

        t_quantile = t.ppf(confidence_level, copula_df)
        standalone_ec = sum(
            normalized[rt]["EL"] + t_quantile * normalized[rt]["UL"]
            for rt in risk_types
        )
        diversification_benefit = standalone_ec - EC_total

        # Marginal via simulation: average contribution in tail scenarios
        cutoff = np.quantile(total_losses, confidence_level)
        tail_mask = total_losses >= cutoff  # Upper tail for losses (positive capital)

        tail_contrib = np.zeros(len(risk_types))
        for i in range(len(risk_types)):
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

        standalone_ec = sum(
            normalized[rt]["EL"] + z * normalized[rt]["UL"] for rt in risk_types
        )
        diversification_benefit = standalone_ec - EC_total

        # Marginal: Euler allocation of UL part + pro-rata EL
        if UL_portfolio > 1e-8:
            marginal_ul = z * (corr_matrix @ ul_vec) * ul_vec / UL_portfolio
        else:
            marginal_ul = np.zeros(len(risk_types))

        el_share = (
            el_vec / EL_total
            if EL_total > 0
            else np.ones(len(risk_types)) / len(risk_types)
        )
        marginal_ec = marginal_ul + el_share * EL_total

        marginal = pd.Series(marginal_ec, index=risk_types, name="EC_Marginal")

    return EL_total, UL_portfolio, EC_total, marginal, diversification_benefit
