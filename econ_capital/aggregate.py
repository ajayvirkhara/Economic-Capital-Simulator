"""
Firm-wide Economic Capital Aggregation across Market, Credit, and Operational Risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, Any, Tuple


def normalize_risk_results(
    market_results: Dict[str, Any],
    credit_results: Dict[str, Any],
    oprisk_capital: float,
) -> Dict[str, Dict[str, float]]:
    """
    Normalize the three risk outputs into a common {risk_type: {"EL": ..., "UL": ...}} format.
    """
    # Market Risk: Use 1Y VaR 99.9% as proxy for EC
    # Assume EL ≈ 0 for Market Risk (pure unexpected loss)
    market_ec = market_results.get("var_1y_999", 0.0) or market_results.get(
        "es_1y_999", 0.0
    )

    # Credit Risk: Already has EL_total and UL_total (EC = EL + z * UL)
    credit_el = credit_results.get("EL_total", 0.0)
    credit_ul = credit_results.get("UL_total", 0.0)

    # OpRisk: Capital is full VaR → treat as unexpected loss (EL typically embedded or small)
    oprisk_ec = float(oprisk_capital)

    return {
        "Market": {"EL": 0.0, "UL": market_ec},
        "Credit": {"EL": credit_el, "UL": credit_ul},
        "OpRisk": {"EL": 0.0, "UL": oprisk_ec},  # Standard treatment
    }


def aggregate_economic_capital(
    risk_results: Dict[str, Dict[str, float]],
    correlations: Dict[str, Dict[str, float]] | None = None,
    confidence_level: float = 0.999,
) -> Tuple[float, float, float, pd.Series, float]:
    """
    Aggregate diversified Economic Capital using Gaussian copula approximation.

    Returns:
        EL_total, UL_portfolio, EC_total, marginal_contributions, diversification_benefit
    """
    if correlations is None:
        correlations = {
            "Market": {"Credit": 0.3, "OpRisk": 0.1},
            "Credit": {"Market": 0.3, "OpRisk": 0.2},
            "OpRisk": {"Market": 0.1, "Credit": 0.2},
        }

    risk_types = list(risk_results.keys())
    el_vec = np.array([risk_results[rt]["EL"] for rt in risk_types])
    ul_vec = np.array([risk_results[rt]["UL"] for rt in risk_types])

    EL_total = float(el_vec.sum())

    # Build full correlation matrix
    n = len(risk_types)
    corr_matrix = np.eye(n)
    for i, rt1 in enumerate(risk_types):
        for j, rt2 in enumerate(risk_types):
            if i != j:
                corr = correlations.get(rt1, {}).get(
                    rt2, correlations.get(rt2, {}).get(rt1, 0.0)
                )
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

    # Portfolio variance
    portfolio_var = ul_vec @ corr_matrix @ ul_vec
    UL_portfolio = np.sqrt(max(portfolio_var, 0.0))

    # Total Economic Capital
    z = norm.ppf(confidence_level)
    EC_total = EL_total + z * UL_portfolio

    # Standalone (undiversified Ecomomic Capital)
    standalone_ec = sum(
        risk_results[rt]["EL"] + z * risk_results[rt]["UL"] for rt in risk_types
    )
    diversification_benefit = standalone_ec - EC_total

    # Marginal contributions using Euler allocation
    if UL_portfolio > 1e-8:
        marginal_ul = z * (corr_matrix @ ul_vec) * ul_vec / UL_portfolio
    else:
        marginal_ul = np.zeros(n)
    marginal = pd.Series(marginal_ul + el_vec, index=risk_types, name="EC_Marginal")

    return EL_total, UL_portfolio, EC_total, marginal, diversification_benefit
