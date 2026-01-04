"""
Economic Capital Simulator — Unified Firm-Wide Framework

High-level API for running the full Economic Capital simulation across:
- Market Risk
- Credit Risk
- Operational Risk

Provides:
- `run_full_simulation()`: Orchestrates all modules and returns aggregated EC
- `aggregate_economic_capital()`: Core diversification logic
- Clean imports and version exposure
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

__all__ = [
    "run_full_simulation",
    "aggregate_economic_capital",
    "normalize_risk_results",
]

# Public API from submodules
from .aggregate import aggregate_economic_capital, normalize_risk_results


def run_full_simulation(
    config_override: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Execute all three risk modules and return firm-wide aggregated Economic Capital.

    Parameters
    ----------
    config_override : dict, optional
        Configuration overrides passed to individual modules (future extension)
    verbose : bool, default True
        Print progress and summary to console

    Returns
    -------
    dict
        Aggregated results including:
        - EL_total, UL_portfolio, EC_total
        - marginal_contributions
        - diversification_benefit
        - individual_risk_contributions
    """
    # Lazy imports of driver mains
    from .credit_risk.run_creditrisk_report import main as _run_credit_risk
    from .market_risk.run_marketrisk_report import main as _run_market_risk
    from .op_risk.run_oprisk_report import main as _run_op_risk

    if verbose:
        print("=" * 70)
        print("ECONOMIC CAPITAL SIMULATOR — FIRM-WIDE RUN")
        print(f"Start Time: {datetime.now():%Y-%m-%d %H:%M:%S}")
        print("=" * 70)

    config_override = config_override or {}

    # 1. Execute individual risk modules
    if verbose:
        print("\n1. Running Market Risk module...")
    market_results = _run_market_risk()  # Must return results dict

    if verbose:
        print("2. Running Credit Risk module...")
    credit_results = _run_credit_risk()  # Must return report_data dict

    if verbose:
        print("3. Running Operational Risk module...")
    oprisk_capital = _run_op_risk()  # Must return total capital (float)

    if verbose:
        print("\nAll individual risk modules completed successfully.")

    # 2. Normalize results to common format
    normalized = normalize_risk_results(
        market_results=market_results,
        credit_results=credit_results,
        oprisk_capital=oprisk_capital,
    )

    if verbose:
        print("\nNormalized Individual Contributions:")
        for risk, vals in normalized.items():
            print(
                f"   {risk:8} | EL: £{vals['EL']:>15,.0f} | UL: £{vals['UL']:>15,.0f}"
            )

    # 3. Aggregate with diversification
    EL_total, UL_portfolio, EC_total, marginal, div_benefit = (
        aggregate_economic_capital(risk_results=normalized)
    )

    if verbose:
        print("\n" + "=" * 70)
        print("FIRM-WIDE ECONOMIC CAPITAL RESULTS")
        print("=" * 70)
        print(f"Total Expected Loss (EL)       : £{EL_total:>18,.0f}")
        print(f"Portfolio Unexpected Loss (UL) : £{UL_portfolio:>18,.0f}")
        print(f"Total Economic Capital (99.9%) : £{EC_total:>18,.0f}")
        print(f"Diversification Benefit        : £{div_benefit:>18,.0f}")
        print("\nMarginal Contributions:")
        for risk, contrib in marginal.items():
            print(f"   {risk:<8} : £{contrib:>18,.0f}")
        print(f"\nEnd Time: {datetime.now():%Y-%m-%d %H:%M:%S}")
        print("=" * 70)

    # 4. Return structured results
    aggregated_results = {
        "run_timestamp": datetime.now().isoformat(),
        "EL_total": EL_total,
        "UL_portfolio": UL_portfolio,
        "EC_total": EC_total,
        "diversification_benefit": div_benefit,
        "marginal_contributions": marginal.to_dict(),
        "individual_risks": normalized,
        "config_override": config_override,
    }

    # 5. Save summary JSON
    reports_root = Path("econ_capital/reports")
    reports_root.mkdir(parents=True, exist_ok=True)
    summary_path = (
        reports_root / f"FirmWide_EC_Summary_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    with open(summary_path, "w") as f:
        import json

        json.dump(aggregated_results, f, indent=2, default=str)

    if verbose:
        print(f"\nConsolidated summary saved to: {summary_path}")

    return aggregated_results


# Load global config
GLOBAL_CONFIG = (
    yaml.safe_load(Path("default.yaml").open()) if Path("default.yaml").exists() else {}
)
