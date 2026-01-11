"""
Master Aggregator Script for Firm-Wide Economic Capital

This script orchestrates the full Economic Capital simulation:
1. Runs Market Risk report → captures full results
2. Runs Credit Risk report → captures full structured results
3. Runs Operational Risk report → captures full stress test details and total capital
4. Aggregates all three into diversified firm-wide EC
5. Prints summary and saves consolidated text/JSON report
6. Generates detailed firm-wide Excel report with breakdowns from all risks
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from econ_capital.market_risk.run_marketrisk_report import main as run_market_risk
from econ_capital.credit_risk.run_creditrisk_report import main as run_credit_risk
from econ_capital.op_risk.run_oprisk_report import main as run_op_risk
from econ_capital.aggregate import normalize_risk_results, aggregate_economic_capital
from econ_capital.firmwide_reporting import generate_firmwide_ec_report


def main():
    print("=== Firm-Wide Economic Capital Aggregation ===")
    print(f"Run Date: {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    # 1. Run individual modules and capture full results
    print("1. Running Market Risk module...")
    full_market_results = run_market_risk()  # Full dict with breakdowns

    print("2. Running Credit Risk module...")
    full_credit_results = run_credit_risk()  # Full dict with portfolio details

    print("3. Running Operational Risk module...")
    full_op_results = run_op_risk()  # Dict with total_capital + stress test details

    # Extract scalars needed for aggregation
    market_results = {
        "var_1y_999": full_market_results.get("var_1y_999", 0.0),
        "es_1y_999": full_market_results.get("es_1y_999", 0.0),
    }

    # Extract both base and WWR (fallback to base if missing)
    credit_base_ec = full_credit_results.get("EC_total", 0.0)
    credit_wwr_ec = full_credit_results.get("EC_WWR_total", credit_base_ec)

    print(f"Credit EC (base):     £{credit_base_ec:,.0f}")
    if credit_wwr_ec != credit_base_ec:
        print(f"Credit EC (WWR):      £{credit_wwr_ec:,.0f}")
        print(f"WWR impact:           {credit_wwr_ec / credit_base_ec - 1:+.1%}")

    oprisk_baseline_metrics = full_op_results.get("baseline_metrics", {})
    oprisk_var_999 = oprisk_baseline_metrics.get("capital_999", 0.0)
    oprisk_expected_loss = oprisk_baseline_metrics.get("expected_loss", 0.0)
    op_results = {
        "capital_999": oprisk_var_999,
        "expected_loss": oprisk_expected_loss,
    }

    # 2. Normalize to common format
    normalized = normalize_risk_results(
        market_results=market_results,
        credit_results=full_credit_results,
        op_results=op_results,
    )

    print("\nNormalized Risk Contributions:")
    for risk, vals in normalized.items():
        print(f"   {risk:8} | EL: £{vals['EL']:>12,.0f} | UL: £{vals['UL']:>12,.0f}")

    # 3. Aggregate with diversification
    EL_total, UL_portfolio, EC_total, marginal, div_benefit = (
        aggregate_economic_capital(
            market_results=market_results,
            credit_results=full_credit_results,
            op_results=op_results,
            confidence_level=0.999,
            copula_df=7.0,
        )
    )

    print("\n" + "=" * 60)
    print("FIRM-WIDE ECONOMIC CAPITAL RESULTS")
    print("=" * 60)
    print(f"Total Expected Loss (EL)         : £{EL_total:>15,.0f}")
    print(f"Portfolio Unexpected Loss (UL)   : £{UL_portfolio:>15,.0f}")
    print(f"Total Economic Capital (99.9%)   : £{EC_total:>15,.0f}")
    print(f"Diversification Benefit          : £{div_benefit:>15,.0f}")
    print("\nMarginal Contributions:")
    for risk, contrib in marginal.items():
        print(f"   {risk:8} : £{contrib:>15,.0f}")

    # 4. Build enriched summary with detailed data
    readable_time = datetime.now().strftime("%d %B %Y, %H:%M:%S")

    output_dir = Path("econ_capital/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_timestamp": readable_time,
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "EL_total": EL_total,
        "UL_portfolio": UL_portfolio,
        "EC_total": EC_total,
        "diversification_benefit": div_benefit,
        "marginal_contributions": marginal.to_dict(),
        "individual_risks": normalized,
        "market_details": full_market_results,
        "credit_details": full_credit_results,
        "op_details": full_op_results,
        "correlations": {
            "Market": {"Credit": 0.3, "OpRisk": 0.1},
            "Credit": {"Market": 0.3, "OpRisk": 0.2},
            "OpRisk": {"Market": 0.1, "Credit": 0.2},
        },
    }

    # 5. Serializable objects
    def _to_serializable(val):
        """Recursively convert pandas objects and other non-JSON types to serializable forms."""
        if isinstance(val, pd.Series):
            return val.to_dict()
        if isinstance(val, pd.DataFrame):
            return val.to_dict(orient="records")
        if isinstance(val, np.integer):
            return int(val)
        if isinstance(val, np.floating):
            return float(val)
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, datetime):
            return val.isoformat()
        if isinstance(val, Path):
            return str(val)
        if hasattr(val, "__dict__"):
            return {k: _to_serializable(v) for k, v in val.__dict__.items()}
        if isinstance(val, (list, tuple)):
            return [_to_serializable(item) for item in val]
        if isinstance(val, dict):
            return {k: _to_serializable(v) for k, v in val.items()}
        return val

    # Create a JSON-safe copy
    try:
        safe_summary = json.loads(json.dumps(summary, default=_to_serializable))
    except (TypeError, ValueError) as e:
        print(f"Warning: Could not fully serialize summary to JSON due to: {e}")

    json_path = output_dir / f"FirmWide_EC_Summary_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(json_path, "w") as f:
        json.dump(safe_summary, f, indent=2)

    txt_path = output_dir / f"FirmWide_EC_Summary_{datetime.now():%Y%m%d_%H%M%S}.txt"
    with open(txt_path, "w") as f:
        f.write("FIRM-WIDE ECONOMIC CAPITAL SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Run Date: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        f.write(f"Total Expected Loss       : £{EL_total:,.0f}\n")
        f.write(f"Portfolio UL              : £{UL_portfolio:,.0f}\n")
        f.write(f"Total Economic Capital    : £{EC_total:,.0f}\n")
        f.write(f"Diversification Benefit   : £{div_benefit:,.0f}\n\n")
        f.write("Marginal Contributions:\n")
        for risk, contrib in marginal.items():
            f.write(f"   {risk:<8}: £{contrib:,.0f}\n")

    # 6. Generate detailed Excel report
    report_path = generate_firmwide_ec_report(
        aggregated_results=summary, output_dir="econ_capital/reports"
    )

    print(f"\nDetailed firm-wide Excel report generated: {report_path}")
    print("\nConsolidated summary saved to:")
    print(f"   {json_path}")
    print(f"   {txt_path}")

    print("\n=== Aggregation Complete ===")


if __name__ == "__main__":
    main()
