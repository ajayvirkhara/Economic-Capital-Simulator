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
from typing import Dict, Any

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
from econ_capital.correlation_models import DynamicCorrelationEstimator
from econ_capital.utils import setup_logging

logger = setup_logging(__name__)

# ========== EXTRACT TIME SERIES ==========


def _extract_market_returns(market_results: Dict[str, Any]) -> pd.Series:
    """Extract market return time series for correlation estimation."""
    # Assuming market_results contains risk_factors
    if "risk_factors" in market_results:
        # Use aggregate market factor (e.g., SPY or first factor)
        factors = market_results["risk_factors"]
        if isinstance(factors, pd.DataFrame) and len(factors.columns) > 0:
            return factors.iloc[:, 0]  # First factor as proxy

    # Fallback: simulate from results
    vol_annual = np.sqrt(market_results.get("UL", 0)) / np.sqrt(252)
    days = 252
    return pd.Series(
        np.random.normal(0, vol_annual, days),
        index=pd.date_range(end=pd.Timestamp.now(), periods=days, freq="D"),
    )


def _extract_credit_spreads(credit_results: Dict[str, Any]) -> pd.Series:
    """Extract credit spread changes for correlation estimation."""
    # If credit_results has historical spreads, use them
    # Otherwise synthesize from portfolio volatility

    ul_credit = credit_results.get("UL_total", credit_results.get("UL", 0))
    vol_daily = np.sqrt(ul_credit) / np.sqrt(252)
    days = 252

    return pd.Series(
        np.random.normal(0, vol_daily * 0.01, days),  # Spread changes in %
        index=pd.date_range(end=pd.Timestamp.now(), periods=days, freq="D"),
    )


def _extract_oprisk_losses(op_results: Dict[str, Any]) -> pd.Series:
    """Extract operational loss time series."""
    # OpRisk is event-driven; synthesize daily loss indicators

    baseline = op_results.get("baseline_metrics", {})
    expected_loss = baseline.get("expected_loss", 0)
    frequency = expected_loss / 1e6 if expected_loss > 0 else 0.1  # events/day

    days = 252
    # Poisson events
    events = np.random.poisson(frequency, days)

    return pd.Series(
        events, index=pd.date_range(end=pd.Timestamp.now(), periods=days, freq="D")
    )


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

    aggregation_kwargs = {
        "confidence_level": 0.999,
        "copula_df": 3.0,  # enforced here
        "n_sim": 750_000,  # safer for df=3
        "seed": 42,
    }

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

    # 3. Dynamic correlation estimation

    print("\nEstimating dynamic inter-risk correlations...")

    # Extract time series for correlation estimation
    market_returns = _extract_market_returns(full_market_results)
    credit_spreads = _extract_credit_spreads(full_credit_results)
    oprisk_losses = _extract_oprisk_losses(full_op_results)

    # Initialize estimator
    corr_estimator = DynamicCorrelationEstimator(
        method="regime_switching",  # or "rolling"
        window=252,
        stress_multiplier=1.5,
    )

    # Market returns stress override: create a regime break
    if len(market_returns) >= 252:
        stress_override = np.concatenate(
            [
                np.random.normal(
                    0, 0.005, 192
                ),  # First 192 days: very calm (0.5% daily vol)
                np.random.normal(0, 0.008, 40),  # Days 192-231: normal (0.8% daily vol)
                np.random.normal(
                    0, 0.04, 20
                ),  # Days 232-251: STRESS (4% daily vol - 5x spike)
            ]
        )

        market_returns_test = pd.Series(
            stress_override,
            index=market_returns.index[-252:]
            if len(market_returns) >= 252
            else market_returns.index,
        )
        print("   >>> STRESS TEST MODE: Injected synthetic volatility spike <<<")
        print(
            f"   Recent 20-day vol: {market_returns_test.tail(20).std() * np.sqrt(252):.1%}"
        )
        print(
            f"   Long 60-day vol:   {market_returns_test.tail(60).std() * np.sqrt(252):.1%}"
        )
        print(
            f"   Vol Ratio: {(market_returns_test.tail(20).std() / market_returns_test.tail(60).std()):.2f}x"
        )
    else:
        market_returns_test = market_returns
        print("   >>> NORMAL MODE: Using actual market data <<<")

    # Estimate current correlation matrix
    dynamic_corr_matrix, regime_name = corr_estimator.estimate_correlation_matrix(
        market_returns=market_returns_test,
        credit_spreads=credit_spreads,
        oprisk_losses=oprisk_losses,
    )

    print(f"   Detected Regime: {regime_name}")
    print("   Dynamic Correlation Matrix:")
    print(f"   {dynamic_corr_matrix}")

    # 4. Aggregate with diversification
    print("\nAggregating with t-copula (df=3)...")

    use_t_copula = (
        aggregation_kwargs.get("copula_df") is not None
        and aggregation_kwargs["copula_df"] > 2
    )

    if use_t_copula:
        df = aggregation_kwargs["copula_df"]
        nsim = aggregation_kwargs.get("n_sim", 500_000)
        method_text = f"t-copula (ν = {df:.1f}, Monte Carlo, {nsim:,} paths)"
    else:
        method_text = "Gaussian copula (closed-form / variance-covariance)"

    logger.info(f"Normalization method: {method_text}")

    for risk, vals in normalized.items():
        logger.info(
            f"  {risk}: UL=£{vals['UL']:,.0f}, "
            f"Total_Standalone=£{vals['Total_Standalone']:,.0f}"
        )

    logger.info(
        f"Using {'dynamic' if dynamic_corr_matrix is not None else 'static'} "
        f"correlation matrix (regime: {regime_name or 'default'}) "
        f"with shape {(dynamic_corr_matrix.shape if dynamic_corr_matrix is not None else '(3,3)')} "
    )

    EL_total, UL_portfolio, EC_total, marginal, div_benefit = (
        aggregate_economic_capital(
            market_results=full_market_results,
            credit_results=full_credit_results,
            op_results=op_results,
            correlation_matrix=dynamic_corr_matrix,
            correlation_regime=regime_name,
            **aggregation_kwargs,
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

    # 5. Build enriched summary with detailed data
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
            "matrix": dynamic_corr_matrix.tolist(),
            "regime": regime_name,
            "method": "regime_switching",
            "static_fallback": {
                "Market": {"Credit": 0.3, "OpRisk": 0.1},
                "Credit": {"Market": 0.3, "OpRisk": 0.2},
                "OpRisk": {"Market": 0.1, "Credit": 0.2},
            },
        },
        "correlation_matrix_array": dynamic_corr_matrix,
        "correlation_regime": regime_name,
    }

    # 6. Serializable objects
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

    # 7. Generate detailed Excel report
    _ = generate_firmwide_ec_report(
        aggregated_results=summary, output_dir="econ_capital/reports"
    )

    print("\nConsolidated summary saved to:")
    print(f"   {json_path}")
    print(f"   {txt_path}")

    print("\n=== Aggregation Complete ===")

    return {
        "EL_total": EL_total,
        "UL_portfolio": UL_portfolio,
        "EC_total": EC_total,
        "diversification_benefit": div_benefit,
        "marginal_contributions": marginal,
        "individual_risks": normalized,
        "market_details": full_market_results,
        "credit_details": full_credit_results,
        "op_details": full_op_results,
        "run_timestamp": datetime.now().isoformat(),
        "aggregation_kwargs": aggregation_kwargs,
    }


if __name__ == "__main__":
    main()
