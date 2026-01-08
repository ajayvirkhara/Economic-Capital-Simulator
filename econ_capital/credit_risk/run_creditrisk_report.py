"""
Credit Risk Economic Capital (EC) Report Driver
===============================================
- Loads counterparty exposures from econ_capital/credit_risk/data/
- Simulates exposure and credit loss metrics.
- Computes Portfolio Economic Capital with Sector-based differentiation.
- Generates a regulatory-grade Excel report in econ_capital/credit_risk/reports/
"""

from __future__ import annotations
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from scipy.stats import norm
import logging

# Internal package imports
from econ_capital.credit_risk.data_loaders import (
    load_dummy_credit_data,
    load_issuer_spreads_csv,
)
from econ_capital.credit_risk.trade_models import Trade, NettingSet
from econ_capital.credit_risk.csa import CSA
from econ_capital.credit_risk.exposure_engine import ExposureEngine
from econ_capital.credit_risk.creditrisk_reporting import generate_creditrisk_report
from econ_capital.credit_risk.config import DEFAULT_CONFIG

# Silence the noisy exposure engine logger
lda_logger = logging.getLogger("econ_capital.credit_risk.exposure_engine")
lda_logger.setLevel(logging.WARNING)
lda_logger.propagate = False


def main():
    # ------------------------------------------------------------------
    # 1. PATH SETUP
    # ------------------------------------------------------------------
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    DATA_DIR = PROJECT_ROOT / "econ_capital" / "credit_risk" / "data"
    REPORTS_DIR = PROJECT_ROOT / "econ_capital" / "credit_risk" / "reports"

    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- Credit Risk Report Engine ---\n")
    print(f"Root: {PROJECT_ROOT}\n")

    # ------------------------------------------------------------------
    # 2. LOAD CONFIG
    # ------------------------------------------------------------------
    config = DEFAULT_CONFIG.copy()
    # Override with a local YAML if it exists in the root
    config_path = PROJECT_ROOT / "config" / "credit_config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            yaml_conf = yaml.safe_load(f) or {}
            config.update(yaml_conf.get("credit_risk", {}))

    # ------------------------------------------------------------------
    # 3. LOAD DATA
    # ------------------------------------------------------------------
    csv_file = DATA_DIR / "counterparty_exposures.csv"

    if csv_file.exists():
        print(f"Loading exposures from: {csv_file}\n")
        df_positions = load_issuer_spreads_csv(str(csv_file))
        # Filter for EAD measures
        df_positions = df_positions[df_positions["measure"].str.upper() == "EAD"]
    else:
        print(f"WARNING: {csv_file} not found. Using default dummy data.")
        df_positions = load_dummy_credit_data()

    # ------------------------------------------------------------------
    # 4. SIMULATION & RISK ENGINE
    # ------------------------------------------------------------------
    unique_cptys = df_positions["counterparty"].unique()
    print(f"Simulating risks for {len(unique_cptys)} counterparties...\n")

    # Shared market paths for systematic exposure (Stylized GBM)
    n_paths = config.get("n_paths", 10000)
    times = np.linspace(0, 5, config.get("horizon_steps", 51))
    rng = np.random.default_rng(config.get("seed", 42))

    # Simulate SP500 paths with positive drift and high volatility
    mu_annual = 0.08  # 8% expected annual return
    sigma_annual = 0.20  # 20% annual volatility

    # Time steps: 50 intervals over 10 years
    dt = np.diff(times)  # shape (50,)
    dt = dt[np.newaxis, :]  # shape (1, 50) for broadcasting

    Z = rng.standard_normal((n_paths, len(times) - 1))

    log_returns = (mu_annual - 0.5 * sigma_annual**2) * dt + sigma_annual * np.sqrt(
        dt
    ) * Z

    log_prices = np.cumsum(
        np.concatenate([np.zeros((n_paths, 1)), log_returns], axis=1), axis=1
    )

    prices = 100.0 * np.exp(log_prices)  # Start at S0 = 100

    market_paths = {"SP500": prices}

    results_list = []

    for cpty in unique_cptys:
        cpty_data = df_positions[df_positions["counterparty"] == cpty]

        # Aggregate Notional/EAD from CSV
        total_ead_input = cpty_data["value"].sum()
        pd_val = cpty_data["pd_annual"].mean()

        # Build Netting Set
        trades = [
            Trade(
                name=f"{cpty}_Exposure",
                factor="SP500",
                w=total_ead_input * 1.0,  # linear exposure
                gamma=0.01,  # positive convexity
            )
        ]
        ns = NettingSet(counterparty=cpty, trades=trades, csa=CSA(threshold=5_000_000))

        # Run Exposure Engine
        engine = ExposureEngine(ns, market_paths, times, n_paths)
        _, summary = engine.compute_exposure_profile()
        ead_final = summary["EAD_final"].iloc[0]

        # Compute Expected Loss (EL) - Basel Standard
        ead_final_scalar = summary["EAD_final"].iloc[0]
        el = ead_final_scalar * pd_val * 0.45

        # Compute Bernoulli Unexpected Loss (UL)
        # UL = EAD * LGD * sqrt(PD * (1-PD))
        ul = ead_final * 0.45 * np.sqrt(pd_val * (1 - pd_val))

        # Determine Sector for reporting/differentiation
        sector = "Corporate"
        if "BANK" in cpty.upper():
            sector = "Bank"
        elif "HEDGE" in cpty.upper() or "FUND" in cpty.upper():
            sector = "Hedge Fund"

        results_list.append(
            {
                "name": cpty,
                "Sector": sector,
                "EAD": ead_final,
                "PD": pd_val,
                "LGD": 0.45,
                "EL": el,
                "UL": ul,
            }
        )

    df_results = pd.DataFrame(results_list)

    # ------------------------------------------------------------------
    # 5. PORTFOLIO AGGREGATION (ECONOMIC CAPITAL)
    # ------------------------------------------------------------------
    n = len(df_results)
    corr_matrix = np.full((n, n), config.get("default_correlation", 0.2))
    np.fill_diagonal(corr_matrix, 1.0)

    # Portfolio Unexpected Loss (diversified)
    ul_vec = df_results["UL"].values
    portfolio_var = ul_vec @ corr_matrix @ ul_vec
    UL_total = np.sqrt(max(portfolio_var, 0.0))

    # Portfolio totals
    EL_total = df_results["EL"].sum()
    z = norm.ppf(config.get("confidence_level", 0.999))
    EC_total = EL_total + z * UL_total

    # Marginal EC: Euler allocation on UL part + pro-rata EL
    if UL_total > 1e-8:
        marginal_ul = z * (corr_matrix @ ul_vec) * ul_vec / UL_total
    else:
        marginal_ul = np.zeros(n)

    df_results["EC_Marginal"] = marginal_ul + df_results["EL"].values

    # Sort by Capital impact for the report
    df_results = df_results.sort_values("EC_Marginal", ascending=False)

    # ------------------------------------------------------------------
    # 6. GENERATE REPORT
    # ------------------------------------------------------------------
    report_data = {
        "EL_total": EL_total,
        "UL_total": UL_total,
        "EC_total": EC_total,
        "capital_breakdown": df_results.set_index("name")["EC_Marginal"],
        "full_data": df_results,
    }

    output_path = generate_creditrisk_report(
        config=config, engine=None, results=report_data, output_dir=str(REPORTS_DIR)
    )

    # Detailed Terminal Output
    print("\n" + "=" * 70)
    print("CREDIT RISK EXECUTION SUMMARY")
    print("=" * 70)
    print(f"{'Report Name:':<20} {output_path.name}")
    print(f"{'Total EL:':<20} £{EL_total:,.2f}")
    print(f"{'Total UL:':<20} £{UL_total:,.2f}")
    print(f"{'Portfolio EC:':<20} £{EC_total:,.2f}")
    print("-" * 70)

    # Sector Breakdown
    print("EC by Sector:")
    sector_summary = df_results.groupby("Sector")["EC_Marginal"].sum()
    for sector, val in sector_summary.items():
        print(f"  {sector:<18} £{val:,.2f}")

    print("-" * 70)

    # Top 5 Counterparties
    print(f"{'Top 5 Counterparties by Marginal EC':<35} {'EC Contribution'}")
    for _, row in df_results.head(5).iterrows():
        print(f"  {row['name']:<33} £{row['EC_Marginal']:,.2f}")

    print("=" * 70 + "\n")

    print(f"\nReport successfully generated: {output_path}\n")

    return report_data


if __name__ == "__main__":
    main()
