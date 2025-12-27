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

# Internal package imports
from econ_capital.credit_risk.data_loaders import (
    load_dummy_credit_data,
    load_issuer_spreads_csv,
)
from econ_capital.credit_risk.trade_models import Trade, NettingSet
from econ_capital.credit_risk.csa import CSA
from econ_capital.credit_risk.exposure_engine import ExposureEngine
from econ_capital.credit_risk.default_model import CreditInputs, compute_expected_loss
from econ_capital.credit_risk.ccr_engine import aggregate_credit_losses
from econ_capital.credit_risk.creditrisk_reporting import generate_creditrisk_report
from econ_capital.credit_risk.config import DEFAULT_CONFIG


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

    print("--- Credit Risk Report Engine ---")
    print(f"Root: {PROJECT_ROOT}")

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
        print(f"Loading exposures from: {csv_file}")
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
    print(f"Simulating risks for {len(unique_cptys)} counterparties...")

    # Shared market paths for systematic exposure (Stylized GBM)
    n_paths = config.get("n_paths", 2000)
    times = np.linspace(0, 1.0, config.get("horizon_steps", 6))
    rng = np.random.default_rng(config.get("seed", 42))

    # Simulate a single factor (SP500) that drives exposures
    prices = 100 * np.exp(
        np.cumsum(rng.normal(0.01, 0.1, size=(n_paths, len(times))), axis=1)
    )
    market_paths = {"SP500": prices}

    results_list = []

    for cpty in unique_cptys:
        cpty_data = df_positions[df_positions["counterparty"] == cpty]

        # Aggregate Notional/EAD from CSV
        total_ead_input = cpty_data["value"].sum()
        pd_val = cpty_data["pd_annual"].mean()

        # Build Netting Set
        trades = [Trade(name=f"{cpty}_Exposure", factor="SP500", w=total_ead_input)]
        ns = NettingSet(counterparty=cpty, trades=trades, csa=CSA(threshold=1_000_000))

        # Run Exposure Engine
        engine = ExposureEngine(ns, market_paths, times, n_paths)
        _, summary = engine.compute_exposure_profile()
        ead_final = summary["EAD_final"].iloc[0]

        # Compute Expected Loss (EL)
        credit_input = CreditInputs(counterparty=cpty, pd_annual=pd_val, lgd=0.45)
        el, _ = compute_expected_loss(
            times, np.full_like(times, ead_final), credit_input
        )

        # Compute Unexpected Loss (UL) approximation
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

    EL_total, UL_total, EC_total, alloc_fractions = aggregate_credit_losses(
        df_results["EL"].values,
        df_results["UL"].values,
        corr_matrix,
        confidence=config.get("confidence_level", 0.999),
    )

    # Map allocations (£) back to the dataframe
    df_results["EC_Marginal"] = alloc_fractions * EC_total

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

    print(f"\nReport successfully generated: {output_path.name}")
    print(f"Target Directory: {REPORTS_DIR}")
    print(f"Total Portfolio EC: £{EC_total:,.2f}")

    return report_data


if __name__ == "__main__":
    main()
