"""
Demo run for the Operational Risk engine.

Execute directly with:
    python -m econ_capital.op_risk
"""

from __future__ import annotations

import numpy as np
from econ_capital.utils import setup_logging
from .lda_engine import run_monte_carlo_simulation
from .data_loaders import load_frequency_data, load_severity_data

logger = setup_logging(__name__)


def main() -> None:
    """Run a demo simulation of operational risk economic capital."""

    # --- Load data ---
    freq_df = load_frequency_data("data/frequency_real.csv")
    sev_df = load_severity_data("data/severity_real.csv")

    # Build models from CSV inputs
    fitted_models = {
        "frequency": {
            "dist": "poisson",
            "lambda": freq_df["lambda"].iloc[0],
        },
        "severity": {
            "dist": "lognormal",
            "mu": sev_df["mu"].iloc[0],
            "sigma": sev_df["sigma"].iloc[0],
        },
    }

    # --- Configure simulation ---
    config = {
        "n_years": 100_000,
        "seed": 123,
        "mitigation_factor": 0.90,
    }

    # --- Run LDA simulation ---
    results = run_monte_carlo_simulation(fitted_models=fitted_models, config=config)
    loss_dist = results["loss_distribution"]

    # --- Compute capital metrics ---
    expected_loss = loss_dist.mean()
    var_999 = np.quantile(loss_dist, 0.999)
    economic_capital = var_999 - expected_loss

    # --- Print results ---
    print("=== Operational Risk Economic Capital Results ===")
    print(f"Expected Loss:        {expected_loss:,.0f}")
    print(f"99.9% VaR:            {var_999:,.0f}")
    print(f"Economic Capital:     {economic_capital:,.0f}\n")

    print("=== Frequency Stats ===")
    print(results["frequency_stats"])

    print("\n=== Severity Stats ===")
    print(results["severity_stats"])


if __name__ == "__main__":
    logger.info("Running Operational Risk demo")
    main()
    logger.info("Simulation finished successfully.")
