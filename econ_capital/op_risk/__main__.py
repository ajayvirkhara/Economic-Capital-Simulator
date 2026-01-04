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
from .lda_engine import lda_run_engine

logger = setup_logging(__name__)


def main() -> None:
    """Run a demo simulation using the full LDA engine."""
    print("Running Operational Risk LDA Engine Demo...\n")

    try:
        # Run the full engine with default config
        loss_dist, fitted_models, metrics = lda_run_engine()

        print("=== Simulation Successful ===")
        print(f"Simulated paths: {len(loss_dist):,}")
        print(f"Expected Loss:     £{metrics.get('expected_loss', 0):,.0f}")
        print(f"99.9% VaR (Capital): £{metrics.get('capital_999', 0):,.0f}")
        print(f"Mean Loss:         £{metrics.get('mean_loss', 0):,.0f}")
        print(f"Std Dev:           £{metrics.get('std_loss', 0):,.0f}")

        print("\n--- Fitted Models (first UoM) ---")
        if fitted_models:
            first_uom = next(iter(fitted_models))
            print(f"{first_uom}:")
            print(f"  Frequency: λ = {fitted_models[first_uom]['freq_params']['lambda']:.3f}")
            print(f"  Severity:  μ = {fitted_models[first_uom]['sev_params']['lognormal_mu']:.3f}, "
                  f"σ = {fitted_models[first_uom]['sev_params']['lognormal_sigma']:.3f}")

        print("\nDemo completed successfully.")
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    logger.info("Running Operational Risk demo")
    main()
    logger.info("Demo finished.")