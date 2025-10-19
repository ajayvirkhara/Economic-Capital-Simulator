"""
Standalone demo for ExposureEngine using stylised GBM paths.

Run with:
    python -m econ_capital.credit_risk.demo_exposure
"""

import numpy as np

from econ_capital.utils import setup_logging, set_global_seed, timed_section
from econ_capital.credit_risk import (
    Trade,
    NettingSet,
    CSA,
    ExposureEngine,
    compute_counterparty_risk_profiles,
    aggregate_credit_losses,
)


# pylint: disable=too-many-positional-arguments
def _simulate_sp500_paths(
    n_paths: int,
    times: np.ndarray,
    s0=100.0,
    mu=0.0,
    sigma=0.2,
    seed=42,
):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, len(times)))
    S = np.empty_like(z)
    S[:, 0] = s0
    dt = np.diff(np.concatenate([[0.0], times]))
    for k in range(1, len(times)):
        S[:, k] = S[:, k - 1] * np.exp(
            (mu - 0.5 * sigma**2) * dt[k - 1] + sigma * np.sqrt(dt[k - 1]) * z[:, k - 1]
        )
    return {"SP500": S}


def main():
    set_global_seed(42)
    logger = setup_logging(__name__)

    # Time grid & simulation
    times = np.linspace(0.0, 1.0, 6)
    n_paths = 5000
    market_paths = _simulate_sp500_paths(
        n_paths, times, s0=100.0, mu=0.0, sigma=0.2, seed=42
    )

    # Define trades & CSA
    trades = [
        Trade(name="IRS_like", factor="SP500", w=0.8, gamma=0.0),
        Trade(name="Option_like", factor="SP500", w=0.1, gamma=0.002),
    ]
    csa = CSA(threshold=1.0, mta=0.1, im=0.5, vm_calls_per_day=1)

    # Create netting set and engine
    ns = NettingSet(counterparty="CPTY_A", trades=trades, csa=csa)
    engine = ExposureEngine(netting_set=ns, market_paths=market_paths, times=times)

    with timed_section("compute_exposure_profile"):
        _, summary = engine.compute_exposure_profile()

    # Portfolio-level Credit Risk Example
    counterparties = [
        {"name": "CPTY_A", "EAD": 100, "PD": 0.02, "LGD": 0.6},
        {"name": "CPTY_B", "EAD": 150, "PD": 0.015, "LGD": 0.5},
        {"name": "CPTY_C", "EAD": 120, "PD": 0.01, "LGD": 0.4},
    ]

    df = compute_counterparty_risk_profiles(counterparties)
    logger.info("Counterparty EL/UL table:\n%s", df.to_string(index=False))

    corr = np.full((3, 3), 0.3)
    np.fill_diagonal(corr, 1.0)

    EL_total, UL_total, EC_total = aggregate_credit_losses(df["EL"], df["UL"], corr)
    print("\n=== Portfolio Credit Capital Summary ===")
    print(f"Expected Loss (EL): {EL_total:,.2f}")
    print(f"Unexpected Loss (UL): {UL_total:,.2f}")
    print(f"Economic Capital (EC): {EC_total:,.2f}")

    logger.info("Summary (head):\n%s", summary.head().to_string(index=False))
    print(summary.head())


if __name__ == "__main__":
    main()
