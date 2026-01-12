"""
Standalone demo for ExposureEngine using stylised GBM paths and integrated Credit Capital components.

Run with:
    python -m econ_capital.credit_risk.demo_exposure
"""

import numpy as np
import pandas as pd
from scipy.stats import t

from econ_capital.utils import setup_logging, timed_section
from econ_capital.credit_risk import (
    Trade,
    NettingSet,
    CSA,
    ExposureEngine,
    compute_counterparty_risk_profiles,
    aggregate_credit_losses,
    simulate_credit_factors,
)

logger = setup_logging(__name__)


def main():
    # ────────────────────────────────────────────────────────────────
    # Parameters
    n_paths = 5000
    n_steps = 13
    times = np.linspace(0, 1.0, n_steps)
    seed = 42

    # ────────────────────────────────────────────────────────────────
    # 1. Simulate market paths (stylized SP500-like)
    sp500_paths = _simulate_sp500_paths(
        n_paths=n_paths, times=times, s0=100.0, mu=0.0, sigma=0.25, seed=seed
    )["SP500"]  # Get the actual NumPy array

    market_paths = {"SP500": sp500_paths}

    print("Market paths shape:", market_paths["SP500"].shape)  # (n_paths, n_steps)

    # ────────────────────────────────────────────────────────────────
    # 2. Define real trades + CSA
    trades = [
        Trade(name="Vanilla IRS", factor="SP500", w=0.85, gamma=0.0),
        Trade(name="Equity Forward", factor="SP500", w=0.40, gamma=0.004),
        Trade(name="Call Spread", factor="SP500", w=0.15, gamma=0.012),
    ]

    csa = CSA(
        threshold=5_000_000,
        mta=2_000_000,
        im=3_500_000,
        vm_calls_per_day=1,  # daily margin calls
    )

    netting_set = NettingSet(counterparty="DEMO_REAL_PORTFOLIO", trades=trades, csa=csa)

    # ────────────────────────────────────────────────────────────────
    # 3. Run ExposureEngine
    with timed_section("Real Exposure Engine"):
        engine = ExposureEngine(
            netting_set=netting_set,
            market_paths=market_paths,
            times=times,
            n_paths=n_paths,
            pfe_quantile=0.975,
            alpha_factor=1.4,
        )
        exposure_paths, exposure_summary = engine.compute_exposure_profile()

    demo_ead = float(exposure_summary["EAD_final"].iloc[-1])
    logger.info(f"Computed real EAD (alpha-adjusted): £{demo_ead:,.0f}")

    # ────────────────────────────────────────────────────────────────
    # 4. Counterparty risk profiles (dummy portfolio for demo)
    dummy_data = [
        {"name": "CPTY_A", "EAD": demo_ead * 1.2, "PD": 0.01, "LGD": 0.40},
        {"name": "CPTY_B", "EAD": demo_ead * 0.8, "PD": 0.02, "LGD": 0.45},
        {"name": "CPTY_C", "EAD": demo_ead * 1.0, "PD": 0.015, "LGD": 0.50},
    ]
    df = pd.DataFrame(dummy_data)
    risk_df = compute_counterparty_risk_profiles(df.to_dict("records"))

    # ────────────────────────────────────────────────────────────────
    # 5. Path-wise WWR (strong tail effect)
    credit_factors = simulate_credit_factors(
        n_paths=n_paths, n_steps=len(risk_df), corr=0.25, seed=seed + 1
    )

    el_paths = risk_df["EL"].values[None, :] * (
        1 + 0.35 * 3.5 * np.maximum(credit_factors, 0)
    )
    ul_paths = risk_df["UL"].values[None, :] * (
        1 + 0.35 * 5.0 * np.maximum(credit_factors, 0)
    )

    portfolio_losses = el_paths.sum(axis=1) + 3.09 * ul_paths.sum(axis=1) * 0.7
    EC_wwr_pathwise = np.quantile(portfolio_losses, 0.999)

    base_ec = risk_df["EL"].sum() + t.ppf(0.999, 3) * risk_df["UL"].sum() * 0.7

    print(f"\nBase EC (no WWR):     £{base_ec:,.0f}")
    print(f"WWR-adjusted EC:      £{EC_wwr_pathwise:,.0f}")
    print(f"Increase:             {EC_wwr_pathwise / base_ec - 1:.1%}")

    # For aggregation compatibility
    risk_df["EL_WWR"] = el_paths.mean(axis=0)
    risk_df["UL_WWR"] = ul_paths.mean(axis=0)

    # ────────────────────────────────────────────────────────────────
    # 6. Final portfolio aggregation
    corr = np.full((len(risk_df), len(risk_df)), 0.3)
    np.fill_diagonal(corr, 1.0)

    EL_total, UL_total, EC_total, alloc = aggregate_credit_losses(
        el=risk_df["EL_WWR"].values, ul=risk_df["UL_WWR"].values, corr=corr
    )

    print("\n=== Final Portfolio Credit Capital (WWR-adjusted) ===")
    print(f"Expected Loss:     £{EL_total:,.0f}")
    print(f"Unexpected Loss:   £{UL_total:,.0f}")
    print(f"Economic Capital:  £{EC_total:,.0f}")
    print("Allocation:", alloc)


def _simulate_sp500_paths(n_paths, times, s0=100.0, mu=0.0, sigma=0.25, seed=42):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, len(times)))
    S = np.empty_like(z)
    S[:, 0] = s0
    dt = np.diff(np.concatenate([[0.0], times]))
    for k in range(1, len(times)):
        S[:, k] = S[:, k - 1] * np.exp(
            (mu - 0.5 * sigma**2) * dt[k] + sigma * np.sqrt(dt[k]) * z[:, k]
        )
    return {"SP500": S}


if __name__ == "__main__":
    setup_logging(level="INFO")
    main()
