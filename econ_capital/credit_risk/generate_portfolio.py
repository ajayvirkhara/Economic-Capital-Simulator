"""
Utility to generate a realistic counterparty exposure CSV for the Credit Risk module.
Run this to create 'counterparty_exposures.csv'.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_portfolio_csv(filename="counterparty_exposures.csv", n_cptys=80, seed=42):
    current_dir = Path(__file__).resolve().parent
    data_dir = current_dir / "data"

    # Create the data directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)

    output_path = data_dir / "counterparty_exposures.csv"
    rng = np.random.default_rng(seed)

    data = []
    # Current Excel-style date for 'as_of_date' (approx)
    date_val = 46022

    for i in range(n_cptys):
        # Assign sector to differentiate Capital Allocation
        # Hedge Funds: High PD, High WWR (simulated via higher PD here)
        # Banks: Low PD, High EAD
        sector = rng.choice(
            ["Bank", "Corp", "HedgeFund"], p=[0.2, 0.5, 0.3]
        )  # Sector distribution: 20% Bank, 50% Corp, 30% HedgeFund

        cpty_name = f"CPTY_{i + 1:03d}_{sector}"

        if sector == "Bank":
            # Banks: Large Notional, Low PD
            value = rng.uniform(100_000_000, 700_000_000)
            pd_annual = rng.uniform(0.0005, 0.0020)  # 5-20 bps
        elif sector == "Corp":
            # Corps: Medium Notional, Medium PD
            value = rng.uniform(20_000_000, 120_000_000)
            pd_annual = rng.uniform(0.0050, 0.0300)  # 50-300 bps
        else:
            # HedgeFunds: Smaller Notional, High PD
            value = rng.uniform(5_000_000, 75_000_000)
            pd_annual = rng.uniform(0.0300, 0.100)  # 300-1000 bps

        # Add row
        data.append(
            {
                "counterparty": cpty_name,
                "instrument_id": f"INST_{i}_{sector}",
                "id_type": "INTERNAL",
                "as_of_date": date_val,
                "measure": "EAD",  # Key field for the engine
                "value": int(value),
                "units": "absolute",  # (no bps conversion)
                "currency": "USD",
                "pd_annual": round(pd_annual, 5),
            }
        )

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(
        f"Successfully generated {filename} with {len(df)} counterparties at: {output_path}."
    )
    print(df.groupby(df["counterparty"].str.split("_").str[-1])["value"].sum())


if __name__ == "__main__":
    generate_portfolio_csv()
