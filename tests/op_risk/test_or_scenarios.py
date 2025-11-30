# Standard library imports
from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Module under test
from econ_capital.op_risk.scenarios import (
    build_base_profile,
    build_scenario_set_from_data,
    deterministic_shock,
    discover_uoms,
    generate_multiplicative_scenarios,
    validate_scenario_set,
    Scenario,
    ScenarioSet,
)


# =============================================================================
# Fixtures – small, realistic datasets used by almost every test
# =============================================================================
# pylint: disable=redefined-outer-name


@pytest.fixture
def tiny_freq_df() -> pd.DataFrame:
    """Frequency data for three UoMs. LEGAL deliberately has a zero count."""
    return pd.DataFrame(
        {
            "UoM": ["FRAUD", "FRAUD", "CYBER", "CYBER", "LEGAL"],
            "Period": ["2021", "2022", "2021", "2022", "2021"],
            "Count": [10, 12, 3, 5, 0],
        }
    )


@pytest.fixture
def tiny_sev_df() -> pd.DataFrame:
    """Severity data that matches the frequency fixture. LEGAL has only a zero."""
    return pd.DataFrame(
        {
            "UoM": ["FRAUD", "FRAUD", "CYBER", "CYBER", "LEGAL"],
            "Loss_Amount": [100_000, 250_000, 1_200_000, 800_000, 0],
        }
    )


@pytest.fixture
def empty_sev_uom_freq_df() -> pd.DataFrame:
    """Frequency for a UoM that has no positive losses."""
    return pd.DataFrame({"UoM": ["ORPHAN"], "Period": ["2021"], "Count": [7]})


@pytest.fixture
def empty_sev_uom_sev_df() -> pd.DataFrame:
    """Only zeroes in severity – forces the fallback logic in build_base_profile."""
    return pd.DataFrame(
        {
            "UoM": ["ORPHAN", "ORPHAN", "ORPHAN"],
            "Loss_Amount": [0, 0, 0],
        }
    )


# =============================================================================
# Core unit tests
# =============================================================================


def test_discover_uoms(tiny_freq_df: pd.DataFrame, tiny_sev_df: pd.DataFrame) -> None:
    # Discover_uoms must return the intersection between risk types
    assert discover_uoms(tiny_freq_df, tiny_sev_df) == ["CYBER", "FRAUD", "LEGAL"]


def test_build_base_profile(
    tiny_freq_df: pd.DataFrame, tiny_sev_df: pd.DataFrame
) -> None:
    profile = build_base_profile(tiny_freq_df, tiny_sev_df)

    # All three UoMs must appear
    assert set(profile.keys()) == {"CYBER", "FRAUD", "LEGAL"}

    # FRAUD: average count = (10 + 12) / 2 = 11
    assert profile["FRAUD"]["lambda"] == 11.0

    # Mean positive loss for FRAUD = 175 000 → log(175 000)
    assert np.isclose(profile["FRAUD"]["lognormal_mu"], np.log(175_000))

    # LEGAL has no events and a zero loss → defaults kick in
    assert profile["LEGAL"]["lambda"] == 0.0
    assert profile["LEGAL"]["lognormal_mu"] == 0.0  # log(1.0)
    assert profile["LEGAL"]["sev_scale"] == 1.0  # fallback scale


def test_build_base_profile_handles_no_positive_losses(
    empty_sev_uom_freq_df: pd.DataFrame,
    empty_sev_uom_sev_df: pd.DataFrame,
) -> None:
    # Building a dummy emerging risk
    profile = build_base_profile(empty_sev_uom_freq_df, empty_sev_uom_sev_df)
    orphan = profile["ORPHAN"]

    assert orphan["lambda"] == 7.0
    assert orphan["lognormal_mu"] == 0.0  # log(1.0) when no positive losses
    assert orphan["sev_scale"] == 1.0  # safe default


def test_deterministic_shock() -> None:
    # Simple sanity check that a 100 % shock doubles everything
    base = {"A": {"lambda": 10, "lognormal_mu": 12, "sev_scale": 2}}
    scenario = deterministic_shock(base, freq_pct=1.0, sev_pct=1.0, name="double")

    assert scenario.name == "double"
    assert scenario.freq_multiplier["A"] == 2.0
    assert scenario.sev_scale_multiplier["A"] == 2.0
    # Critical: severity shift is additive on the log scale → exp(shift) ≈ 2
    assert np.isclose(scenario.sev_mu_shift["A"], np.log(2.0))


def test_generate_multiplicative_scenarios_is_reproducible() -> None:
    base = {"U1": {"lambda": 1}, "U2": {"lambda": 2}}

    # Same seed → identical output. This is required for auditability.
    a = generate_multiplicative_scenarios(base, n=3, seed=999)
    b = generate_multiplicative_scenarios(base, n=3, seed=999)

    assert len(a) == 3
    assert a[0].freq_multiplier == b[0].freq_multiplier
    assert a[1].sev_mu_shift["U1"] == b[1].sev_mu_shift["U1"]


def test_validate_scenario_set_catches_mismatch() -> None:
    # If a scenario forgets a UoM the engine would crash deep inside the Monte-Carlo.
    base_profile = {"A": {}, "B": {}}
    bad = Scenario(
        name="bad",
        freq_multiplier={"A": 1.5},  # missing UoM B → should fail
        sev_mu_shift={"A": 0.1, "B": 0.1},
        sev_scale_multiplier={"A": 1.2, "B": 1.3},
    )
    scenario_set = ScenarioSet(base_profile=base_profile, scenarios=[bad])

    with pytest.raises(ValueError, match="UoM keys mismatch"):
        validate_scenario_set(scenario_set)


# =============================================================================
# Integration tests
# =============================================================================


def test_build_scenario_set_from_data_integration(
    tiny_freq_df: pd.DataFrame,
    tiny_sev_df: pd.DataFrame,
) -> None:
    """
    Full end-to-end test: write CSVs → load them → generate a scenario set.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        freq_path = Path(tmpdir) / "freq.csv"
        sev_path = Path(tmpdir) / "sev.csv"

        tiny_freq_df.to_csv(freq_path, index=False)
        tiny_sev_df.to_csv(sev_path, index=False)

        scenario_set = build_scenario_set_from_data(
            str(freq_path),
            str(sev_path),
            n_random=2,
            seed=42,
        )

        assert isinstance(scenario_set, ScenarioSet)
        assert len(scenario_set.scenarios) == 3  # 1 deterministic + 2 random
        assert any(s.name == "2x_uniform_shock" for s in scenario_set.scenarios)
        assert set(scenario_set.base_profile.keys()) == {"CYBER", "FRAUD", "LEGAL"}

        # Spot-check a couple of numbers we calculated manually above
        assert scenario_set.base_profile["FRAUD"]["lambda"] == 11.0
        assert np.isclose(
            scenario_set.base_profile["FRAUD"]["lognormal_mu"], np.log(175_000)
        )


def test_build_scenario_set_from_data_deterministic_only(
    tiny_freq_df: pd.DataFrame,
    tiny_sev_df: pd.DataFrame,
) -> None:
    """
    When n_random=0 we only want the hard-coded 2× uniform shock.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        freq_path = Path(tmpdir) / "freq.csv"
        sev_path = Path(tmpdir) / "sev.csv"
        tiny_freq_df.to_csv(freq_path, index=False)
        tiny_sev_df.to_csv(sev_path, index=False)

        scenario_set = build_scenario_set_from_data(
            str(freq_path), str(sev_path), n_random=0
        )

        assert len(scenario_set.scenarios) == 1
        shock = scenario_set.scenarios[0]
        assert shock.name == "2x_uniform_shock"
        assert all(v == 2.0 for v in shock.freq_multiplier.values())
        assert all(v == 2.0 for v in shock.sev_scale_multiplier.values())
