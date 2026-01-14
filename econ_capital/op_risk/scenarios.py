"""
Scenario generation utilities for Operational Risk LDA engine

Provides:
- discover_uoms: discover unit-of-measure (UoM) keys from frequency & severity data
- build_base_profile: base frequency and severity summaries per UoM
- generate_scenarios: stochastic and deterministic scenario generation
- apply_scenario_to_config: create a config-like structure suitable for lda_run_engine
- export_scenarios: small helper to save scenarios (csv/yaml)

Design goals:
- lightweight, testable, no heavy external deps beyond numpy/pandas
- deterministic when seed provided
- compatible with existing data loaders (load_frequency_data/load_severity_data)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional, Tuple
import logging
import yaml
import numpy as np
import pandas as pd
from copy import deepcopy

from econ_capital.op_risk.data_loaders import load_frequency_data, load_severity_data

logger = logging.getLogger(__name__)


@dataclass
class Scenario:
    """Single scenario describing frequency and severity shifts per UoM."""

    name: str
    freq_multiplier: Dict[str, float]  # multiply fitted lambda by this
    sev_mu_shift: Dict[str, float]  # additive shift to lognormal mu
    sev_scale_multiplier: Dict[str, float]  # multiply severity scale/beta by this
    note: Optional[str] = None


@dataclass
class ScenarioSet:
    """Collection of scenarios with basic metadata."""

    base_profile: Dict[
        str, Dict[str, float]
    ]  # e.g., {"UoM": {"lambda": x, "lognormal_mu": y, ...}}
    scenarios: List[Scenario]


# -------------------------
# Discovery and base profile
# -------------------------


def discover_uoms(freq_df: pd.DataFrame, sev_df: pd.DataFrame) -> List[str]:
    """Return sorted intersection of UoMs present in both dataframes."""
    uoms = sorted(set(freq_df["UoM"].unique()) & set(sev_df["UoM"].unique()))
    logger.debug("Discovered UoMs: %s", uoms)
    return uoms


def build_base_profile(
    freq_df: pd.DataFrame, sev_df: pd.DataFrame, uoms: Optional[Iterable[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute simple base profile per UoM used as scenario anchor
    - frequency lambda estimate as mean count per period
    - severity: fitted lognormal mu (log(mean of positives)) and empirical scale proxy
    """
    if uoms is None:
        uoms = discover_uoms(freq_df, sev_df)
    profile: Dict[str, Dict[str, float]] = {}
    for uom in uoms:
        counts = freq_df[freq_df["UoM"] == uom]["Count"].dropna().values
        losses = sev_df[sev_df["UoM"] == uom]["Loss_Amount"].dropna().values
        pos_losses = losses[losses > 0] if len(losses) else np.array([])
        lambda_hat = float(np.mean(counts)) if len(counts) else 0.0
        mu_hat = float(np.log(np.mean(pos_losses))) if pos_losses.size else np.log(1.0)
        scale_proxy = float(np.std(pos_losses)) if pos_losses.size else 1.0
        profile[uom] = {
            "lambda": lambda_hat,
            "lognormal_mu": mu_hat,
            "sev_scale": scale_proxy,
        }
    logger.debug("Base profile built for %d UoMs", len(profile))
    return profile


# -------------------------
# Scenario generation
# -------------------------


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def deterministic_shock(
    base_profile: Dict[str, Dict[str, float]],
    freq_pct: float = 0.5,
    sev_pct: float = 0.5,
    name: str = "deterministic_shock",
) -> Scenario:
    """
    Single deterministic scenario that increases frequency by freq_pct (e.g. 0.5 -> +50%)
    and increases severity scale by sev_pct.
    """
    freq_multiplier = {uom: 1.0 + freq_pct for uom in base_profile.keys()}
    sev_mu_shift = {
        uom: np.log(1.0 + sev_pct) for uom in base_profile.keys()
    }  # additive on log scale
    sev_scale_multiplier = {uom: 1.0 + sev_pct for uom in base_profile.keys()}
    return Scenario(
        name=name,
        freq_multiplier=freq_multiplier,
        sev_mu_shift=sev_mu_shift,
        sev_scale_multiplier=sev_scale_multiplier,
        note="Deterministic uniform shock",
    )


def generate_multiplicative_scenarios(
    base_profile: Dict[str, Dict[str, float]],
    n: int = 20,
    freq_scale: Tuple[float, float] = (1.2, 3.0),
    sev_mu_scale: Tuple[float, float] = (0.5, 2.0),
    sev_scale_multiplier: Tuple[float, float] = (1.1, 2.5),
    adverse: bool = False,
    seed: Optional[int] = None,
) -> List[Scenario]:
    """
    Generate n stochastic multiplicative scenarios
    - freq_scale: (min, max) multiplier range for frequency
    - sev_mu_scale: (min, max) additive range for log(mu)
    - sev_scale_multiplier: (min, max) multiplier range for severity scale/beta
    """
    rng = _rng(seed)
    if adverse:
        freq_scale = (1.5, 4.0)  # Stronger for adverse
        sev_mu_scale = (0.8, 3.0)
        sev_scale_multiplier = (1.5, 3.5)
    else:
        freq_scale = (1.2, 3.0)  # Milder base
        sev_mu_scale = (0.5, 2.0)
        sev_scale_multiplier = (1.2, 2.5)
    uoms = list(base_profile.keys())
    scenarios: List[Scenario] = []
    for k in range(n):
        freq_mult = {
            uom: float(rng.uniform(freq_scale[0], freq_scale[1])) for uom in uoms
        }
        sev_mu_shift = {
            uom: float(rng.uniform(sev_mu_scale[0], sev_mu_scale[1])) for uom in uoms
        }
        sev_scale_mult = {
            uom: float(rng.uniform(sev_scale_multiplier[0], sev_scale_multiplier[1]))
            for uom in uoms
        }
        scenarios.append(
            Scenario(
                name=f"rand_{k + 1}",
                freq_multiplier=freq_mult,
                sev_mu_shift=sev_mu_shift,
                sev_scale_multiplier=sev_scale_mult,
            )
        )
    logger.debug("Generated %d multiplicative scenarios", len(scenarios))
    return scenarios


def build_scenario_set_from_data(
    freq_data_path: str,
    sev_data_path: str,
    n_random: int = 10,
    seed: Optional[int] = None,
    config_dict: Optional[Dict[str, Any]] = None,
) -> ScenarioSet:
    """
    High level helper that:
    - loads frequency & severity data using existing loaders
    - builds base profile
    - incorporates user-defined scenarios from YAML config
    - adds stochastic and other pre-defined scenarios
    """
    freq_df = load_frequency_data(freq_data_path)
    sev_df = load_severity_data(sev_data_path)
    uoms = discover_uoms(freq_df, sev_df)
    base = build_base_profile(freq_df, sev_df, uoms=uoms)
    scenarios: List[Scenario] = []

    # Load scenarios from the YAML 'stress_tests' section
    if config_dict and "stress_tests" in config_dict:
        yaml_tests = config_dict["stress_tests"]
        for s_name, s_params in yaml_tests.items():
            # Get multipliers, default to 1.0 (no change) if missing
            f_mult = float(s_params.get("frequency_multiplier", 1.0))
            s_mult = float(s_params.get("severity_multiplier", 1.0))

            # Create UoM-mapped dictionaries for the Scenario object
            scenarios.append(
                Scenario(
                    name=s_name,
                    freq_multiplier={uom: f_mult for uom in uoms},
                    sev_mu_shift={uom: float(np.log(s_mult)) for uom in uoms},
                    sev_scale_multiplier={u: min(s_mult**0.6, 5.0) for u in uoms},
                    note=f"YAML-driven scenario: {s_name} (mean x{s_mult:.1f}, dispersion x min(({s_mult:.1f}*0.6), 5)",
                )
            )

    # ────────────────────────────────────────────────
    # Deterministic / named shock scenarios
    # ────────────────────────────────────────────────
    scenarios.append(
        deterministic_shock(base, freq_pct=1.0, sev_pct=1.0, name="2x Uniform Shock")
    )
    scenarios.append(
        deterministic_shock(base, freq_pct=0.2, sev_pct=1.2, name="Severe Cyber Attack")
    )
    scenarios.append(
        deterministic_shock(base, freq_pct=0.01, sev_pct=1.5, name="Pandemic")
    )

    # ────────────────────────────────────────────────
    # Stochastic (Monte Carlo style) scenarios
    # ────────────────────────────────────────────────
    if n_random > 0:
        # Normal / benign-ish direction
        scenarios.extend(
            generate_multiplicative_scenarios(
                base, n=n_random, seed=seed, adverse=False
            )
        )

        # Optional adverse scenarios
        # scenarios.extend(
        #    generate_multiplicative_scenarios(
        #        base,
        #        n=n_random // 2,
        #        seed=(seed + 1 if seed is not None else None),
        #        adverse=True,
        #    )
        # )

    return ScenarioSet(base_profile=base, scenarios=scenarios)


# -------------------------
# Utilities to apply & export
# -------------------------


def apply_scenario_to_config(
    base_config: Dict[str, Any], scenario: Scenario
) -> Dict[str, Any]:
    """
    Convert a Scenario into a config-like dict that the LDA engine can consume.
    Uses a deep copy to avoid mutating the shared base_config and stores full
    per-UoM overrides under 'uom_overrides' so the engine can apply them per UoM.
    """
    # Make a full independent copy of the config so we don't break the original
    cfg = deepcopy(base_config)

    # Ensure required sections exist
    cfg.setdefault("uom_overrides", {})

    # Store the full dictionaries
    cfg["uom_overrides"]["freq_multiplier"] = scenario.freq_multiplier
    cfg["uom_overrides"]["sev_mu_shift"] = scenario.sev_mu_shift
    cfg["uom_overrides"]["sev_scale_multiplier"] = scenario.sev_scale_multiplier

    # Store name
    cfg["scenario_name"] = scenario.name

    # Safe logging — only if it's a dict
    if isinstance(scenario.freq_multiplier, dict):
        logger.debug(
            "Applied scenario '%s' with per-UoM stress: %s",
            scenario.name,
            list(scenario.freq_multiplier),
        )
    else:
        logger.debug("Applied scenario '%s' with uniform stress", scenario.name)

    return cfg


def export_scenarios_to_yaml(scenario_set: ScenarioSet, path: str | Path) -> None:
    """Write ScenarioSet to a compact YAML for reproducibility."""
    out = {
        "base_profile": scenario_set.base_profile,
        "scenarios": [asdict(s) for s in scenario_set.scenarios],
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    logger.info("Exported %d scenarios to %s", len(scenario_set.scenarios), p)


# -------------------------
# Simple validation
# -------------------------


def validate_scenario_set(scenario_set: ScenarioSet) -> None:
    """Ensure multipliers and shifts have matching UoM keys as base_profile"""
    base_keys = set(scenario_set.base_profile.keys())
    for s in scenario_set.scenarios:
        if (
            set(s.freq_multiplier.keys()) != base_keys
            or set(s.sev_mu_shift.keys()) != base_keys
            or set(s.sev_scale_multiplier.keys()) != base_keys
        ):
            raise ValueError(f"Scenario '{s.name}' UoM keys mismatch with base_profile")


# -------------------------
# Example CLI helper when module executed directly
# -------------------------
def _demo_cli():
    """Quick demo that creates and exports scenarios using config/op_config.yaml if present"""
    try:
        default_freq = "econ_capital/op_risk/data/freq_data.csv"
        default_sev = "econ_capital/op_risk/data/sev_data.csv"
        ss = build_scenario_set_from_data(
            default_freq, default_sev, n_random=4, seed=123
        )
        export_scenarios_to_yaml(ss, "config/op_scenarios.yaml")
        print("Demo scenarios exported to config/op_scenarios.yaml")
    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as e:
        logger.exception("Demo scenario generation failed: %s", e)


if __name__ == "__main__":
    _demo_cli()
