from pathlib import Path
import yaml
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_YAML = PROJECT_ROOT / "default.yaml"


def load_global_defaults() -> Dict[str, Any]:
    if DEFAULT_YAML.exists():
        with open(DEFAULT_YAML, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return data
    return {}


GLOBAL_DEFAULTS = load_global_defaults()


def merge_with_global(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge module config on top of global defaults.
    Global 'simulation' block is used as base, then overridden.
    Global 'global.seed' is promoted to top-level 'seed' for convenience.
    """
    merged = GLOBAL_DEFAULTS.copy()

    # Deep merge simulation block
    merged_sim = merged.get("simulation", {}).copy()
    merged_sim.update(config.get("simulation", {}))
    merged["simulation"] = merged_sim

    # Merge other top-level keys (module overrides global)
    for key, value in config.items():
        if key != "simulation":
            merged[key] = value

    # Promote global.seed
    if "global" in GLOBAL_DEFAULTS and "seed" in GLOBAL_DEFAULTS["global"]:
        merged["seed"] = GLOBAL_DEFAULTS["global"]["seed"]

    return merged


def load_correlation_config() -> Dict[str, Any]:
    """Load correlation-specific configuration."""
    global_config = load_global_defaults()
    return global_config.get(
        "correlation",
        {
            "method": "static",
            "rolling_window": 252,
            "stress_multiplier": 1.5,
        },
    )
