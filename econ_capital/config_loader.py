from pathlib import Path
import yaml
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent
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
    merged = {}

    # Merge simulation section: global first, then module override
    global_sim = GLOBAL_DEFAULTS.get("simulation", {})
    module_sim = config.get("simulation", {})
    merged["simulation"] = {**global_sim, **module_sim}

    # Merge rest of global (e.g. global:) then full module config
    merged = {**GLOBAL_DEFAULTS, **merged, **config}

    # Promote global seed to top level if present
    if "global" in GLOBAL_DEFAULTS and "seed" in GLOBAL_DEFAULTS["global"]:
        merged.setdefault("seed", GLOBAL_DEFAULTS["global"]["seed"])

    return merged
