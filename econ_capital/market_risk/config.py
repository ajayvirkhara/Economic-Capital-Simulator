"""Default configuration + tickers + YAML loader for Market Risk."""

from __future__ import annotations
from typing import Any, Dict
from pathlib import Path
import yaml
from econ_capital.utils import setup_logging

logger = setup_logging(__name__)

# -------------------------------
# Simulation defaults
# -------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "n_paths": 500_000,
    "horizon_days": 10,
    "var_q": 0.999,
    "scaling_days_year": 252,
    "df_t": 7.0,
    "cov_method": "EWMA",  # "EWMA" | "SAMPLE" | "GARCH"
    "ewma_lambda": 0.97,
    "fix_mean": True,
    "seed": 42,
    "allocation_method": "Euler-ES",
}

# -------------------------------
# Default ticker mapping
# -------------------------------
DEFAULT_TICKERS: Dict[str, str] = {
    "SPY": "Equities (US)",
    "EFA": "Equities (Developed ex-US)",
    "EEM": "Equities (EM)",
    "TLT": "US Treasuries (long duration)",
    "LQD": "IG Credit ETF",
    "HYG": "HY Credit ETF",
    "GLD": "Gold",
    "USO": "Oil",
    "EURUSD=X": "EURUSD",
    "GBPUSD=X": "GBPUSD",
}


# -------------------------------
# YAML config loader
# -------------------------------
def load_market_yaml(path: str = "config/market_config.yaml") -> Dict[str, Any]:
    """Load YAML config, return {} if file missing or malformed."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        root = yaml.safe_load(f) or {}
    return root.get("market_risk", {})


def resolve_tickers(yaml_cfg: Dict[str, Any]) -> Dict[str, str]:
    """Return tickers from YAML if provided, else Python defaults."""
    if yaml_cfg and "tickers" in yaml_cfg and isinstance(yaml_cfg["tickers"], dict):
        return yaml_cfg["tickers"]
    return DEFAULT_TICKERS
