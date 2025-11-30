"""
Public API for the Operational Risk module.
"""

from .config import OpRiskConfig
from .lda_engine import lda_run_engine
from .data_loaders import load_frequency_data, load_severity_data
from .frequency_models import fit_poisson
from .severity_models import fit_lognormal_gpd, simulate_severity
from .insurance import apply_mitigation


# High-level convenience function — belongs at package level
def run_oprisk_pipeline(config_path: str = "config/op_config.yaml"):
    """
    Executes the full OpRisk LDA pipeline from a given YAML config path.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Tuple of (loss_distribution, fitted_models, metrics)
    """
    cfg = OpRiskConfig(config_path)
    cfg.validate()
    return lda_run_engine(cfg.as_dict())


__all__ = [
    "OpRiskConfig",
    "lda_run_engine",
    "load_frequency_data",
    "load_severity_data",
    "fit_poisson",
    "fit_lognormal_gpd",
    "simulate_severity",
    "apply_mitigation",
    "run_oprisk_pipeline",
]
