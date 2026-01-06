"""Global configuration and defaults for Credit Risk simulations."""

DEFAULT_CONFIG = {
    "n_paths": 5000,
    "horizon_steps": 6,
    "confidence_level": 0.999,
    "discount_rate_annual": 0.02,
    "recovery_rate": 0.4,
    "default_correlation": 0.5,
    "seed": 42,
    "corr": 0.2,
    "alpha_factor": 1.4,
}
