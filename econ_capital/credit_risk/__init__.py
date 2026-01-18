"""
Public API for the Credit Risk module.

Re-exports core classes and functions from submodules for unified access.

Exposed
-------
- Trade, NettingSet : trade & netting structures
- CSA : collateralization terms
- ExposureEngine : simulation engine
- compute_cva, compute_expected_loss : CVA/EL metrics
- compute_counterparty_risk_profiles, aggregate_credit_losses : CCR capital
- load_dummy_credit_data, load_issuer_spreads_csv, load_credit_indexes : loaders
"""

from .trade_models import Trade, NettingSet
from .csa import CSA
from .exposure_engine import ExposureEngine
from .default_model import compute_cva, compute_expected_loss, compute_flat_hazard
from .ccr_engine import compute_counterparty_risk_profiles, aggregate_credit_losses
from .data_loaders import (
    load_dummy_credit_data,
    load_issuer_spreads_csv,
    load_credit_indexes,
    CSV_SCHEMA,
)
from .allocation import marginal_contribution, allocate_ec
from .market_model import simulate_credit_factors
from .wwr import adjust_for_wwr
from .config import DEFAULT_CONFIG
from .market_model import simulate_term_structure_volatility

__all__ = [
    # Trade & exposure structure
    "Trade",
    "NettingSet",
    "CSA",
    "ExposureEngine",
    # Credit loss & CVA modelling
    "compute_cva",
    "compute_expected_loss",
    "compute_flat_hazard",
    # Counterparty credit risk
    "compute_counterparty_risk_profiles",
    "aggregate_credit_losses",
    # Capital allocation & risk adjustments
    "marginal_contribution",
    "allocate_ec",
    "simulate_credit_factors",
    "adjust_for_wwr",
    # Config
    "DEFAULT_CONFIG",
    # Data utilities
    "load_dummy_credit_data",
    "load_issuer_spreads_csv",
    "load_credit_indexes",
    "CSV_SCHEMA",
    # Term structure vol
    "simulate_term_structure_volatility",
]
