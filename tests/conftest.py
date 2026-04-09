import logging
import pytest
import numpy as np
from econ_capital.utils import setup_logging, set_global_seed
import econ_capital.op_risk.stress_tests as st
import econ_capital.op_risk.config as cfg
import os

os.environ.setdefault("FRED_API_KEY", "test-key")


@pytest.fixture(scope="session", autouse=True)
def _init_logging_and_seed():
    """
    Global Pytest fixture that runs once per test session.

    Purpose:
    --------
    - Sets up consistent logging for all test modules
    - Seeds all RNGs for reproducible Monte Carlo results
    """
    setup_logging()
    set_global_seed(42)
    yield  # all tests run after this line
    logging.getLogger(__name__).info("All tests completed successfully.")


# ----------------------------------------------------------------------
# Deterministic mock — returns exact, predictable capital
# ----------------------------------------------------------------------
BASE_CAPITAL = 1_000_000.0


def mock_lda_run_engine(config_dict: dict) -> dict:
    """
    Deterministic mock of the LDA engine used in all stress tests.

    Returns exact capital based only on scenario overrides:
        capital = 1_000_000 × freq_mult × exp(mu_shift) × scale_mult

    Makes tests:
    - Instantaneous
    - 100% repeatable
    - Immune to missing files or config validation
    """
    overrides = config_dict.get("uom_overrides", {})

    freq = np.prod(list(overrides.get("freq_multiplier", {}).values()) or [1.0])
    mu_shift = sum(overrides.get("sev_mu_shift", {}).values() or [0.0])
    scale = np.prod(list(overrides.get("sev_scale_multiplier", {}).values()) or [1.0])

    capital = BASE_CAPITAL * freq * np.exp(mu_shift) * scale
    return {"capital_999": capital}


st.lda_run_engine = mock_lda_run_engine
cfg.OpRiskConfig.validate = lambda self: None
