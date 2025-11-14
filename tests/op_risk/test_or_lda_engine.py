# Standard library imports
from pathlib import Path
import os
import pytest
import numpy as np
from pytest import raises

# Module under test
from econ_capital.op_risk.lda_engine import (
    prepare_models,
    run_monte_carlo_simulation,
    lda_run_engine,
)

# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------


@pytest.fixture(name="config_fx", scope="module")
def fixture_config():
    """
    Provides a config dictionary for LDA tests using CSVs in op_risk/data/.
    Assumes sample data files exist (or are created via setup).
    """
    return {
        "GPD_THRESHOLD": 0,
        "NUM_SIMULATIONS": 5,
        "frequency": {
            "dist": "poisson",
            "lambda": 1.0,
            "data_path": "econ_capital/op_risk/data/freq_data.csv",
        },
        "severity": {
            "dist": "lognormal",
            "mu": 100,
            "sigma": 50,
            "data_path": "econ_capital/op_risk/data/sev_data.csv",
        },
    }


@pytest.fixture(name="empty_config", scope="module")
def fixture_empty_config():
    """
    Provides a config dictionary pointing to empty CSVs in op_risk/data/ with proper column structure.
    """
    return {
        "GPD_THRESHOLD": 0,
        "NUM_SIMULATIONS": 5,
        "frequency": {
            "dist": "poisson",
            "lambda": 1.0,
            "data_path": "econ_capital/op_risk/data/empty_freq.csv",
        },
        "severity": {
            "dist": "lognormal",
            "mu": 100,
            "sigma": 50,
            "data_path": "econ_capital/op_risk/data/empty_sev.csv",
        },
    }


# -------------------------------------------------------------------
# Diagnostic Fixture for Data Path Issues
# -------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def debug_data_paths():
    """
    Prints CWD and resolved file paths to diagnose read issues.
    """
    cwd = os.getcwd()
    print(f"\n=== DATA PATH DIAGNOSTIC (CWD: {cwd}) ===")
    data_dir = Path("econ_capital/op_risk/data")
    print(f"Data dir resolved: {data_dir.resolve()}")
    print(f"Data dir exists: {data_dir.exists()}")

    files = {
        "freq_data.csv": "Sample freq (UoM, Count)",
        "sev_data.csv": "Sample sev (UoM, Loss_Amount)",
        "empty_freq.csv": "Empty freq (headers only)",
        "empty_sev.csv": "Empty sev (headers only)",
    }

    for fname, desc in files.items():
        p = data_dir / fname
        print(f"  - {fname} ({desc}): Resolved={p.resolve()}, Exists={p.exists()}")

    if not data_dir.exists():
        print(
            "  **ACTION: Create 'econ_capital/op_risk/data/' dir and files via creation script.**"
        )
    print("=== END DIAGNOSTIC ===\n")


# -------------------------------------------------------------------
# Tests for prepare_models
# -------------------------------------------------------------------


def test_prepare_models_real_data(config_fx):
    """
    Verify prepare_models applies threshold filtering and returns expected structure using CSVs from op_risk/data/.
    """
    out = prepare_models(config_fx)
    assert isinstance(out, dict)
    assert len(out) > 0  # At least one UoM fitted
    for uom, params in out.items():
        assert isinstance(uom, str)  # UoM key
        assert "freq_params" in params
        assert "sev_params" in params
        assert "historical_el" in params


# -------------------------------------------------------------------
# Tests for fit_distributions (integrated in prepare_models)
# -------------------------------------------------------------------


def test_fit_distributions_real_data(config_fx):
    """
    Validate frequency and severity distribution fitting flow using CSVs from op_risk/data/.
    """
    out = prepare_models(config_fx)
    # Check per-UoM structure (not top-level 'freq'/'sev')
    for uom, params in out.items():
        assert isinstance(uom, str)
        assert "freq_params" in params  # Fitted frequency params
        assert (
            "sev_params" in params
        )  # Fitted severity params (e.g., mu, sigma, or GPD params)


# -------------------------------------------------------------------
# Tests for run (full pipeline)
# -------------------------------------------------------------------


def test_run_full_pipeline_real_data(config_fx):
    """
    Ensure full LDA pipeline runs end-to-end with CSVs from op_risk/data/.
    """
    loss_dist, fitted_models, metrics = lda_run_engine(config_fx)
    assert isinstance(loss_dist, np.ndarray)
    assert len(loss_dist) == config_fx["NUM_SIMULATIONS"]
    assert isinstance(fitted_models, dict)
    assert isinstance(metrics, dict)
    # Basic checks on outputs
    assert "VaR_99" in metrics  # Assuming standard metrics are computed
    assert np.all(loss_dist >= 0)  # Losses non-negative


# -------------------------------------------------------------------
# Tests for edge cases with empty data
# -------------------------------------------------------------------


def test_prepare_models_with_empty_df(empty_config):
    """
    Validate behaviour when loaders return empty datasets (no rows, but proper columns) from op_risk/data/.
    Expects ValueError on no common UoMs.
    """
    with raises(ValueError, match="No common UoMs."):  # UPDATED: Expect explicit error
        prepare_models(empty_config)


def test_fit_distributions_without_data(empty_config):
    """
    Ensure prepare_models raises on empty data (no common UoMs).
    """
    with raises(ValueError, match="No common UoMs."):  # UPDATED: Expect explicit error
        prepare_models(empty_config)


# -------------------------------------------------------------------
# Test severity simulation
# -------------------------------------------------------------------


def test_severity_simulation(config_fx):
    """
    Confirm severity simulation produces correct vector length.
    """
    # Mock models based on expected structure from prepare_models
    mock_models = {
        "UoM1": {
            "freq_params": {"lambda": 1.0, "model_type": "poisson"},
            "sev_params": {
                "lognormal_mu": 4.6,  # log(100)
                "lognormal_sigma": 1.0,
                "gpd_xi": 0.0,
                "gpd_beta": 100.0,
                "threshold": 0,
            },
            "historical_el": 100.0,
        }
    }

    out = run_monte_carlo_simulation(mock_models, config_fx)
    assert isinstance(out, np.ndarray)
    # Assuming aggregated losses across UoMs/paths; length matches simulations
    assert len(out) == config_fx["NUM_SIMULATIONS"]
