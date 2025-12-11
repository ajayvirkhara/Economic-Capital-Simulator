# Standard library imports
from pathlib import Path
import os
import pytest
import numpy as np
import pandas as pd
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
    Provides a nested config dictionary for LDA tests using CSVs in op_risk/data/.
    Assumes sample data files exist (or are created via setup).
    """
    return {
        "frequency": {
            "data_path": "econ_capital/op_risk/data/freq_data.csv",
            "dist": "poisson",
            "lambda": 1.0,
        },
        "severity": {
            "data_path": "econ_capital/op_risk/data/sev_data.csv",
            "dist": "lognormal",
            "mu": 100,
            "sigma": 50,
            "GPD_THRESHOLD": 100_000,
        },
        "simulation": {
            "num_simulations": 5_000,
            "random_seed": 42,
        },
        "insurance": {
            "enabled": False,
        },
    }


@pytest.fixture(name="empty_config", scope="module")
def fixture_empty_config():
    """
    Provides a config dictionary pointing to empty CSVs in op_risk/data/ with proper column structure.
    """
    return {
        "frequency": {
            "data_path": "econ_capital/op_risk/data/empty_freq.csv",
            "dist": "poisson",
            "lambda": 1.0,
        },
        "severity": {
            "data_path": "econ_capital/op_risk/data/empty_sev.csv",
            "dist": "lognormal",
            "mu": 100,
            "sigma": 50,
            "GPD_THRESHOLD": 100_000,
        },
        "simulation": {
            "num_simulations": 5_000,
            "random_seed": 42,
        },
        "insurance": {
            "enabled": False,
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
    assert len(loss_dist) == config_fx["simulation"]["num_simulations"]
    assert isinstance(fitted_models, dict)
    assert isinstance(metrics, dict)
    # Basic checks on outputs
    assert "VaR_990" in metrics or "VaR_999" in metrics  # Per-mille naming
    assert np.all(loss_dist >= 0)  # Losses non-negative


# -------------------------------------------------------------------
# Tests for edge cases with empty data
# -------------------------------------------------------------------


def test_prepare_models_with_empty_df(empty_config):
    """
    Validate behaviour when loaders return empty datasets (no rows, but proper columns) from op_risk/data/.
    Expects ValueError on no common UoMs.
    """
    with raises(ValueError, match="No common UoMs."):
        prepare_models(empty_config)


def test_fit_distributions_without_data(empty_config):
    """
    Ensure prepare_models raises on empty data (no common UoMs).
    """
    with raises(ValueError, match="No common UoMs."):
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
    assert len(out) == config_fx["simulation"]["num_simulations"]


def test_numerical_stability_handles_nan_inf_and_clipping(config_fx, monkeypatch):
    def mock_sim(*args, **kwargs):
        return np.array([np.nan, np.inf, -1e10, 2e16, 1e15, 0])

    monkeypatch.setattr(
        "econ_capital.op_risk.lda_engine.run_monte_carlo_simulation", mock_sim
    )

    loss_dist, _, _ = lda_run_engine(config_fx)
    assert not np.any(np.isnan(loss_dist))
    assert not np.any(np.isinf(loss_dist))
    assert np.all(loss_dist >= 0)
    assert np.all(loss_dist <= 1e15)


def test_full_insurance_mitigation_path(config_fx):
    config_fx["insurance"] = {
        "enabled": True,
        "coverage": 100_000,
        "deductible": 10_000,
        "coverage_pct": 0.9,
        "agg_limit": 1_000_000,
        "agg_deductible": 50_000,
    }

    models = {
        "UoM1": {
            "freq_params": {"lambda": 10.0, "model_type": "poisson"},
            "sev_params": {
                "lognormal_mu": 11.0,  # ~60k per loss
                "lognormal_sigma": 1.0,
                "threshold": 100_000,
                "tail_prob": 0.05,
                "gpd_xi": 0.0,
                "gpd_beta": 50_000.0,
            },
        }
    }
    config_fx["simulation"]["num_simulations"] = 1000

    losses = run_monte_carlo_simulation(models, config_fx)
    # With insurance, losses should be significantly reduced
    no_ins_cfg = config_fx.copy()
    no_ins_cfg["insurance"]["enabled"] = False
    losses_no_ins = run_monte_carlo_simulation(models, no_ins_cfg)

    assert losses.mean() < losses_no_ins.mean() * 0.7  # at least 30% reduction


def test_negative_binomial_branch_is_reachable(config_fx, monkeypatch):
    config_fx["frequency"]["dist"] = "negative_binomial"

    # Make data have variance > mean to avoid fallback
    mock_freq_df = pd.DataFrame(
        {
            "UoM": ["UoM1"] * 10,
            "Count": [0, 0, 0, 0, 1, 2, 3, 5, 8, 15],  # mean ~3.4, var high
        }
    )
    mock_sev_df = pd.DataFrame(
        {"UoM": ["UoM1"] * 5, "Loss_Amount": [1000, 2000, 3000, 4000, 5000]}
    )

    monkeypatch.setattr(
        "econ_capital.op_risk.lda_engine.load_frequency_data", lambda x: mock_freq_df
    )
    monkeypatch.setattr(
        "econ_capital.op_risk.lda_engine.load_severity_data", lambda x: mock_sev_df
    )

    # Should raise NotImplementedError from the branch
    with pytest.raises(NotImplementedError):
        prepare_models(config_fx)
