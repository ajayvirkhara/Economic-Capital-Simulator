"""
Unit tests for econ_capital.market_risk.covariance module.
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest
import sys

from econ_capital.market_risk.covariance import (
    ewma_cov,
    sample_cov,
    garch_vols,
    garch_cov,
)


# --- Fixtures and Constants ---


@pytest.fixture
def dummy_returns():
    """Provides a consistent DataFrame for testing covariance estimators."""
    np.random.seed(42)
    data = np.random.randn(100, 3)
    # Ensure mean returns are close to zero for simplicity
    data[0:5] = 0.0  # Introduce some zero values for fillna robustness check
    df = pd.DataFrame(data, columns=["A", "B", "C"])
    return df


@pytest.fixture
def expected_sample_cov(dummy_returns):
    """Provides the pandas built-in sample covariance for comparison."""
    return dummy_returns.cov()


# --- Tests for ewma_cov ---


def test_ewma_cov_symmetry(dummy_returns):
    """Tests that the EWMA covariance matrix is symmetric, as expected for a covariance measure."""

    cov = ewma_cov(dummy_returns, lamb=0.97)

    # Check for symmetry by comparing the matrix to its transpose
    assert np.allclose(cov, cov.T), "Covariance matrix is not symmetric"


def test_ewma_cov_shape(dummy_returns):
    """Tests that the EWMA covariance matrix has the correct shape (k x k)."""
    returns = dummy_returns
    cov = ewma_cov(returns, lamb=0.94)
    expected_shape = (returns.shape[1], returns.shape[1])
    assert cov.shape == expected_shape
    assert all(cov.index == returns.columns)
    assert all(cov.columns == returns.columns)


def test_ewma_cov_handles_nas(dummy_returns):
    """Tests that the EWMA estimator handles initial NaNs by replacing them with 0.0."""
    returns = dummy_returns.copy()
    # Introduce NaNs at the start of the series (where fillna(0.0) is important)
    returns.iloc[0, 0] = np.nan

    # It should compute without error and be positive semi-definite (cov >= 0)
    cov = ewma_cov(returns, lamb=0.97)

    assert not cov.isnull().any().any(), "EWMA should not produce NaNs"
    # Check if the matrix is Positive Semi-Definite (PSD)
    eigenvalues = np.linalg.eigvalsh(cov.to_numpy())
    assert np.all(eigenvalues >= -1e-8), "EWMA matrix is not Positive Semi-Definite"


# --- Tests for sample_cov ---


def test_sample_cov_correctness(dummy_returns, expected_sample_cov):
    """Tests that sample_cov matches the standard pandas unbiased sample covariance."""
    cov = sample_cov(dummy_returns)
    assert np.allclose(cov, expected_sample_cov), (
        "Sample covariance does not match pandas.DataFrame.cov()"
    )


def test_sample_cov_shape(dummy_returns):
    """Tests that the sample covariance matrix has the correct shape (k x k)."""
    returns = dummy_returns
    cov = sample_cov(returns)
    expected_shape = (returns.shape[1], returns.shape[1])
    assert cov.shape == expected_shape


# --- Tests for garch_vols ---


# Mocking the external 'arch' library to avoid ImportError
@patch("arch.arch_model")
def test_garch_vols_correct_return(mock_arch_model, dummy_returns):
    """Tests that garch_vols returns a Series of the correct length and type,
    and verifies the GARCH volatility scaling."""

    # Setup mock GARCH result object
    mock_res = MagicMock()
    # Mock conditional_volatility to return deterministic values
    mock_res.conditional_volatility.iloc.__getitem__.side_effect = [
        pd.Series([2.0]),  # Vol for factor A (2.0%)
        pd.Series([1.0]),  # Vol for factor B (1.0%)
        pd.Series([3.0]),  # Vol for factor C (3.0%)
    ]

    # Configure arch_model mock to return the mock result object
    mock_arch_model.return_value.fit.return_value = mock_res

    # Call the function
    vols = garch_vols(dummy_returns)

    # Assertions
    assert isinstance(vols, pd.Series)
    assert len(vols) == dummy_returns.shape[1]
    assert all(vols.index == dummy_returns.columns)

    # Check the actual values after the internal 100.0 scaling/rescaling
    expected_vols = pd.Series({"A": 0.02, "B": 0.01, "C": 0.03})
    assert np.allclose(vols, expected_vols), (
        "GARCH vols did not scale/return expected values"
    )


# Mocking the external 'arch' library to avoid ImportError
@patch("arch.arch_model")
def test_garch_vols_calls_arch_model_correctly(mock_arch_model, dummy_returns):
    """Tests that GARCH is called correctly with scaled returns (returns * 100.0)."""

    # Setup the mock just to let the function run through
    mock_res = MagicMock()
    mock_res.conditional_volatility.iloc.__getitem__.return_value = pd.Series([1.0])

    # Configure arch_model mock to return the mock result object
    mock_arch_model.return_value.fit.return_value = mock_res

    garch_vols(dummy_returns)

    # Assert that arch_model was called for each column
    assert mock_arch_model.call_count == dummy_returns.shape[1]

    # Check the call arguments for the first column ('A')
    # Returns should be scaled by 100.0
    first_call_args, _ = mock_arch_model.call_args_list[0]
    called_returns = first_call_args[0]

    assert np.allclose(called_returns, dummy_returns["A"] * 100.0)
    assert mock_arch_model.call_args_list[0][1]["vol"] == "Garch"
    assert mock_arch_model.call_args_list[0][1]["p"] == 1
    assert mock_arch_model.call_args_list[0][1]["q"] == 1


def test_garch_vols_raises_import_error(dummy_returns, monkeypatch):
    """Tests that garch_vols falls back to EWMA when 'arch' is missing."""

    # Patch the import itself to raise ImportError
    def mock_import_error(*args, **kwargs):
        raise ImportError("No module named 'arch'")

    # Monkeypatch the 'from arch import arch_model' line
    monkeypatch.setattr(
        "econ_capital.market_risk.covariance.arch_model",
        None,
        raising=False,
    )
    # Make the import statement fail
    monkeypatch.setitem(sys.modules, "arch", None)  # Ensure 'arch' not in sys.modules

    # Call function
    vols = garch_vols(dummy_returns)

    # Should return EWMA proxy (Series of last std values)
    assert isinstance(vols, pd.Series)
    assert len(vols) == dummy_returns.shape[1]
    assert vols.index.tolist() == dummy_returns.columns.tolist()
    assert all(vols > 0)  # Standard deviations are positive


# --- Tests for garch_cov ---


# Patching garch_vols to control the input volatilities
@patch("econ_capital.market_risk.covariance.garch_vols")
def test_garch_cov_calculation_and_symmetry(mock_garch_vols, dummy_returns):
    """Tests the core GARCH covariance calculation: Correl x Outer(Vols, Vols),
    and checks for symmetry."""

    # Setup mock vols (0.02, 0.03, 0.04)
    vols_data = pd.Series([0.02, 0.03, 0.04], index=dummy_returns.columns)
    mock_garch_vols.return_value = vols_data

    # Calculate expected result manually: Outer(Vols, Vols) * Corr
    corr = dummy_returns.corr()
    outer_vols = np.outer(vols_data, vols_data)
    expected_cov = pd.DataFrame(
        outer_vols * corr.to_numpy(),
        index=dummy_returns.columns,
        columns=dummy_returns.columns,
    )

    # Run the function
    cov = garch_cov(dummy_returns)

    # Assertions
    assert np.allclose(cov, expected_cov), "GARCH covariance calculation is incorrect"
    assert np.allclose(cov, cov.T), "GARCH covariance matrix is not symmetric"
    assert cov.shape == (3, 3)


# Mocking the external 'arch' library to avoid ImportError
@patch("arch.arch_model")
def test_garch_cov_vs_sample_cov(mock_arch_model, dummy_returns):
    """Simple check that GARCH covariance is generally different from sample covariance,
    indicating that the GARCH vol component had an effect."""

    # 1. Configure the mock to return a known, non-sample volatility
    mock_res = MagicMock()

    # We assume 3 factors, so we set a vol of 2.0 (i.e., 2%) for each of the three calls.
    mock_res.conditional_volatility.iloc.__getitem__.side_effect = [
        pd.Series([2.0]),
        pd.Series([2.0]),
        pd.Series([2.0]),
    ]
    mock_arch_model.return_value.fit.return_value = mock_res

    # 2. Execute the functions
    cov_garch = garch_cov(dummy_returns)
    cov_sample = sample_cov(dummy_returns)

    # 3. Assert the difference, verifying that the vol adjustment logic was successfully executed.
    assert not np.allclose(cov_garch, cov_sample), (
        "GARCH Cov should differ from Sample Cov due to GARCH vol effect."
    )
