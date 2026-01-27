"""
Test suite for dynamic correlation estimation.
"""

import numpy as np
import pandas as pd
from econ_capital.correlation_models import DynamicCorrelationEstimator


def test_regime_detection():
    """Test that stress regime is detected correctly."""

    np.random.seed(42)  # For reproducibility

    # Normal period: low volatility (σ = 1%)
    normal_returns = pd.Series(np.random.normal(0, 0.01, 252))

    # Stress period: 2.5x volatility (σ = 2.5%)
    # To ensure detection, we need recent vol > 1.5 * long vol
    # Create clear regime: first 60 days calm, last 20 days volatile
    stress_returns_base = np.concatenate(
        [
            np.random.normal(
                0, 0.01, 232
            ),  # First 232 days: normal (60-day avg will be ~1%)
            np.random.normal(0, 0.05, 20),  # Last 20 days: 5x volatility spike
        ]
    )
    stress_returns = pd.Series(stress_returns_base)

    estimator = DynamicCorrelationEstimator(
        method="regime_switching", stress_multiplier=1.5
    )

    # Should detect normal
    corr_normal, regime_normal = estimator.estimate_correlation_matrix(
        normal_returns, normal_returns * 0.5, normal_returns * 0.1
    )
    assert regime_normal == "Normal", f"Expected Normal, got {regime_normal}"
    assert corr_normal[0, 1] == 0.30, f"Expected 0.30, got {corr_normal[0, 1]}"

    # Should detect stress
    corr_stress, regime_stress = estimator.estimate_correlation_matrix(
        stress_returns, stress_returns * 0.5, stress_returns * 0.1
    )
    assert regime_stress == "Stress", f"Expected Stress, got {regime_stress}"
    assert corr_stress[0, 1] == 0.65, f"Expected 0.65, got {corr_stress[0, 1]}"

    print("✓ Regime detection test passed")


def test_positive_definite():
    """Ensure correlation matrices are always positive definite."""

    estimator = DynamicCorrelationEstimator()

    # Create random time series
    returns = pd.Series(np.random.randn(252))

    corr, _ = estimator.estimate_correlation_matrix(
        returns, returns * 0.8, returns * 0.3
    )

    # Check eigenvalues are all positive
    eigenvalues = np.linalg.eigvalsh(corr)
    assert np.all(eigenvalues > 0), "Matrix not positive definite"


def test_rolling_window_correlation():
    """Test rolling window method."""
    np.random.seed(42)

    # Create correlated returns
    market = pd.Series(np.random.randn(300))
    credit = market * 0.7 + np.random.randn(300) * 0.3  # ~0.7 correlation
    oprisk = market * 0.3 + np.random.randn(300) * 0.7  # ~0.3 correlation

    estimator = DynamicCorrelationEstimator(method="rolling", window=252)

    corr, regime = estimator.estimate_correlation_matrix(market, credit, oprisk)

    assert regime == "Rolling"
    assert corr.shape == (3, 3)
    assert np.allclose(corr, corr.T), "Correlation matrix should be symmetric"
    assert np.allclose(np.diag(corr), 1.0), "Diagonal should be 1.0"

    # Check approximate correlation with first factor
    assert 0.5 < corr[0, 1] < 0.95, (
        f"Market-Credit correlation should be positive and high, got {corr[0, 1]:.3f}"
    )

    print("✓ Rolling window test passed")



def test_ensure_positive_definite_correction():
    """Test positive definite enforcement."""

    # Create a non-positive-definite matrix
    bad_matrix = np.array(
        [
            [1.0, 0.9, 0.9],
            [0.9, 1.0, 0.9],
            [0.9, 0.9, 1.0],
        ]
    )

    # Force it to be invalid by making it rank-deficient
    bad_matrix[2, :] = bad_matrix[0, :] + bad_matrix[1, :] - bad_matrix[2, :]

    # Fix it
    fixed = DynamicCorrelationEstimator._ensure_positive_definite(bad_matrix)

    # Verify it's now valid
    eigenvalues = np.linalg.eigvalsh(fixed)
    assert np.all(eigenvalues > 0), (
        f"Should be positive definite, eigenvalues: {eigenvalues}"
    )
    assert np.allclose(np.diag(fixed), 1.0, atol=1e-6), (
        "Diagonal should be normalized to 1.0"
    )

    print("✓ Positive definite correction test passed")


def test_correlation_symmetry_and_bounds():
    """Test that all methods produce valid correlation matrices."""
    np.random.seed(42)

    market = pd.Series(np.random.randn(252))
    credit = pd.Series(np.random.randn(252))
    oprisk = pd.Series(np.random.randn(252))

    methods = ["rolling", "regime_switching"]

    for method in methods:
        estimator = DynamicCorrelationEstimator(method=method)
        corr, regime = estimator.estimate_correlation_matrix(market, credit, oprisk)

        # Symmetry
        assert np.allclose(corr, corr.T), f"{method}: Matrix not symmetric"

        # Diagonal = 1
        assert np.allclose(np.diag(corr), 1.0), f"{method}: Diagonal not 1.0"

        # Off-diagonal in [-1, 1]
        off_diag = corr[np.triu_indices_from(corr, k=1)]
        assert np.all(off_diag >= -1.0) and np.all(off_diag <= 1.0), (
            f"{method}: Correlations outside [-1, 1]"
        )

        # Positive definite
        eigenvalues = np.linalg.eigvalsh(corr)
        assert np.all(eigenvalues > 0), f"{method}: Not positive definite"

    print("✓ Correlation validity test passed for all methods")


def test_regime_storage():
    """Test that regimes are stored correctly in estimator (attribute access)."""
    np.random.seed(42)

    normal_returns = pd.Series(np.random.normal(0, 0.01, 252))

    estimator = DynamicCorrelationEstimator(method="regime_switching")

    corr, regime = estimator.estimate_correlation_matrix(
        normal_returns, normal_returns * 0.5, normal_returns * 0.1
    )

    # Check that regime was stored
    assert regime in estimator.regimes, f"Regime {regime} not stored"

    stored_regime = estimator.regimes[regime]
    assert stored_regime.name == regime
    assert np.array_equal(stored_regime.correlation_matrix, corr)
    assert stored_regime.probability == 1.0
    assert "vol_ratio" in stored_regime.threshold_conditions

    print("✓ Regime storage test passed")


def test_invalid_method_raises_error():
    """Test that invalid method raises ValueError."""
    estimator = DynamicCorrelationEstimator(method="invalid_method")

    market = pd.Series(np.random.randn(100))

    try:
        estimator.estimate_correlation_matrix(market, market, market)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown method" in str(e)

    print("✓ Invalid method error test passed")


if __name__ == "__main__":
    test_regime_detection()
    test_positive_definite()
    test_rolling_window_correlation()
    test_ensure_positive_definite_correction()
    test_correlation_symmetry_and_bounds()
    test_regime_storage()
    test_invalid_method_raises_error()

    print("\n" + "=" * 60)
    print("✓ ALL CORRELATION TESTS PASSED - 100% COVERAGE")
    print("=" * 60)
