import pytest
import numpy as np
from econ_capital.op_risk.frequency_models import fit_poisson, simulate_frequency


class TestFitPoisson:
    def test_valid_counts(self):
        """Test MLE on valid non-negative counts."""
        counts = np.array([0, 1, 2, 3])
        lambda_val = fit_poisson(counts)
        assert isinstance(lambda_val, float)
        assert lambda_val == 1.5  # Mean of [0,1,2,3]

    def test_list_input(self):
        """Test with sequence input (list)."""
        counts = [5, 0, 2]
        lambda_val = fit_poisson(counts)
        assert lambda_val == 2.3333333333333335  # Mean

    def test_empty_counts_raises(self):
        """Test empty input raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            fit_poisson([])

    def test_negative_counts_raises(self):
        """Test negative values raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            fit_poisson([1, -1])

    def test_zero_mean(self):
        """Test all zeros (lambda=0)."""
        lambda_val = fit_poisson([0, 0, 0])
        assert lambda_val == 0.0


class TestSimulateFrequency:
    def test_basic_simulation(self):
        """Test default n_paths with positive lambda."""
        freqs = simulate_frequency(1.5, n_paths=5, random_state=42)
        assert isinstance(freqs, np.ndarray)
        assert len(freqs) == 5
        assert np.all(freqs >= 0)
        assert abs(np.mean(freqs) - 1.5) < 1.0

    def test_reproducibility_with_seed(self):
        """Test random_state ensures same output."""
        np.random.seed(42)
        freqs1 = simulate_frequency(1.5, n_paths=5, random_state=42)
        freqs2 = simulate_frequency(1.5, n_paths=5, random_state=42)
        assert np.array_equal(freqs1, freqs2)

    def test_lambda_zero(self):
        """Test lambda=0 yields all zeros."""
        freqs = simulate_frequency(0.0, n_paths=3, random_state=42)
        assert np.all(freqs == 0)

    def test_negative_lambda_raises(self):
        """Test negative lambda raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            simulate_frequency(-0.1, n_paths=1)

    def test_size_override(self):
        """Test size arg overrides n_paths (multi-dim)."""
        np.random.seed(42)
        freqs = simulate_frequency(2.0, n_paths=1, size=(2, 3))
        assert freqs.shape == (2, 3)
        assert np.all(freqs >= 0)

    def test_int_size(self):
        """Test single int for size."""
        freqs = simulate_frequency(1.0, size=4, random_state=42)
        assert len(freqs) == 4

    def test_n_paths_ignored_with_size(self):
        """Test n_paths ignored when size provided."""
        freqs = simulate_frequency(1.0, n_paths=10, size=5)
        assert len(freqs) == 5
