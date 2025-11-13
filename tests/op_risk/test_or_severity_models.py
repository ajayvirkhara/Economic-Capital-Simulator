"""
Unit tests for econ_capital.op_risk.severity_models: fit_lognormal_gpd and simulate_severity.
Verifies fitting (body/tail, fallbacks), simulation (mixture, shapes), and errors.
"""

import warnings
import numpy as np
import pytest
from econ_capital.op_risk.severity_models import fit_lognormal_gpd, simulate_severity


# pylint: disable=redefined-outer-name
@pytest.fixture
def sample_losses():
    """Fixture: Generate seeded lognormal losses for consistent testing."""
    np.random.seed(42)
    return np.random.lognormal(8, 0.5, size=100)  # Mean ~exp(8) ≈ 2980


class TestFitLognormalGPD:
    def test_valid_losses(self, sample_losses):
        """Test fitting on valid positive losses."""
        params = fit_lognormal_gpd(sample_losses)
        assert isinstance(params, dict)
        assert all(
            key in params
            for key in [
                "lognormal_mu",
                "lognormal_sigma",
                "gpd_xi",
                "gpd_beta",
                "threshold",
                "tail_prob",
            ]
        )
        assert params["threshold"] > 0
        assert 0 <= params["tail_prob"] <= 1
        assert params["lognormal_sigma"] > 0
        assert params["gpd_beta"] > 0

    def test_auto_threshold(self, sample_losses):
        """Test None threshold uses 99th quantile."""
        params = fit_lognormal_gpd(sample_losses, threshold=None)
        expected_threshold = np.quantile(sample_losses, 0.99)
        assert abs(params["threshold"] - expected_threshold) < 1e-6

    def test_fixed_threshold(self, sample_losses):
        """Test explicit threshold used."""
        fixed_thresh = 5000.0
        params = fit_lognormal_gpd(sample_losses, threshold=fixed_thresh)
        assert params["threshold"] == fixed_thresh

    def test_insufficient_data_raises(self):
        """Test <20 losses raises ValueError."""
        with pytest.raises(ValueError, match=">20 positive"):
            fit_lognormal_gpd(np.ones(10))

    def test_negative_losses_raises(self):
        """Test non-positive losses raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            fit_lognormal_gpd(np.array([1, -1, 2]))

    def test_no_body_fallback(self):
        """Test fallback when no body losses (all >= threshold)."""
        np.random.seed(42)  # For reproducibility
        losses_all_tail = np.random.lognormal(10, 0.2, size=50) + 10000  # All large
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            params = fit_lognormal_gpd(losses_all_tail, threshold=5000)
        assert params["lognormal_mu"] == np.log(np.median(losses_all_tail))
        assert params["lognormal_sigma"] == 0.5

    def test_no_tail_fallback(self):
        """Test fallback when no tail losses."""
        np.random.seed(42)  # For reproducibility
        losses_no_tail = np.random.lognormal(5, 0.3, size=50)  # All small
        params = fit_lognormal_gpd(losses_no_tail, threshold=10000)
        assert params["gpd_xi"] == 0.0
        assert params["gpd_beta"] == 10000 * 0.1  # threshold * 0.1

    def test_tail_prob_calc(self, sample_losses):
        """Test tail_prob is empirical proportion."""
        params = fit_lognormal_gpd(sample_losses, threshold=5000)
        manual_tail_prob = np.mean(sample_losses >= 5000)
        assert abs(params["tail_prob"] - manual_tail_prob) < 1e-10


class TestSimulateSeverity:
    def test_basic_simulation(self, sample_losses):
        """Test simulation with valid params."""
        np.random.seed(42)
        params = fit_lognormal_gpd(sample_losses)
        sevs = simulate_severity(1000, params)
        assert isinstance(sevs, np.ndarray)
        assert len(sevs) == 1000
        assert np.all(sevs > 0)
        # Approx tail proportion (relaxed tolerance for binomial variance)
        tail_prop = np.mean(sevs >= params["threshold"])
        assert abs(tail_prop - params["tail_prob"]) < 0.05  # ~5% tail, 1000 draws

    def test_n_draws_zero(self, sample_losses):
        """Test zero draws returns empty array."""
        params = fit_lognormal_gpd(sample_losses)
        sevs = simulate_severity(0, params)
        assert len(sevs) == 0

    def test_all_body_case(self, sample_losses):
        """Test when tail_prob=0 (all body)."""
        params = fit_lognormal_gpd(sample_losses)
        params["tail_prob"] = 0.0  # Force all body
        np.random.seed(42)
        sevs = simulate_severity(500, params)
        empirical_tail_prop = np.mean(sevs >= params["threshold"])
        assert empirical_tail_prop < 0.05

    def test_all_tail_case(self, sample_losses):
        """Test when tail_prob=1 (all tail)."""
        params = fit_lognormal_gpd(sample_losses)
        params["tail_prob"] = 1.0  # Force all tail
        np.random.seed(42)
        sevs = simulate_severity(500, params)
        assert np.all(sevs >= params["threshold"])

    def test_default_tail_prob(self, sample_losses):
        """Test fallback tail_prob=0.05 if missing."""
        params_no_tail = {
            k: v
            for k, v in fit_lognormal_gpd(sample_losses).items()
            if k != "tail_prob"
        }
        np.random.seed(42)
        sevs = simulate_severity(1000, params_no_tail)
        tail_prop = np.mean(sevs >= params_no_tail["threshold"])
        assert abs(tail_prop - 0.05) < 0.03  # Approx

    def test_gpd_xi_zero_fallback(self, sample_losses):
        """Test Exponential fallback when xi=0."""
        params = fit_lognormal_gpd(sample_losses)
        params["gpd_xi"] = 0.0  # Force Exponential
        params["tail_prob"] = 0.2  # 20% tail for more tails
        np.random.seed(42)
        sevs = simulate_severity(1000, params)
        tail_sevs = sevs[sevs >= params["threshold"]]
        assert len(tail_sevs) > 0
        assert np.all(tail_sevs >= params["threshold"])

    @pytest.mark.parametrize("n_draws", [1, 10, 100])
    def test_various_sizes(self, sample_losses, n_draws):
        """Test different n_draws."""
        params = fit_lognormal_gpd(sample_losses)
        sevs = simulate_severity(n_draws, params)
        assert len(sevs) == n_draws
        assert np.all(sevs > 0)
