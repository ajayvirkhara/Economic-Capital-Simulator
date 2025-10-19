"""
Unit tests for econ_capital.credit_risk.default_model
"""

import numpy as np

from econ_capital.credit_risk import compute_flat_hazard


# Tests that the computed flat hazard rate is positive and constant across all time steps.
def test_flat_hazard_positive():
    times = np.linspace(0, 1, 5)
    lam = compute_flat_hazard(times, 0.02)
    assert np.all(lam > 0)
    assert np.allclose(lam, lam[0])


# Tests that the hazard rate respects internal clipping bounds (e.g., non-negative) even with extreme input PDs.
def test_flat_hazard_clip_bounds():
    times = np.linspace(0, 1, 3)
    lam_high = compute_flat_hazard(times, 2.0)
    lam_low = compute_flat_hazard(times, -1.0)
    assert np.all(lam_high > 0)
    assert np.all(lam_low >= 0)


# Tests that the computed flat hazard rate is numerically correct based on the input probability of default (PD).
def test_flat_hazard_reasonable_value():
    times = np.linspace(0, 1, 4)
    pd = 0.1
    lam = compute_flat_hazard(times, pd)
    expected = -np.log(1 - pd)
    assert np.isclose(lam[0], expected, atol=1e-6)


# Tests that the output hazard rate array has the same dimension/shape as the input time array.
def test_shape_matches_times():
    times = np.linspace(0, 2, 10)
    lam = compute_flat_hazard(times, 0.05)
    assert lam.shape == times.shape
