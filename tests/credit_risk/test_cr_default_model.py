"""
Unit tests for econ_capital.credit_risk.default_model
"""

import numpy as np
import pandas as pds

import pytest

from econ_capital.credit_risk.default_model import (
    compute_flat_hazard,
    CreditInputs,
    incremental_default_prob,
    ead_from_exposure,
    compute_cva,
    compute_expected_loss,
)

# pylint: disable=redefined-outer-name

# --- Fixtures and Setup ---


# Define a standard set of inputs for the main engine tests
@pytest.fixture
def data_fixture():
    """Provides a consistent time grid, exposure, and credit parameters."""
    # 4 steps, 1-year horizon
    times = np.array([0.25, 0.5, 0.75, 1.0])

    # Simple EAD profile (e.g., mean exposure)
    ead_profile = np.array([100.0, 90.0, 80.0, 70.0])

    # Pathwise exposures for EAD calculation (4 paths, 4 steps)
    pathwise_exposures = np.array(
        [
            [100, 95, 90, 85],
            [110, 100, 90, 80],
            [90, 85, 80, 75],
            [120, 115, 110, 105],
        ]
    )

    credit_params = CreditInputs(counterparty="TestCo", pd_annual=0.01, lgd=0.4)

    return times, ead_profile, pathwise_exposures, credit_params


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


# Tests effective_lgd and the requirement of LGD/Recovery.
def test_credit_inputs_lgd_and_recovery():
    # Test LGD provided
    c1 = CreditInputs(counterparty="A", pd_annual=0.01, lgd=0.3)
    assert c1.effective_lgd() == 0.3

    # Test Recovery provided (0.7 recovery -> 0.3 LGD)
    c2 = CreditInputs(counterparty="B", pd_annual=0.01, recovery=0.7)
    assert c2.effective_lgd() == pytest.approx(0.3)

    # Test default LGD = 0.6 when neither provided
    c3 = CreditInputs(counterparty="C", pd_annual=0.01)
    assert c3.effective_lgd() == 0.6

    # Test both provided (LGD takes precedence)
    c4 = CreditInputs(counterparty="D", pd_annual=0.01, lgd=0.5, recovery=0.8)
    assert c4.effective_lgd() == 0.5


# Test that pd_annual is required at instantiation
def test_credit_inputs_missing_pd_annual():
    with pytest.raises(TypeError):
        CreditInputs(counterparty="MissingPD")  # missing pd_annual


# Tests the calculation of a flat hazard curve from pd_annual
def test_credit_inputs_get_hazard_flat_pd(data_fixture):
    times, _, _, credit_params = data_fixture

    hazard_curve = credit_params.get_hazard_curve(times)
    expected_lam = -np.log(1.0 - credit_params.pd_annual)
    assert np.allclose(hazard_curve, expected_lam)
    assert hazard_curve.shape == times.shape


# Tests the numerical correctness of incremental PD calculation
def test_incremental_default_prob_correctness():
    times = np.array([0.5, 1.0])  # dt=0.5
    hazard = np.array([0.1, 0.1])  # Hazard: flat 10% (0.1)

    expected_dpd = np.array([1.0 - np.exp(-0.05), np.exp(-0.05) - np.exp(-0.10)])

    dpd = incremental_default_prob(times, hazard)

    assert np.allclose(dpd, expected_dpd, atol=1e-5)
    assert np.isclose(dpd.sum(), 1.0 - np.exp(-0.10))  # Total PD over the period
    assert dpd.shape == times.shape


# Tests the ValueError when hazard and times shapes do not match
def test_incremental_default_prob_mismatch_shape():
    times = np.array([0.5, 1.0])
    hazard_bad = np.array([0.1])
    with pytest.raises(ValueError, match="hazard and times must have same shape"):
        incremental_default_prob(times, hazard_bad)


# Tests the Expected Exposure (EE) method
def test_ead_from_exposure_ee(data_fixture):
    _, _, pathwise_exposures, _ = data_fixture

    expected_ee = pathwise_exposures.mean(axis=0)

    ead_ee = ead_from_exposure(pathwise_exposures, method="EE")
    assert np.allclose(ead_ee, expected_ee)


# Tests the Quantile method (e.g., PFE)
def test_ead_from_exposure_quantile(data_fixture):
    _, _, pathwise_exposures, _ = data_fixture
    quantile_level = 0.75

    expected_pfe = np.quantile(pathwise_exposures, quantile_level, axis=0)

    ead_pfe = ead_from_exposure(
        pathwise_exposures, method="quantile", quantile=quantile_level
    )
    assert np.allclose(ead_pfe, expected_pfe)


# Tests validation for shape, method, and quantile bounds
def test_ead_from_exposure_validation():
    # Test non-2D input
    with pytest.raises(ValueError, match="Exposure array must be 2D"):
        ead_from_exposure(np.array([1, 2, 3]))

    # Test invalid method
    with pytest.raises(ValueError, match="Invalid method"):
        ead_from_exposure(np.zeros((3, 3)), method="InvalidMethod")

    # Test invalid quantile bounds
    with pytest.raises(ValueError, match="Quantile must be between 0 and 1"):
        ead_from_exposure(np.zeros((3, 3)), method="quantile", quantile=1.5)
        ead_from_exposure(np.zeros((3, 3)), method="quantile", quantile=0.0)


# Tests CVA computation with a flat hazard and verifies output structure
def test_compute_cva_basic(data_fixture):
    times, ead_profile, _, credit_params = data_fixture

    total, profile = compute_cva(times, ead_profile, credit_params)

    assert total > 0.0

    assert np.isclose(total, profile["Loss_bucket"].sum())

    assert np.all(profile["DF"] < 1.0)

    assert isinstance(profile, pds.DataFrame)
    expected_cols = [
        "time",
        "EAD",
        "LGD",
        "hazard",
        "dPD",
        "DF",
        "Loss_bucket",
        "Loss_cum",
    ]
    assert all(col in profile.columns for col in expected_cols)


# Tests Expected Loss computation (undiscounted)
def test_compute_expected_loss_basic(data_fixture):
    times, ead_profile, _, credit_params = data_fixture

    el_total, el_profile = compute_expected_loss(times, ead_profile, credit_params)

    assert el_total > 0.0

    assert np.allclose(el_profile["DF"], 1.0)


# Tests CVA using an explicit discount curve
def test_compute_cva_with_explicit_discount(data_fixture):
    times, ead_profile, _, credit_params = data_fixture
    explicit_discount = np.array([0.98, 0.95, 0.92, 0.89])

    total, profile = compute_cva(
        times, ead_profile, credit_params, discount=explicit_discount
    )

    assert np.allclose(profile["DF"], explicit_discount)
    assert total > 0
