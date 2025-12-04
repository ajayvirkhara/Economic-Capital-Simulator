import pytest

from econ_capital.op_risk.insurance import apply_mitigation

# pylint: disable=redefined-outer-name


# -------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------
@pytest.fixture
def simple_losses() -> list[float]:
    """Three representative gross losses used in most tests."""
    return [100.0, 300.0, 50.0]


# -------------------------------------------------------------
# Tests
# -------------------------------------------------------------


def test_per_loss_basic(simple_losses):
    payout = apply_mitigation(
        losses=simple_losses,
        deductible=50,
        limit=200,
        coverage=1.0,
    )
    # 100 → 50  |  300 → 200 (limit)  |  50 → 0
    assert payout == pytest.approx([50.0, 200.0, 0.0])


def test_per_loss_coinsurance(simple_losses):
    payout = apply_mitigation(
        losses=simple_losses,
        deductible=0,
        limit=200,
        coverage=0.5,
    )
    # 100→50 | 300→100 | 50→25
    assert payout == pytest.approx([50.0, 100.0, 25.0])
    assert sum(payout) == pytest.approx(175.0)


def test_aggregate_deductible():
    losses = [100.0, 200.0, 300.0]
    payout = apply_mitigation(
        losses=losses,
        deductible=0,
        agg_deductible=200,
        agg_limit=None,
        coverage=1,
    )
    # Total gross 600 → aggregate deductible 200 → insurer pays 400
    assert sum(payout) == pytest.approx(400.0)


def test_aggregate_limit():
    losses = [100.0, 100.0, 100.0]
    payout = apply_mitigation(
        losses,
        agg_deductible=0,
        agg_limit=150,
        coverage=1,
    )
    assert sum(payout) == pytest.approx(150.0)
    assert payout[0] >= payout[1] >= payout[2] >= 0


def test_aggregate_deductible_and_limit():
    losses = [100.0, 200.0, 300.0]
    payout = apply_mitigation(
        losses,
        agg_deductible=150,
        agg_limit=300,
        coverage=1,
    )
    # 600 – 150 = 450 → capped by agg_limit → insurer pays 300
    assert sum(payout) == pytest.approx(300.0)


def test_apply_mitigation_full_chain():
    losses = [300.0, 200.0, 100.0]
    result = apply_mitigation(
        losses,
        deductible=50,
        limit=200,
        coverage=0.5,
        agg_deductible=100,
        agg_limit=300,
    )
    # After per-loss + coinsurance → 200 total
    # → agg ded 100 → insurer pays 100
    assert sum(result) == pytest.approx(100.0)


def test_apply_mitigation_no_terms_is_zero(simple_losses):
    result = apply_mitigation(simple_losses)  # no parameters = no coverage
    assert result == pytest.approx([0.0, 0.0, 0.0])


def test_apply_mitigation_handles_empty():
    assert apply_mitigation([]) == []
    assert sum(apply_mitigation([])) == 0.0


def test_apply_mitigation_negative_losses_raises():
    with pytest.raises(ValueError, match="(?i)negative"):
        apply_mitigation([-100.0, 200.0])


def test_per_loss_then_aggregate_mono_increasing(simple_losses):
    payout = apply_mitigation(
        losses=simple_losses,
        deductible=20,
        limit=200,
        coverage=1.0,
        agg_deductible=50,
        agg_limit=500,
    )
    assert all(x >= 0 for x in payout)
    # Each payout cannot exceed the original loss or the per-loss limit
    assert all(p <= min(gross, 200) for p, gross in zip(payout, simple_losses))


def test_coinsurance_effect():
    losses = [500.0]
    payout_full = apply_mitigation(losses, deductible=0, limit=500, coverage=1.0)
    payout_half = apply_mitigation(losses, deductible=0, limit=500, coverage=0.5)
    assert sum(payout_half) == pytest.approx(sum(payout_full) * 0.5)
