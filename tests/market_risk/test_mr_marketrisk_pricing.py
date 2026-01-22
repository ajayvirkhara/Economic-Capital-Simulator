"""
Unit tests for econ_capital.market_risk.marketrisk_pricing module.
"""

import numpy as np
import pytest

from econ_capital.market_risk.marketrisk_pricing import (
    EquityPosition,
    BondPosition,
    FXForward,
    EuropeanOption,
    InterestRateSwap,
    Portfolio,
)


# --- Tests for EquityPosition ---


def test_equity_revalue():
    """Test equity P&L calculation."""
    pos = EquityPosition(quantity=1000, current_price=100.0, underlying_factor="SPY")
    shocked_prices = np.array([105.0, 95.0])
    pnl = pos.revalue(shocked_prices)
    assert np.allclose(pnl, [5000.0, -5000.0])


# --- Tests for BondPosition ---


def test_bond_revalue():
    """Test bond duration-convexity approximation."""
    pos = BondPosition(
        notional=1_000_000,
        current_price=100.0,
        modified_duration=5.0,
        convexity=25.0,
        yield_factor="USD10Y",
        current_yield=0.03,
    )
    shocked_yields = np.array([0.035, 0.025])  # +50bps, -50bps
    pnl = pos.revalue(shocked_yields)

    assert np.allclose(pnl, [-24687.5, 25312.5], atol=1e-1)


# --- Tests for FXForward ---


def test_fxforward_revalue():
    """Test FX forward MTM."""
    pos = FXForward(
        notional=1_000_000,
        strike=1.20,
        maturity=1.0,
        fx_spot_factor="EURUSD",
        domestic_rate=0.02,
        foreign_rate=0.01,
        current_spot=1.18,
    )
    shocked_spots = np.array([1.22, 1.15])
    pnl = pos.revalue(shocked_spots)

    F_market = shocked_spots * np.exp((0.02 - 0.01) * 1.0)
    DF = np.exp(-0.02 * 1.0)
    expected = 1_000_000 * (F_market - 1.20) * DF
    assert np.allclose(pnl, expected)


# --- Tests for EuropeanOption ---


def test_european_option_revalue():
    """Test Black-Scholes revaluation."""
    pos = EuropeanOption(
        strike=100.0,
        maturity=1.0,
        option_type="call",
        volatility=0.20,
        quantity=100,
        underlying_factor="SPY",
        risk_free_rate=0.02,
    )
    pos.current_spot = 100.0
    shocked_spots = np.array([105.0, 95.0])
    pnl = pos.revalue(shocked_spots)

    # Compute expected
    current_price = pos._bs_price(100.0, 1.0)
    shocked_prices = pos._bs_price(shocked_spots, 1.0)
    expected_pnl = (shocked_prices - current_price) * 100
    assert np.allclose(pnl, expected_pnl, atol=1e-2)


def test_european_option_zero_maturity():
    """Test at maturity."""
    pos = EuropeanOption(
        strike=100.0,
        maturity=0.0,
        option_type="call",
        volatility=0.20,
        quantity=1,
        underlying_factor="SPY",
    )
    pos.current_spot = 100.0
    shocked_spots = np.array([105.0, 95.0])
    pnl = pos.revalue(shocked_spots, time_to_mat=0.0)
    assert np.allclose(pnl, [5.0, 0.0])  # Intrinsic value


def test_european_option_missing_current_spot():
    """Test error if current_spot not set."""
    pos = EuropeanOption(
        strike=100.0,
        maturity=1.0,
        option_type="call",
        volatility=0.20,
        quantity=1,
        underlying_factor="SPY",
    )
    # current_spot not set
    with pytest.raises(ValueError, match="current_spot must be set"):
        pos.revalue(np.array([105.0]))


# --- Tests for InterestRateSwap ---


def test_swap_revalue():
    """Test swap DV01 approximation."""
    pos = InterestRateSwap(
        notional=10_000_000,
        fixed_rate=0.03,
        tenor_years=5.0,
        rate_factor="USD10Y",
    )
    pos.current_rate = 0.03
    shocked_rates = np.array([0.031, 0.029])  # +10bps, -10bps
    pnl = pos.revalue(shocked_rates)

    expected = 5000 * np.array([10.0, -10.0])
    assert np.allclose(pnl, expected)


def test_swap_missing_current_rate():
    """Test error if current_rate not set."""
    pos = InterestRateSwap(
        notional=10_000_000, fixed_rate=0.03, tenor_years=5.0, rate_factor="USD10Y"
    )
    # current_rate not set
    with pytest.raises(ValueError, match="current_rate must be set"):
        pos.revalue(np.array([0.031]))


# --- Tests for Portfolio ---


def test_portfolio_revalue_all():
    """Test aggregate revaluation."""
    portfolio = Portfolio()

    # Add positions
    eq = EquityPosition(1000, 100.0, "SPY")
    bond = BondPosition(1_000_000, 100.0, 5.0, 25.0, "USD10Y", 0.03)
    portfolio.add_position(eq)
    portfolio.add_position(bond)

    # Shocks
    market_shocks = {"SPY": np.array([105.0, 95.0]), "USD10Y": np.array([0.035, 0.025])}
    total_pnl = portfolio.revalue_all(market_shocks)

    # Expected: equity pnl + bond pnl
    eq_pnl = eq.revalue(market_shocks["SPY"])
    bond_pnl = bond.revalue(market_shocks["USD10Y"])
    assert np.allclose(total_pnl, eq_pnl + bond_pnl)


def test_portfolio_missing_shock():
    """Test handling missing shocks (should skip)."""
    portfolio = Portfolio()
    eq = EquityPosition(1000, 100.0, "SPY")
    portfolio.add_position(eq)

    market_shocks = {"TLT": np.array([105.0])}  # Wrong factor
    total_pnl = portfolio.revalue_all(market_shocks)
    assert np.allclose(total_pnl, 0.0)
