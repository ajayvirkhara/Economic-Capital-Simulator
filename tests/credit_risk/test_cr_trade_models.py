"""
Unit tests for econ_capital.credit_risk.trade_models
"""

import numpy as np
import pytest

from econ_capital.credit_risk.trade_models import (
    VanillaSwap,
    EuropeanOption,
)


# Test VanillaSwap dataclass and basic attributes
def test_vanilla_swap_initialization():
    swap = VanillaSwap(
        name="TestSwap",
        factor="USD10Y",
        w=1.0,
        notional=1e6,
        fixed_rate=0.03,
        tenor_years=5,
    )
    assert swap.name == "TestSwap"
    assert swap.factor == "USD10Y"
    assert swap.notional == 1e6
    assert swap.fixed_rate == 0.03
    assert swap.tenor_years == 5


# Test VanillaSwap.price method
def test_vanilla_swap_price():
    swap = VanillaSwap(
        name="TestSwap",
        factor="USD10Y",
        w=1.0,
        notional=1e6,
        fixed_rate=0.03,
        tenor_years=5,
    )

    # Mock market_data: rates at time_idx=0
    market_data = {
        "USD10Y": np.array([[0.035, 0.04], [0.032, 0.038]])
    }  # shape (2 paths, 2 steps)
    time_idx = 0

    mtm = swap.price(market_data, time_idx)
    assert mtm.shape == (2,)  # One value per path

    # Manual check: dv01 = 1e6 * 5 / 10000 = 50
    # rate_diff_bps: for path0= (0.035-0.03)*10000=50bps → mtm=50*50=2500
    # path1=(0.032-0.03)*10000=20bps → mtm=50*20=1000
    expected_mtm = np.array([25000, 10000])
    np.testing.assert_allclose(mtm, expected_mtm)


# Test error if missing factor in VanillaSwap.price
def test_vanilla_swap_price_missing_factor():
    swap = VanillaSwap(name="TestSwap", factor="USD10Y", w=1.0)
    market_data = {"EUR10Y": np.array([[0.03]])}
    with pytest.raises(ValueError, match="Missing factor USD10Y"):
        swap.price(market_data, time_idx=0)


# Test EuropeanOption dataclass and basic attributes
def test_european_option_initialization():
    option = EuropeanOption(
        name="TestOption",
        factor="SP500",
        w=1.0,
        strike=100.0,
        maturity=1.0,
        option_type="call",
        volatility=0.25,
        risk_free_rate=0.02,
    )
    assert option.name == "TestOption"
    assert option.factor == "SP500"
    assert option.strike == 100.0
    assert option.option_type == "call"


# Test EuropeanOption.price for call
def test_european_option_price_call():
    option = EuropeanOption(
        name="TestCall",
        factor="SP500",
        w=1.0,
        strike=100.0,
        maturity=1.0,
        option_type="call",
        volatility=0.25,
        risk_free_rate=0.02,
    )

    # Mock market_data: S at time_idx=0 (T=1.0 remaining)
    market_data = {"SP500": np.array([[105.0], [95.0]])}  # 2 paths, 1 step
    time_idx = 0

    prices = option.price(market_data, time_idx)
    assert prices.shape == (2,)

    expected_prices = np.array([14, 8.15]) # Approximate
    np.testing.assert_allclose(prices, expected_prices, atol=0.1)


# Test EuropeanOption.price for put
def test_european_option_price_put():
    option = EuropeanOption(
        name="TestPut",
        factor="SP500",
        w=1.0,
        strike=100.0,
        maturity=1.0,
        option_type="put",
        volatility=0.25,
        risk_free_rate=0.02,
    )

    market_data = {"SP500": np.array([[105.0], [95.0]])}
    time_idx = 0

    prices = option.price(market_data, time_idx)
    assert prices.shape == (2,)

    # Manual: put prices should be lower for S> K, higher for S< K
    assert prices[0] < prices[1]  # Put worth less OTM, more ITM


# Test time to maturity decay in EuropeanOption.price
def test_european_option_price_time_decay():
    option = EuropeanOption(
        name="TestCall",
        factor="SP500",
        w=1.0,
        strike=100.0,
        maturity=1.0,
        option_type="call",
    )

    market_data = {"SP500": np.array([[100.0, 100.0]])}  # 1 path, 2 steps (S constant)

    price_t0 = option.price(market_data, time_idx=0)  # T=1.0
    price_t1 = option.price(
        market_data, time_idx=1
    )  # T≈0.9 (assuming time_idx*0.1 decay)

    assert price_t0 > price_t1  # Time decay reduces option value
