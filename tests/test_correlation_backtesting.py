"""
Test suite for correlation backtesting.
"""

import numpy as np
import pandas as pd
from econ_capital.correlation_models import DynamicCorrelationEstimator
from econ_capital.correlation_backtesting import CorrelationBacktester


def test_backtest_execution():
    """Test that backtesting runs without errors."""

    np.random.seed(42)

    # Create synthetic time series with known stress period
    dates = pd.date_range("2008-01-01", periods=500, freq="D")

    # Normal returns (first 250 days)
    normal_returns = np.random.normal(0, 0.01, 250)

    # Stress returns (next 250 days - 3x volatility)
    stress_returns = np.random.normal(0, 0.03, 250)

    combined_returns = np.concatenate([normal_returns, stress_returns])

    market = pd.Series(combined_returns, index=dates)
    credit = market * 0.7 + pd.Series(np.random.randn(500) * 0.01, index=dates)
    oprisk = pd.Series(np.random.randn(500) * 0.005, index=dates)

    estimator = DynamicCorrelationEstimator(method="regime_switching")
    backtester = CorrelationBacktester(estimator)

    # Run backtest
    results = backtester.run_backtest(market, credit, oprisk)

    assert not results.empty
    assert "regime" in results.columns
    assert "vol_ratio" in results.columns
    assert "market_credit_corr" in results.columns

    print("✓ Backtest execution test passed")


def test_regime_accuracy_calculation():
    """Test accuracy metrics calculation."""

    np.random.seed(42)

    dates = pd.date_range("2008-01-01", periods=300, freq="D")
    market = pd.Series(np.random.randn(300) * 0.01, index=dates)
    credit = pd.Series(np.random.randn(300) * 0.01, index=dates)
    oprisk = pd.Series(np.random.randn(300) * 0.005, index=dates)

    estimator = DynamicCorrelationEstimator(method="regime_switching")
    backtester = CorrelationBacktester(estimator)

    backtester.run_backtest(market, credit, oprisk)

    # Calculate accuracy
    metrics = backtester.evaluate_regime_accuracy()

    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "accuracy" in metrics

    # All metrics should be between 0 and 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1_score"] <= 1
    assert 0 <= metrics["accuracy"] <= 1

    print("✓ Accuracy calculation test passed")


def test_report_generation():
    """Test that report generation works."""

    np.random.seed(42)

    dates = pd.date_range("2008-01-01", periods=300, freq="D")
    market = pd.Series(np.random.randn(300) * 0.01, index=dates)
    credit = pd.Series(np.random.randn(300) * 0.01, index=dates)
    oprisk = pd.Series(np.random.randn(300) * 0.005, index=dates)

    estimator = DynamicCorrelationEstimator(method="regime_switching")
    backtester = CorrelationBacktester(estimator)

    backtester.run_backtest(market, credit, oprisk)

    report = backtester.generate_report()

    assert isinstance(report, str)
    assert len(report) > 100
    assert "BACKTEST REPORT" in report
    assert "Precision" in report
    assert "Recall" in report

    print("✓ Report generation test passed")


if __name__ == "__main__":
    test_backtest_execution()
    test_regime_accuracy_calculation()
    test_report_generation()

    print("\n" + "=" * 60)
    print("✓ ALL BACKTESTING TESTS PASSED")
    print("=" * 60)
