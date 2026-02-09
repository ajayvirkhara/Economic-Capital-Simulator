"""
Dynamic Correlation Estimation for Inter-Risk Dependencies
Supports: Regime-Switching and Rolling Windows
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Literal
from dataclasses import dataclass


@dataclass
class CorrelationRegime:
    """Container for regime-specific correlation parameters."""

    name: str
    correlation_matrix: np.ndarray
    probability: float
    threshold_conditions: Dict[str, float]


class DynamicCorrelationEstimator:
    """
    Estimates time-varying inter-risk correlations using multiple methods.
    """

    def __init__(
        self,
        method: Literal[
            "rolling",
            "regime_switching",
        ] = "regime_switching",
        window: int = 252,
        stress_multiplier: float = 1.5,
    ):
        self.method = method
        self.window = window
        self.stress_multiplier = stress_multiplier
        self.regimes: Dict[str, CorrelationRegime] = {}

    def estimate_correlation_matrix(
        self,
        market_returns: pd.Series,
        credit_spreads: pd.Series,
        oprisk_losses: pd.Series,
        current_regime: Optional[str] = None,
    ) -> Tuple[np.ndarray, str]:
        """
        Estimate 3x3 correlation matrix for Market-Credit-OpRisk.

        Returns
        -------
        correlation_matrix : np.ndarray (3, 3)
        regime_name : str
        """

        if self.method == "regime_switching":
            return self._regime_switching_correlation(
                market_returns, credit_spreads, oprisk_losses
            )
        elif self.method == "rolling":
            return self._rolling_window_correlation(
                market_returns, credit_spreads, oprisk_losses
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _regime_switching_correlation(
        self,
        market_returns: pd.Series,
        credit_spreads: pd.Series,
        oprisk_losses: pd.Series,
    ) -> Tuple[np.ndarray, str]:
        """
        Detect stress vs normal regime and return appropriate correlations.
        """

        # Detect current regime based on market volatility
        recent_vol = market_returns.tail(20).std() * np.sqrt(252)
        long_vol = market_returns.tail(60).std() * np.sqrt(252)

        # Stress indicator
        is_stress = recent_vol > self.stress_multiplier * long_vol

        if is_stress:
            # STRESS REGIME: Higher correlations (contagion)
            regime_name = "Stress"
            corr_matrix = np.array(
                [
                    [1.0, 0.65, 0.35],  # Market-Credit jumps to 0.65
                    [0.65, 1.0, 0.45],  # Credit-OpRisk increases
                    [0.35, 0.45, 1.0],  # Market-OpRisk increases
                ]
            )
        else:
            # NORMAL REGIME: Base correlations
            regime_name = "Normal"
            corr_matrix = np.array(
                [
                    [1.0, 0.30, 0.10],
                    [0.30, 1.0, 0.20],
                    [0.10, 0.20, 1.0],
                ]
            )

        # Store regime
        self.regimes[regime_name] = CorrelationRegime(
            name=regime_name,
            correlation_matrix=corr_matrix,
            probability=1.0,  # Deterministic regime
            threshold_conditions={"vol_ratio": recent_vol / long_vol},
        )

        return corr_matrix, regime_name

    def _rolling_window_correlation(
        self,
        market_returns: pd.Series,
        credit_spreads: pd.Series,
        oprisk_losses: pd.Series,
    ) -> Tuple[np.ndarray, str]:
        """
        Compute correlation from recent rolling window.
        """

        # Combine into DataFrame
        df = pd.DataFrame(
            {
                "Market": market_returns,
                "Credit": credit_spreads,
                "OpRisk": oprisk_losses,
            }
        ).dropna()

        # Rolling correlation on last window days
        recent_corr = df.tail(self.window).corr().values

        # Ensure positive definite
        recent_corr = self._ensure_positive_definite(recent_corr)

        return recent_corr, "Rolling"

    @staticmethod
    def _ensure_positive_definite(corr_matrix: np.ndarray) -> np.ndarray:
        """
        Force a correlation matrix to be positive definite.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)

        # Clamp negative eigenvalues to small positive
        eigenvalues = np.maximum(eigenvalues, 1e-6)

        # Reconstruct
        corr_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Re-normalize to correlations
        d = np.sqrt(np.diag(corr_matrix))
        corr_matrix = corr_matrix / np.outer(d, d)

        return corr_matrix
