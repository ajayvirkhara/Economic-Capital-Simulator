"""
Backtesting and validation for dynamic correlation models.
Validates regime detection against known historical stress periods.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime
from matplotlib import pyplot as plt


class CorrelationBacktester:
    """
    Backtest correlation model performance against historical regimes.
    """
    
    # Known historical stress periods (for validation)
    STRESS_PERIODS = [
        ("2008-09-15", "2009-03-31", "Global Financial Crisis"),
        ("2020-02-20", "2020-04-30", "COVID-19 Crash"),
        ("2022-02-24", "2022-03-31", "Russia-Ukraine War"),
    ]
    
    def __init__(self, estimator):
        """
        Parameters
        ----------
        estimator : DynamicCorrelationEstimator
            The correlation estimator to backtest
        """
        self.estimator = estimator
        self.results = {}
    
    def run_backtest(
        self,
        market_returns: pd.Series,
        credit_spreads: pd.Series,
        oprisk_losses: pd.Series,
        start_date: str = "2007-01-01",
    ) -> pd.DataFrame:
        """
        Run rolling regime detection over historical data.
        
        Returns
        -------
        pd.DataFrame with columns: date, regime, vol_ratio, market_credit_corr
        """
        
        results = []
        
        # Rolling window (needs at least 60 days for regime detection)
        min_window = 60
        
        for i in range(min_window, len(market_returns)):
            window_returns = market_returns.iloc[i-min_window:i]
            window_credit = credit_spreads.iloc[i-min_window:i]
            window_oprisk = oprisk_losses.iloc[i-min_window:i]
            
            # Estimate correlation at this point in time
            corr_matrix, regime = self.estimator.estimate_correlation_matrix(
                window_returns, window_credit, window_oprisk
            )
            
            # Extract key metrics
            vol_short = window_returns.tail(20).std() * np.sqrt(252)
            vol_long = window_returns.tail(60).std() * np.sqrt(252)
            vol_ratio = vol_short / vol_long if vol_long > 0 else 1.0
            
            results.append({
                "date": market_returns.index[i],
                "regime": regime,
                "vol_ratio": vol_ratio,
                "market_credit_corr": corr_matrix[0, 1],
                "market_oprisk_corr": corr_matrix[0, 2],
                "credit_oprisk_corr": corr_matrix[1, 2],
            })
        
        df = pd.DataFrame(results)
        df.set_index("date", inplace=True)
        self.results = df
        
        return df
    
    def evaluate_regime_accuracy(self) -> Dict[str, float]:
        """
        Calculate how well regime detection matches known stress periods.
        
        Returns
        -------
        Dict with precision, recall, and F1 score
        """
        
        if self.results.empty:
            raise ValueError("Run backtest first")
        
        # Mark known stress periods in results
        self.results["actual_stress"] = False
        
        for start, end, label in self.STRESS_PERIODS:
            try:
                mask = (self.results.index >= start) & (self.results.index <= end)
                self.results.loc[mask, "actual_stress"] = True
            except Exception:
                pass  # Date not in dataset
        
        # Calculate confusion matrix
        true_positive = (
            (self.results["regime"] == "Stress") & 
            (self.results["actual_stress"] == True)
        ).sum()
        
        false_positive = (
            (self.results["regime"] == "Stress") & 
            (self.results["actual_stress"] == False)
        ).sum()
        
        false_negative = (
            (self.results["regime"] != "Stress") & 
            (self.results["actual_stress"] == True)
        ).sum()
        
        true_negative = (
            (self.results["regime"] != "Stress") & 
            (self.results["actual_stress"] == False)
        ).sum()
        
        # Metrics
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        accuracy = (true_positive + true_negative) / len(self.results)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": accuracy,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
        }
    
    def plot_regime_timeline(self, save_path: str = None):
        """
        Visualize regime detection over time with stress period overlays.
        """
        
        if self.results.empty:
            raise ValueError("Run backtest first")
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # Plot 1: Volatility ratio with regime coloring
        stress_mask = self.results["regime"] == "Stress"
        
        axes[0].plot(self.results.index, self.results["vol_ratio"], 
                     color="black", linewidth=0.8, label="Vol Ratio")
        axes[0].axhline(y=1.5, color="red", linestyle="--", 
                        label="Stress Threshold (1.5x)")
        axes[0].fill_between(self.results.index, 0, 3, 
                              where=stress_mask, alpha=0.3, 
                              color="red", label="Detected Stress")
        axes[0].set_ylabel("Vol Ratio (20d/60d)")
        axes[0].legend(loc="upper left")
        axes[0].set_title("Regime Detection Backtest")
        axes[0].grid(alpha=0.3)
        
        # Plot 2: Market-Credit Correlation
        axes[1].plot(self.results.index, self.results["market_credit_corr"], 
                     color="blue", linewidth=0.8)
        axes[1].axhline(y=0.30, color="green", linestyle="--", 
                        label="Normal (0.30)")
        axes[1].axhline(y=0.65, color="red", linestyle="--", 
                        label="Stress (0.65)")
        axes[1].fill_between(self.results.index, 0, 1, 
                              where=stress_mask, alpha=0.3, color="red")
        axes[1].set_ylabel("Market-Credit Correlation")
        axes[1].legend(loc="upper left")
        axes[1].grid(alpha=0.3)
        
        # Plot 3: Historical stress periods (ground truth)
        axes[2].fill_between(self.results.index, 0, 1, 
                              where=self.results["actual_stress"], 
                              alpha=0.5, color="orange", 
                              label="Known Stress Periods")
        axes[2].set_ylabel("Stress Label")
        axes[2].set_xlabel("Date")
        axes[2].legend(loc="upper left")
        axes[2].set_ylim(-0.1, 1.1)
        axes[2].set_yticks([0, 1])
        axes[2].set_yticklabels(["Normal", "Stress"])
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def generate_report(self) -> str:
        """
        Generate a text report of backtesting results.
        """
        
        metrics = self.evaluate_regime_accuracy()
        
        report = []
        report.append("=" * 70)
        report.append("CORRELATION MODEL BACKTEST REPORT")
        report.append("=" * 70)
        report.append("")
        report.append(f"Method: {self.estimator.method}")
        report.append(f"Stress Multiplier: {self.estimator.stress_multiplier}x")
        report.append(f"Total Observations: {len(self.results):,}")
        report.append("")
        report.append("-" * 70)
        report.append("REGIME DETECTION ACCURACY")
        report.append("-" * 70)
        report.append(f"Precision:  {metrics['precision']:.2%}  (of detected stress, % actually stress)")
        report.append(f"Recall:     {metrics['recall']:.2%}  (of actual stress, % detected)")
        report.append(f"F1 Score:   {metrics['f1_score']:.2%}  (harmonic mean)")
        report.append(f"Accuracy:   {metrics['accuracy']:.2%}  (overall correctness)")
        report.append("")
        report.append("Confusion Matrix:")
        report.append(f"  True Positives:  {metrics['true_positive']:>6,}")
        report.append(f"  False Positives: {metrics['false_positive']:>6,}")
        report.append(f"  False Negatives: {metrics['false_negative']:>6,}")
        report.append(f"  True Negatives:  {metrics['true_negative']:>6,}")
        report.append("")
        report.append("-" * 70)
        report.append("KNOWN STRESS PERIODS EVALUATED")
        report.append("-" * 70)
        for start, end, label in self.STRESS_PERIODS:
            report.append(f"  • {label}: {start} to {end}")
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)
