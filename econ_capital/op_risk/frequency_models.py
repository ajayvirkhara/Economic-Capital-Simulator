"""
Frequency Models for Op Risk LDA
===============================

Poisson frequency fitting and simulation.

- Fits: MLE (lambda = mean of observed counts)
- Simulation: Vectorized, supports multiple paths, shape control, and random seed

Usage:
    lambda_ = fit_poisson(counts)
    freqs = simulate_frequency(lambda_, n_paths=1000, random_state=42)
"""

from __future__ import annotations
from typing import Union, Optional, Sequence
import numpy as np


def fit_poisson(counts: Union[np.ndarray, Sequence[int]]) -> float:
    """
    Fit Poisson model to observed frequency counts.

    Args:
        counts: Array or list of observed loss counts (non-negative integers)

    Returns:
        float: Fitted Poisson rate parameter (lambda)

    Raises:
        ValueError: If counts empty or contain negatives
    """
    counts = np.asarray(counts)
    if counts.size == 0 or np.any(counts < 0):
        raise ValueError("Counts must be non-empty and non-negative")
    return float(np.mean(counts))  # MLE


def simulate_frequency(
    lambda_: float,
    n_paths: int = 1,
    size: Optional[Union[int, tuple]] = None,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate Poisson frequencies.

    Args:
        lambda_: Poisson rate parameter (non-negative)
        n_paths: Number of simulation paths (ignored if `size` is provided)
        size: Shape of output array; overrides n_paths if provided
        random_state: Seed for reproducible simulation

    Returns:
        np.ndarray: Simulated frequencies
    """
    if lambda_ < 0:
        raise ValueError("Lambda must be non-negative")

    rng = np.random.default_rng(random_state)

    if size is None:
        size = (n_paths,)
    elif isinstance(size, int):
        size = (size,)

    return rng.poisson(lambda_, size=size)
