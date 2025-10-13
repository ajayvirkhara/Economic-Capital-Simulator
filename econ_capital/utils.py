"""
Utility functions for econ_capital package.

Centralized helpers for:
- Logging setup
- Profiling
- Timing of code blocks
- Random seeding for reproducibility
- Shape validation for numpy arrays
"""

import logging
import random
import time
import cProfile
import pstats
from io import StringIO
from contextlib import contextmanager
from typing import Tuple, Optional

import numpy as np


# -----------------------------------------------------------------------------
# 1. Logging Setup
# -----------------------------------------------------------------------------
def setup_logging(name: str = None, level: str = "INFO"):
    """Set up and return a logger for a given module."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level.upper())
    return logger


# -----------------------------------------------------------------------------
# 2. Lightweight Profiling Decorator
# -----------------------------------------------------------------------------
def profile_test(func):
    """
    Decorator to profile runtime performance of test or heavy functions.

    Usage
    -----
    @profile_test
    def test_vm_and_im_effects():
        ...
    """

    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        s = StringIO()
        pstats.Stats(pr, stream=s).sort_stats("cumtime").print_stats(5)
        logging.getLogger(__name__).info(
            "Profiling summary for %s:\n%s", func.__name__, s.getvalue()
        )
        return result

    return wrapper


# -----------------------------------------------------------------------------
# 3. Timed Section Context Manager
# -----------------------------------------------------------------------------
@contextmanager
def timed_section(label: str):
    """
    Context manager for timing code blocks.

    Example
    -------
    >>> with timed_section("Exposure simulation"):
    ...     engine.compute_exposure_profile()
    """
    start = time.perf_counter()
    logging.getLogger(__name__).info("⏳ %s started...", label)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logging.getLogger(__name__).info("✅ %s completed in %.3fs", label, elapsed)


# -----------------------------------------------------------------------------
# 4. Global Seeding for Reproducibility
# -----------------------------------------------------------------------------
def set_global_seed(seed: int = 42) -> None:
    """
    Set seeds for numpy and random for reproducible simulations.

    Example
    -------
    >>> set_global_seed(123)
    """
    np.random.seed(seed)
    random.seed(seed)
    logging.getLogger(__name__).info("Global random seed set to %d", seed)


# -----------------------------------------------------------------------------
# 5. Shape Validation Helper
# -----------------------------------------------------------------------------
def validate_shape(
    arr: np.ndarray, expected_shape: Tuple[Optional[int], ...], name: str = "array"
):
    """
    Ensure that an array has the expected shape. Raises ValueError if not.

    Allows `None` in `expected_shape` to act as a wildcard for a flexible dimension.

    Parameters
    ----------
    arr : np.ndarray
        Array to validate
    expected_shape : tuple
        Expected shape (e.g., (n_paths, n_steps)). Use `None` for a flexible dimension.
    name : str
        Name of the array (used for logging)

    Example
    -------
    >>> import numpy as np
    >>> S = np.zeros((5000, 6))
    >>> # Accepts any number of rows (paths), but requires 6 columns (steps)
    >>> validate_shape(S, (None, 6), name="market_paths")
    """
    # First, check the number of dimensions (rank)
    if len(arr.shape) != len(expected_shape):
        raise ValueError(
            f"{name} has {len(arr.shape)} dimensions ({arr.shape}), "
            f" but expected {len(expected_shape)} dimensions with shape {expected_shape}"
        )

    # Check each dimension, allowing None to pass
    for i, (dim_arr, dim_exp) in enumerate(zip(arr.shape, expected_shape)):
        if dim_exp is not None and dim_arr != dim_exp:
            raise ValueError(
                f"{name} shape {arr.shape} does not match expected {expected_shape}. "
                f"Mismatch in dimension {i} (expected {dim_exp}, got {dim_arr})."
            )
