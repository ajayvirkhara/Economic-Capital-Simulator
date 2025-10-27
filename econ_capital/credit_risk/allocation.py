"""Capital allocation utilities for Credit Risk."""

from __future__ import annotations
import numpy as np


def marginal_contribution(losses: np.ndarray) -> np.ndarray:
    """
    Euler (covariance-based) allocation of portfolio economic capital.

    losses : array shape (n_entities, n_sims) or (n_sims, n_entities)
        Accepts either orientation; canonicalise to shape (n_entities, n_sims).
        If input is (n_sims, n_entities) (i.e. Monte Carlo rows are sims),
        pass losses.T into this function (or transpose inside caller).
    Returns
    -------
    alloc_frac : np.ndarray
        Fractional contributions (length n_entities) summing to 1.
    """
    # Expect (n_entities, n_sims). If caller passed (n_sims, n_entities) detect and transpose
    arr = np.asarray(losses)
    if arr.ndim != 2:
        raise ValueError("losses must be a 2D array")
    if arr.shape[0] > arr.shape[1]:
        # Heuristic: probably shape (n_sims, n_entities) => transpose
        arr = arr.T

    # arr now (n_entities, n_sims)
    total = arr.sum(axis=0)  # sum across entities -> (n_sims,)
    port = total  # alias
    # Covariance between each entity loss and portfolio loss
    cov_matrix = np.cov(np.vstack([arr, port]))  # shape (n_entities+1, n_entities+1)
    cov_with_port = cov_matrix[:-1, -1]  # length n_entities
    var_port = cov_matrix[-1, -1]
    if var_port == 0:
        # Degenerate case: return equal splits
        alloc = np.ones(len(cov_with_port)) / len(cov_with_port)
    else:
        alloc = cov_with_port / var_port
        # Negative allocation could appear if negative covariance; clip to zero then renormalize
        alloc = np.maximum(alloc, 0.0)
        if alloc.sum() > 0:
            alloc = alloc / alloc.sum()
        else:
            alloc = np.ones_like(alloc) / len(alloc)
    return alloc


def allocate_ec(total_ec: float, losses: np.ndarray) -> np.ndarray:
    """
    Allocate a scalar portfolio EC amount to individual entities based on marginal contribution.
    Returns allocated amounts (same order as rows of losses if losses is (n_entities, n_sims)).
    """
    arr = np.asarray(losses)
    if arr.ndim != 2:
        raise ValueError("losses must be 2D")
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    fracs = marginal_contribution(arr)
    return fracs * float(total_ec)
