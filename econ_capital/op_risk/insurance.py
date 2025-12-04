"""
Insurance & risk-mitigation engine.
Pure functions, no I/O, no pandas.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np

# pylint: disable=too-many-positional-arguments


def apply_mitigation(
    losses: List[float],
    limit: Optional[float] = None,
    deductible: float = 0.0,
    coverage: Optional[float] = 0.0,
    agg_limit: Optional[float] = None,
    agg_deductible: float = 0.0,
    attachment: float = 0.0,
    use_numpy: bool = False,
) -> List[float]:
    """
    Apply insurance mitigation to a sequence of losses.

    The function applies per-loss terms first (attachment, deductible, limit, coverage),
    then applies aggregate terms (deductible and limit) on the total recoverable amount.
    If aggregate constraints reduce the total payout, the reduction is distributed
    pro-rata across all covered losses.

    Parameters:
        losses: raw loss amounts
        limit: per-loss limit (None → unlimited)
        deductible: per-loss deductible
        coverage: proportion of covered loss paid by insurer (e.g. 0.8 = 80% coverage)
        agg_limit: maximum total payout across all losses
        agg_deductible: aggregate deductible applied to total recoverable amount
        attachment: minimum loss size required for coverage
        use_numpy: if True, attempt vectorised path (falls back to loop if needed)

    Returns:
        List of mitigated (insured) amounts, one per input loss
    """

    # Validate input
    if any(loss < 0 for loss in losses):
        raise ValueError("Losses cannot be negative")

    if use_numpy:
        return _apply_numpy(
            np.array(losses, dtype=float),
            limit,
            deductible,
            coverage,
            agg_limit,
            agg_deductible,
            attachment,
        ).tolist()

    # Step 1: Apply per-loss terms (attachment, deductible, limit, coverage)
    per_loss_payouts = []
    for loss in losses:
        if loss <= attachment:
            per_loss_payouts.append(0.0)
            continue

        net = max(loss - deductible, 0.0)
        if limit is not None:
            net = min(net, limit)
        net *= coverage if coverage is not None else 0.0
        per_loss_payouts.append(net)

    total_potential = sum(per_loss_payouts)

    # Step 2: Apply aggregate deductible to total recoverable amount
    after_agg_deductible = max(0.0, total_potential - agg_deductible)

    # Step 3: Apply aggregate limit
    final_total_payout = min(after_agg_deductible, agg_limit or float("inf"))

    # Step 4: Scale down pro-rata if aggregate constraints reduce total payout
    if total_potential > 0 and final_total_payout < total_potential:
        scale_factor = final_total_payout / total_potential
        return [p * scale_factor for p in per_loss_payouts]
    else:
        return per_loss_payouts


def _apply_numpy(
    losses: np.ndarray,
    limit,
    deductible,
    coverage,
    agg_limit,
    agg_deductible,
    attachment,
):
    """
    Vectorised fast-path for large simulations.

    Currently delegates to the main Python implementation to ensure
    identical aggregate behavior (especially pro-rata scaling).
    """
    return apply_mitigation(
        losses.tolist(),
        limit=limit,
        deductible=deductible,
        coverage=coverage,
        agg_limit=agg_limit,
        agg_deductible=agg_deductible,
        attachment=attachment,
        use_numpy=False,
    )
