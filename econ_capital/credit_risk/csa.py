"""
Collateral Support Annex (CSA) configuration for Credit Risk module.

Defines variation and initial margin rules for a counterparty netting set.

Fields
------
- threshold : unsecured exposure threshold (before collateral applies)
- mta        : minimum transfer amount — no call if |Δ| ≤ MTA
- im         : initial margin buffer (additive, static)
- vm_calls_per_day / vm_mode + vm_calls : frequency of VM calls
- business_days_per_year : used for mapping years → call counts
"""

from __future__ import annotations
from dataclasses import dataclass

from econ_capital.utils import setup_logging

logger = setup_logging(__name__)


@dataclass
class CSA:
    """
    Defines margining parameters under a CSA agreement.

    Parameters
    ----------
    threshold : float
        Unsecured exposure threshold before collateral is called.
    mta : float
        Minimum transfer amount for variation margin (VM).
    im : float
        Initial margin (IM), static buffer.
    vm_mode : str, optional
        VM call frequency mode: {'per_day', 'per_week', 'per_year'}.
    vm_calls : int, optional
        Number of VM calls per mode unit.
    vm_calls_per_day : int, optional
        Alternative daily call frequency.
    business_days_per_year : int, optional
        Used to scale daily calls (default 252).
    """

    threshold: float = 0.0
    mta: float = 0.0
    im: float = 0.0
    vm_mode: str | None = None
    vm_calls: int | None = None
    vm_calls_per_day: int | None = None
    business_days_per_year: int = 252

    def calls_per_year(self) -> int:
        """Determine variation margin (VM) call frequency per year."""
        # Priority 1: Use vm_mode and vm_calls
        if self.vm_mode and self.vm_calls:
            mode = self.vm_mode
            calls = max(1, int(self.vm_calls))
            if mode == "per_day":
                return self.business_days_per_year * calls
            if mode == "per_week":
                return 52 * calls
            if mode == "per_year":
                return calls
            raise ValueError(
                "CSA.vm_mode must be one of: per_day | per_week | per_year"
            )

        # Priority 2: Use vm_calls_per_day (daily frequency based on business days)
        if self.vm_calls_per_day:
            return self.business_days_per_year * max(1, int(self.vm_calls_per_day))

        # Default: Assume daily VM calls based on business days
        return 252

    def __post_init__(self):
        """Logging statistics."""
        logger.debug(
            "Initialized CSA: threshold=%.3f, mta=%.3f, im=%.3f, vm_mode=%s",
            getattr(self, "threshold", 0.0),
            getattr(self, "mta", 0.0),
            getattr(self, "im", 0.0),
            getattr(self, "vm_mode", "N/A"),
        )
