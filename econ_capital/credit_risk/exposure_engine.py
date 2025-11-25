"""
Exposure Engine for Counterparty Credit Risk (CCR).

Responsibilities
---------------
- Revalue stylised trades under a netting set: MTM_t = Σ [ w·ΔS + 0.5·γ·(ΔS)^2 + add ].
- Apply CSA mechanics: threshold, MTA, IM, flexible VM call schedules.
- Compute EE(t), PFE_q(t), and EPE(t) exposure profiles.

Notes
-----
- This engine is *pricing-agnostic*: it uses stylised revaluation functions.
"""

import numpy as np
import pandas as pd

from econ_capital.utils import setup_logging

from .trade_models import NettingSet
from .exposure_models import _build_collateral_path, _compute_mtm

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# Exposure Engine
# ---------------------------------------------------------------------------
class ExposureEngine:
    """Computes MTM, collateral, and exposure metrics for a given netting set."""

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        netting_set: NettingSet,
        market_paths: dict[str, np.ndarray],
        times: np.ndarray,
        pfe_quantile: float = 0.975,
        alpha_factor: float = 1.4,
    ):
        self.netting_set = netting_set
        self.market_paths = market_paths
        self.times = np.asarray(times, dtype=float)
        self.pfe_quantile = pfe_quantile
        self.alpha_factor = alpha_factor

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def compute_mtm_only(self) -> np.ndarray:
        """Return mark-to-market array before collateral and exposure clipping."""
        return _compute_mtm(self.netting_set.trades, self.market_paths)

    def compute_exposure_profile(self) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Compute pathwise exposure and EE/PFE/EPE summary for the netting set.

        Returns
        -------
        exposure : np.ndarray
            Pathwise positive exposures (n_paths, n_steps)
        summary : pd.DataFrame
            Columns = ['time', 'EE', 'PFE_q', 'EPE_cum']
        """
        logger.info(
            "Running exposure profile computation for counterparty=%s",
            self.netting_set.counterparty,
        )

        mtm = _compute_mtm(self.netting_set.trades, self.market_paths)
        # --- Apply CSA variation margin effects ---
        csa = self.netting_set.csa
        calls_per_year = getattr(csa, "_calls_per_year", lambda: 1)()
        n_steps = mtm.shape[1]
        reset_interval = max(1, n_steps // calls_per_year)

        mtm_vm = mtm.copy()
        for t in range(reset_interval, n_steps, reset_interval):
            # Collateral exchange resets exposure to zero at each VM date
            mtm_vm[:, t:] -= mtm_vm[:, [t - 1]]

        # Apply Initial Margin (constant offset)
        if getattr(csa, "im", 0.0) > 0:
            mtm_vm -= csa.im

        mtm = mtm_vm
        collat = _build_collateral_path(mtm, self.times, self.netting_set.csa)
        exposure = np.maximum(mtm - collat, 0.0)

        EE = exposure.mean(axis=0)
        PFE = np.quantile(exposure, self.pfe_quantile, axis=0)

        _dt = np.diff(np.concatenate([[0.0], self.times]))
        num = np.cumsum(EE * _dt)
        den = np.maximum(np.cumsum(_dt), 1e-12)
        EPE_cum = num / den

        # --- Apply Alpha factor to the final EPE_cum value ---
        EAD_final = EPE_cum[-1] * self.alpha_factor
        # Apply EAD to all time steps for consistency/tracking
        EAD = EPE_cum * self.alpha_factor

        _summary = pd.DataFrame(
            {
                "time": self.times,
                "EE": EE,
                f"PFE_{int(100 * self.pfe_quantile)}": PFE,
                "EPE_cum": EPE_cum,
                "EAD": EAD,
                "EAD_final": EAD_final,
            }
        )

        logger.info(
            "Exposure computation done: EPE_cum_final=%.3f, EAD_final=%.3f, PFE_97.5%%_final=%.3f",
            float(_summary["EPE_cum"].iloc[-1]),
            float(EAD_final),
            float(_summary.filter(like="PFE").iloc[-1].values[0]),
        )
        return exposure, _summary
