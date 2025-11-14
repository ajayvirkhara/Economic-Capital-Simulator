"""
Operational Risk LDA Engine
===============================

Fits models, simulates losses, computes VaR/ES.

Assumptions: UoM independence; Poisson freq; Lognormal+GPD sev.

Usage: dist, models, metrics = lda_run_engine(config)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple
import warnings
import numpy as np
import pandas as pd
from .utils import setup_logging, timed_section
from .data_loaders import load_frequency_data, load_severity_data
from .frequency_models import fit_poisson
from .severity_models import fit_lognormal_gpd, simulate_severity
from .insurance import apply_mitigation
from .config import OpRiskConfig

logger = setup_logging(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def prepare_models(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Fits frequency/severity models per UoM; handles sparse data with defaults.

    Args:
        config: Dict with 'FREQ_MODEL', 'GPD_THRESHOLD', etc.

    Returns:
        Dict of fitted params per UoM.

    Raises:
        ValueError: Data issues.
    """
    # Step 1: Log start and load data from loaders (assumes CSV/Excel or DB query)
    logger.info("Starting model preparation...")

    try:
        if not Path(config["frequency"]["data_path"]).exists():
            raise FileNotFoundError(
                f"Frequency data not found at {config["frequency"]["data_path"]}"
            )
        freq_df = load_frequency_data(
            config["frequency"]["data_path"]
        )  # Loads historical frequency counts (e.g., # losses per period)
        sev_df = load_severity_data(
            config["severity"]["data_path"]
        )  # Loads historical loss amounts
    except Exception as e:
        raise ValueError("Data loading failed.") from e

    # Step 2: Validate DataFrame structure (flexible for column variants)
    freq_count_col = "Count" if "Count" in freq_df.columns else "Frequency"
    if not {"UoM", freq_count_col}.issubset(freq_df.columns) or not {
        "UoM",
        "Loss_Amount",
    }.issubset(sev_df.columns):
        raise ValueError("Missing required columns.")

    # Step 3: Identify common UoMs across datasets and sort for consistency
    uoms = sorted(set(freq_df["UoM"].unique()) & set(sev_df["UoM"].unique()))
    if not uoms:
        raise ValueError("No common UoMs.")

    # Step 4: Initialize dict for fitted models and get model type from config
    fitted_models = {}
    freq_model_type = config.get("FREQ_MODEL", "poisson")

    # Step 5: Loop over each UoM to fit models
    for uom in uoms:
        try:
            with timed_section(f"fit_models_for_{uom}"):  # Time this fitting step
                # Filter data for this UoM
                uom_freq = freq_df[freq_df["UoM"] == uom][freq_count_col].values
                uom_sev = sev_df[sev_df["UoM"] == uom]["Loss_Amount"].values
                uom_sev = uom_sev[
                    uom_sev > 0
                ]  # Filter positives (ignore zero/negative losses)

                # Step 5a: Fit frequency model (e.g., Poisson lambda via MLE)
                if len(uom_freq) < 5:
                    logger.warning("UoM %s: Sparse freq data; default lambda=0.1", uom)
                    freq_lambda = 0.1
                else:
                    if freq_model_type == "poisson":
                        freq_lambda = fit_poisson(
                            uom_freq
                        )  # External fitter (e.g., np.mean for Poisson)
                    else:
                        raise NotImplementedError(
                            f"Model '{freq_model_type}' unsupported."
                        )

                # Step 5b: Fit severity model (hybrid Lognormal body + GPD tail)
                threshold = config.get("GPD_THRESHOLD", 100000)
                if len(uom_sev) < 10:
                    logger.warning(
                        "UoM %s: Sparse sev data; simple Lognormal defaults", uom
                    )
                    mean_loss = np.mean(uom_sev) if len(uom_sev) > 0 else 10000.0
                    sev_params = {
                        "lognormal_mu": np.log(mean_loss),
                        "lognormal_sigma": 1.0,
                        "gpd_xi": 0.0,
                        "gpd_beta": mean_loss,
                        "threshold": threshold,
                    }
                else:
                    sev_params = fit_lognormal_gpd(
                        uom_sev, threshold=threshold
                    )  # External hybrid fitter
                    sev_params["threshold"] = threshold

                # Step 5c: Compute historical Expected Loss (EL) as benchmark
                historical_el = (
                    np.mean(uom_sev) * freq_lambda if len(uom_sev) > 0 else 0.0
                )

                # Step 5d: Store fitted params for this UoM
                fitted_models[uom] = {
                    "freq_params": {
                        "lambda": freq_lambda,
                        "model_type": freq_model_type,
                    },
                    "sev_params": sev_params,
                    "historical_el": historical_el,
                }
                logger.debug("UoM %s fitted. Lambda: %.4f", uom, freq_lambda)
        except ValueError as e:
            logger.error("Fitting failed for %s: %s", uom, e)
            continue  # Skip this UoM; log but don't crash

    # Step 6: Validate at least one model fitted; log summary
    if not fitted_models:
        raise ValueError("No models fitted.")
    logger.info("Fitting complete for {len(fitted_models)} UoMs.")
    return fitted_models


def compute_capital_metrics(
    loss_distribution: np.ndarray, config: Dict[str, Any]
) -> Dict[str, float]:
    """
    Computes VaR/ES/TVaR from sorted loss dist.

    Args:
        loss_distribution: Simulated losses.
        config: With 'VAR_LEVELS', 'ES_ALPHA'.

    Returns:
        Dict of metrics.
    """
    # Step 1: Validate input
    if len(loss_distribution) == 0:
        raise ValueError("Empty distribution.")

    # Step 2: Get config levels (e.g., Basel 99.9% VaR)
    levels = config.get("VAR_LEVELS", [0.95, 0.99, 0.999])
    es_alpha = config.get("ES_ALPHA", 0.995)
    sorted_losses = np.sort(loss_distribution)  # Sort once for efficiency

    # Step 3: Compute VaR and ES for each level
    metrics = {}
    for level in levels:
        var = np.percentile(sorted_losses, level * 100)  # Empirical VaR
        metrics[f"VaR_{int(level*100)}"] = var
        tail_start = int((1 - es_alpha) * len(sorted_losses))  # Tail index
        es = (
            np.mean(sorted_losses[tail_start:])
            if tail_start < len(sorted_losses)
            else var
        )
        metrics[f"ES_{int(es_alpha*100)}"] = es  # Conditional tail mean

    # Step 4: Alias TVaR to ES; add descriptives
    metrics["TVaR_99.9"] = metrics[f"ES_{int(es_alpha*100)}"]
    metrics["Mean_Loss"] = np.mean(loss_distribution)
    metrics["Std_Loss"] = np.std(loss_distribution)

    # Step 5: Log and return
    logger.info("Metrics computed.")
    return metrics


def run_monte_carlo_simulation(
    fitted_models: Dict[str, Dict[str, Any]], config: Dict[str, Any]
) -> np.ndarray:
    """
    Vectorized MC: Sim freq/sev per UoM, aggregate totals.

    Args:
        fitted_models: Per-UoM params.
        config: With 'NUM_SIMULATIONS', 'SEED', limits.

    Returns:
        Total loss dist array.
    """
    # Step 1: Validate and seed for reproducibility
    num_simulations = config.get("NUM_SIMULATIONS", 0)
    if num_simulations <= 0:
        raise ValueError("NUM_SIMULATIONS must be > 0")
    np.random.seed(config.get("SEED", 42))

    # Step 2: Initialize total loss array and extract UoMs
    total_loss_distribution = np.zeros(num_simulations)
    uoms = list(fitted_models.keys())
    if not uoms:
        raise ValueError("No models.")

    # Step 3: Get defaults from config
    threshold_default = config.get("GPD_THRESHOLD", 100000)
    insurance_limit = config.get("UOM_INSURANCE_LIMIT", 5000000)
    deductible = config.get("UOM_DEDUCTIBLE", 100000)

    # Step 4: Log start
    logger.info("Starting sim for {num_simulations:,} paths, {len(uoms)} UoMs.")

    with timed_section("monte_carlo_simulation"):
        # Step 5: Loop over UoMs (parallelizable in future)
        for uom in uoms:
            model = fitted_models[uom]
            freq_params = model["freq_params"]
            sev_params = model["sev_params"]
            threshold = sev_params.get("threshold", threshold_default)

            # Step 5a: Vectorized frequency sim (Poisson for all paths)
            num_losses_per_path = np.random.poisson(
                freq_params["lambda"], size=num_simulations
            )
            if np.all(num_losses_per_path == 0):
                continue  # No contrib from this UoM

            # Step 5b: Compute total severity draws needed
            total_draws_needed = np.sum(num_losses_per_path)
            if total_draws_needed == 0:
                continue

            # Step 5c: Batch severity sim (from fitted hybrid model)
            batch_severities = simulate_severity(
                total_draws_needed, sev_params, threshold=threshold
            )

            # Step 5d: Aggregate per-path losses (loop over paths; vectorize further if needed)
            path_losses = np.zeros(num_simulations)
            idx = 0
            for i in range(num_simulations):
                n = num_losses_per_path[i]
                if n > 0:
                    path_sevs = batch_severities[idx : idx + n]  # Subset for this path
                    # Apply per-loss mitigation (insurance limit/deductible)
                    mitigated = apply_mitigation(
                        path_sevs, limit=insurance_limit, deductible=deductible
                    )
                    path_losses[i] = np.sum(mitigated)  # Sum mitigated losses
                    idx += n
            # Step 5e: Add UoM contrib to total (under independence assumption)
            total_loss_distribution += path_losses

    # Step 6: Log end
    logger.info("Simulation complete.")
    return total_loss_distribution


def lda_run_engine(
    config: Dict[str, Any] | None = None,
) -> Tuple[np.ndarray, Dict[str, Dict[str, Any]], Dict[str, float]]:
    """
    Main: Fit models, run sim, compute metrics.

    Args:
        config: Full config dict.

    Returns:
        (loss_dist, fitted_models, metrics)

    """
    # Step 0: Handle config
    if config is None:
        cfg = OpRiskConfig("config/op_config.yaml")
        cfg.validate()
        config = cfg.as_dict()

    logger.info("--- Starting LDA Engine ---")
    try:
        # Step 1: Prepare/fit models
        fitted_models = prepare_models(config)
        # Step 2: Run simulation
        loss_distribution = run_monte_carlo_simulation(fitted_models, config)
        # Step 3: Compute metrics
        capital_metrics = compute_capital_metrics(loss_distribution, config)
        # Step 4: Log end and return tuple
        logger.info("Run complete: {len(loss_distribution):,} paths.")
        return loss_distribution, fitted_models, capital_metrics
    except Exception as e:
        logger.error("Engine failed: {e}")
        raise RuntimeError(f"LDA run aborted: {e}") from e


if __name__ == "__main__":
    # Step 1: Demo setup (override config for quick test)
    # Demo: Small run for validation
    cfg_obj = OpRiskConfig("config/op_config.yaml")
    cfg_obj.update({"NUM_SIMULATIONS": 10000, "SEED": 42})
    try:
        # Step 2: Run engine
        distribution, models, engine_metrics = lda_run_engine(cfg_obj.as_dict())
        # Step 3: Print summaries
        print("\n--- Loss Dist Summary ---")
        print(pd.Series(distribution).describe(percentiles=[0.95, 0.99, 0.999]))
        print("\n--- Metrics ---")
        for k, v in sorted(engine_metrics.items()):
            print(f"{k}: {v:,.2f}")
        print("\n--- Models (First UoM) ---")
        if models:
            first = next(iter(models))
            print({first: models[first]})
    except ValueError as e:
        print(f"Test failed: {e}")
