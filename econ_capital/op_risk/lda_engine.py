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

from econ_capital.config_loader import merge_with_global
from .utils import setup_logging, timed_section
from .data_loaders import load_frequency_data, load_severity_data
from .frequency_models import fit_poisson
from .severity_models import fit_lognormal_gpd, simulate_severity
from .insurance import apply_mitigation
from .config import OpRiskConfig

logger = setup_logging(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).parent.parent.parent


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
        package_dir = Path(__file__).parent  # op_risk/
        data_dir = package_dir / "data"
        
        freq_path = data_dir / "freq_data.csv"
        sev_path = data_dir / "sev_data.csv"

        if not freq_path.exists() or not sev_path.exists():
            raise FileNotFoundError(f"Data files not found: {freq_path}, {sev_path}")

        freq_df = load_frequency_data(str(freq_path))
        sev_df = load_severity_data(str(sev_path))

        print(f"Successfully loaded frequency data from: {freq_path}")
        print(f"Frequency DataFrame shape: {freq_df.shape}")
        print(freq_df.head())
        print(f"Successfully loaded severity data from: {sev_path}")
        print(f"Severity DataFrame shape: {sev_df.shape}")
        print(sev_df.head())

    except Exception as e:
        print(f"DATA LOADING FAILED: {e}")
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
    freq_model_type = config.get("frequency", {}).get("dist", "poisson").lower()

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
                        freq_lambda = fit_poisson(uom_freq)
                    elif freq_model_type == "negative_binomial":
                        mean = np.mean(uom_freq)
                        var = np.var(uom_freq)
                        if var > mean + 1e-8:  # avoid division by zero
                            p = mean / var
                            r = mean * p / (1 - p)
                        else:
                            p = 0.5
                            r = mean / (1 - p)  # fallback
                        # Store r and p — will be used when we enable NB simulation
                        freq_params = {
                            "r": float(r),
                            "p": float(p),
                            "model_type": freq_model_type,
                        }
                        raise NotImplementedError(
                            f"Model '{freq_model_type}' unsupported."
                        )

                # Step 5b: Fit severity model (hybrid Lognormal body + GPD tail)
                threshold_default = config.get("severity", {}).get(
                    "GPD_THRESHOLD", 100000
                )
                if len(uom_sev) < 10:
                    logger.warning(
                        "UoM %s: Sparse sev data; simple Lognormal defaults", uom
                    )
                    mean_loss = np.mean(uom_sev) if len(uom_sev) > 0 else 10000.0
                    sev_params = {
                        "lognormal_mu": np.log(mean_loss),
                        "lognormal_sigma": 1.5,
                        "gpd_xi": 0.0,
                        "gpd_beta": mean_loss,
                        "threshold": threshold_default,
                    }
                else:
                    sev_params = fit_lognormal_gpd(
                        uom_sev, threshold=threshold_default
                    )  # External hybrid fitter
                    sev_params["threshold"] = threshold_default

                # Step 5c: Compute historical Expected Loss (EL) as benchmark
                historical_el = (
                    np.mean(uom_sev) * freq_lambda if len(uom_sev) > 0 else 0.0
                )

                # Step 5d: Store fitted params for this UoM
                if freq_model_type == "poisson":
                    freq_params = {
                        "lambda": float(freq_lambda),
                        "model_type": freq_model_type,
                    }
                elif freq_model_type == "negative_binomial":
                    freq_params = {
                        "r": float(r),
                        "p": float(p),
                        "model_type": freq_model_type,
                    }
                else:
                    raise ValueError(f"Unsupported frequency model: {freq_model_type}")

                fitted_models[uom] = {
                    "freq_params": freq_params,
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
    logger.info("Fitting complete for %d UoMs.", len(fitted_models))
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
    levels = config.get("simulation", {}).get("VAR_LEVELS", [0.95, 0.99, 0.999])
    es_alpha = config.get("simulation", {}).get("ES_ALPHA", 0.995)
    sorted_losses = np.sort(loss_distribution)  # Sort once for efficiency

    # Step 3: Compute VaR and ES for each level
    metrics = {}
    for level in levels:
        # level is e.g. 0.95, 0.99, 0.999
        per_mille = int(round(level * 1000))  # 0.999 -> 999
        var = np.percentile(sorted_losses, level * 100)
        metrics[f"VaR_{per_mille}"] = float(var)
    # ES: use es_alpha per-mille
    es_per_mille = int(round(es_alpha * 1000))
    tail_start = int((1 - es_alpha) * len(sorted_losses))
    es_val = float(
        np.mean(sorted_losses[tail_start:])
        if tail_start < len(sorted_losses)
        else metrics.get(f"VaR_{per_mille}", var)
    )
    metrics[f"ES_{es_per_mille}"] = es_val

    # canonical aliases expected by reporting/stress pipeline
    if f"VaR_{int(0.999 * 1000)}" in metrics:
        metrics["capital_999"] = metrics[f"VaR_{int(0.999 * 1000)}"]
    else:
        metrics["capital_999"] = metrics.get(
            next((k for k in metrics if k.startswith("VaR_")), None), np.nan
        )

    metrics["expected_loss"] = float(np.mean(loss_distribution))
    metrics["mean_loss"] = float(np.mean(loss_distribution))
    metrics["std_loss"] = float(np.std(loss_distribution))

    # Step 4: Alias TVaR to ES; add descriptives
    metrics["TVaR_99.9"] = metrics[f"ES_{int(round(es_alpha * 1000))}"]
    metrics["Mean_Loss"] = np.mean(loss_distribution)
    metrics["Std_Loss"] = np.std(loss_distribution)

    # Step 5: Log and return
    logger.info("Metrics computed.")
    return metrics


def run_monte_carlo_simulation(
    fitted_models: Dict[str, Dict[str, Any]], config: Dict[str, Any]
) -> np.ndarray:
    """
    Vectorized MC: Sim freq/sev per UoM, aggregate totals.Args:
    fitted_models: Per-UoM params.
    config: With 'num_simulations', 'SEED', limits.

    Returns:
    Total loss dist array.
    """
    # Step 1: Validate and seed for reproducibility
    # Use global default_n_paths if present, otherwise fallback to module-specific or hard default
    sim_cfg = config.get("simulation", {})
    num_simulations = sim_cfg.get(
        "default_n_paths", sim_cfg.get("num_simulations", 250_000)
    )
    if num_simulations <= 0:
        raise ValueError("num_simulations must be > 0")

    # Use top-level seed from merged config (from default.yaml or module override)
    seed = config.get("seed", 42)
    rng = np.random.default_rng(seed)  # Modern, reproducible RNG

    # Step 2: Initialize total loss array and extract UoMs
    total_loss_distribution = np.zeros(num_simulations)
    uoms = list(fitted_models.keys())
    if not uoms:
        raise ValueError("No models.")

    # Step 3: Get defaults from config

    # Respect insurance toggle + stress parameters (for base/default)
    insurance_enabled = config.get("insurance", {}).get("enabled", False)
    freq_multiplier = config.get("frequency", {}).get("multiplier", 1.0)
    sev_mu_shift = config.get("severity", {}).get("mu_shift", 0.0)
    sev_scale_mult = config.get("severity", {}).get("scale_multiplier", 1.0)
    if insurance_enabled:
        logger.info("INSURANCE: STATE IS ENABLED")
    else:
        logger.info("INSURANCE: STATE IS DISABLED")

    ins_config = config.get("insurance", {})
    logger.info(
        f"INSURANCE PARAMS: Per-Loss Limit (coverage): {ins_config.get('coverage')}, Per-Loss Deductible: {ins_config.get('deductible')}"
    )

    # Scenario UoM overrides
    overrides = config.get("uom_overrides", {})
    override_freq_mult = overrides.get("freq_multiplier", {})
    override_mu_shift = overrides.get("sev_mu_shift", {})
    override_scale_mult = overrides.get("sev_scale_multiplier", {})

    # Step 4: Log start
    logger.info(f"Starting sim for {num_simulations:,} paths, {len(uoms)} UoMs.")

    with timed_section("monte_carlo_simulation"):
        # Step 5: Loop over UoMs
        for uom in uoms:
            # Choose stressed parameters:
            # 1) scenario override takes priority
            # 2) else use global stress
            # 3) fallback is 1.0 / 0.0 / 1.0
            uom_freq_mult = float(override_freq_mult.get(uom, freq_multiplier))
            uom_mu_shift = float(override_mu_shift.get(uom, sev_mu_shift))
            uom_scale_mult = float(override_scale_mult.get(uom, sev_scale_mult))

            model = fitted_models[uom]
            freq_params = model["freq_params"]
            sev_params = model["sev_params"]

            # --- START STRESSED PARAMETER CALCULATION ---

            # Frequency stress
            base_lambda = freq_params["lambda"]
            stressed_lambda = base_lambda * uom_freq_mult

            # Severity stress: Create a DEEP COPY of the fitted parameters
            # and apply the stress to the copy.
            stressed_sev_params = sev_params.copy()

            # Ensure we are working with the correct base mu/sigma
            # using either fitted or hardcoded values
            if not config["severity"].get("use_fitted", True):
                mu = config["severity"]["mu"]
                sigma = config["severity"]["sigma"]
            else:
                mu = stressed_sev_params["lognormal_mu"]
                sigma = stressed_sev_params["lognormal_sigma"]

            # Apply UoM-specific severity stress
            stressed_mu = mu + uom_mu_shift
            stressed_sigma = sigma * uom_scale_mult

            # Minimum constraints
            stressed_mu = max(stressed_mu, 8.0)
            stressed_sigma = max(stressed_sigma, 1.0)

            # Update the temporary dictionary with STRESSED parameters
            stressed_sev_params["lognormal_mu"] = stressed_mu
            stressed_sev_params["lognormal_sigma"] = stressed_sigma

            # Step 5a: Frequency simulation
            num_losses_per_path = rng.poisson(stressed_lambda, size=num_simulations)
            if np.all(num_losses_per_path == 0):
                continue

            # Step 5b: Compute total severity draws needed
            total_draws_needed = np.sum(num_losses_per_path)
            if total_draws_needed == 0:
                continue

            # Step 5c: Severity simulation
            batch_severities = simulate_severity(
                n_draws=int(total_draws_needed),  # ensure int
                params=stressed_sev_params,
                rng=rng,
            )
            batch_severities = np.maximum(batch_severities, 5_000)  # no zero losses

            # Step 5d: Aggregate per-path losses
            path_losses = np.zeros(num_simulations)
            idx = 0
            # Ensure insurance parameters are correctly used/defaulted
            insurance_limit = config.get("insurance", {}).get(
                "coverage", None
            )  # Per-loss limit
            deductible = config.get("insurance", {}).get(
                "deductible", 0.0
            )  # default to 0

            deductible = config.get("insurance", {}).get(
                "deductible", 0.0
            )  # Per-loss deductible
            coverage_pct = config.get("insurance", {}).get(
                "coverage_pct", 1.0
            )  # Assuming 1.0 (100%) if not specified
            agg_limit = config.get("insurance", {}).get(
                "agg_limit", None
            )  # Aggregate limit
            agg_deductible = config.get("insurance", {}).get(
                "agg_deductible", 0.0
            )  # Aggregate deductible

            for i in range(num_simulations):
                n = num_losses_per_path[i]
                if n > 0:
                    path_sevs = batch_severities[idx : idx + n]
                    gross_loss_path = np.sum(path_sevs)

                    if insurance_enabled:
                        # 1. Apply mitigation to get insurer PAYOUTS (List[float])
                        payouts = apply_mitigation(
                            path_sevs.tolist(),
                            limit=insurance_limit,
                            deductible=deductible,
                            coverage=coverage_pct,
                            agg_limit=agg_limit,
                            agg_deductible=agg_deductible,
                        )
                        total_payout = np.sum(payouts)

                        # 2. Calculate NET LOSS for the path
                        path_losses[i] = gross_loss_path - total_payout

                    else:
                        # If insurance is disabled, the Net Loss is the Gross Loss
                        path_losses[i] = gross_loss_path

                    idx += n

            # Step 5e: Add UoM contribution to total
            total_loss_distribution += path_losses

    # Step 6: Debugging
    logger.info("=== SIMULATION DEBUG ===")
    logger.info(f"Total simulated paths: {len(total_loss_distribution):,}")
    logger.info(f"Paths with any loss > 0: {np.sum(total_loss_distribution > 0):,}")
    logger.info(f"Max annual loss: {total_loss_distribution.max():,.0f}")
    logger.info(
        f"99.9th percentile: {np.percentile(total_loss_distribution, 99.9):,.0f}"
    )
    logger.info(f"Mean annual loss: {total_loss_distribution.mean():,.0f}")

    # Step 7: Log end
    logger.info("Simulation complete.")
    return total_loss_distribution


def lda_run_engine(
    config: Dict[str, Any] | None = None,
    config_path: str | None = None,
) -> Tuple[np.ndarray, Dict[str, Dict[str, Any]], Dict[str, float]]:
    """
    Main: Fit models, run sim, compute metrics.

    Args:
        config: Full config dict.

    Returns:
        (loss_dist, fitted_models, metrics)

    """
    # Step 0: Handle config
    if config is None or isinstance(config, str) or config_path:
        cfg_path = Path(config_path or config or "config/op_config.yaml")
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        cfg_obj = OpRiskConfig(str(cfg_path))
        cfg_obj.validate()
        config = cfg_obj.as_dict()

    # Step 1: Merge with global defaults (seed, default_n_paths, etc.)
    full_config = merge_with_global(config)

    logger.info("--- Starting LDA Engine ---")
    logger.info(
        f"Using {full_config.get('simulation', {}).get('default_n_paths', 100_000):,} simulations"
    )
    logger.info(f"Random seed: {full_config.get('seed', 'not set')}")

    try:
        # Step 2: Prepare/fit models
        fitted_models = prepare_models(full_config)

        # Step 3: Run simulation using merged config
        loss_distribution = run_monte_carlo_simulation(fitted_models, full_config)

        # Step 4: Numerical stability fixes
        MAX_LOSS_CAP = 1e15  # Set a large, safe cap
        loss_distribution = np.nan_to_num(
            loss_distribution, nan=0.0, posinf=MAX_LOSS_CAP, neginf=0.0
        )  # Handle Inf and NaN values
        loss_distribution = np.clip(
            loss_distribution, 0.0, MAX_LOSS_CAP
        )  # Clip any remaining extreme values to the max cap

        # Step 5: Compute metrics
        capital_metrics = compute_capital_metrics(loss_distribution, full_config)

        # Step 6: Log end and return tuple
        logger.info(f"Run complete: {len(loss_distribution):,} paths.")
        return loss_distribution, fitted_models, capital_metrics

    except Exception as e:
        logger.error("Engine failed: {e}")
        raise RuntimeError(f"LDA run aborted: {e}") from e


if __name__ == "__main__":
    # Step 1: Demo setup (override config for quick test)
    # Demo: Small run for validation
    cfg_obj = OpRiskConfig("config/op_config.yaml")
    cfg_obj.update({"num_simulations": 10000, "SEED": 42})
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
