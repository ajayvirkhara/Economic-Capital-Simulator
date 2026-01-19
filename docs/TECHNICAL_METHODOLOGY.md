> **Note:** For installation and usage instructions, see the [main README](../README.md).  
> Best viewed with a Markdown renderer supporting LaTeX (GitHub, VS Code with Math extensions).

# 📚 Technical Methodology & Implementation Guide

This document provides a deep-dive into the quantitative frameworks, stochastic processes, and mathematical methodologies implemented in the **Economic Capital Simulator**. The project is architected to meet **Pillar 2 (ICAAP)** standards for internal capital adequacy.

---

## Prerequisites

This document assumes familiarity with:
* Stochastic calculus and Monte Carlo simulation
* Copula theory and multivariate distributions
* Basel II/III framework and ICAAP principles
* Risk metrics (VaR, ES, CVA)

**For practitioners**: Understanding of market, credit, and operational risk management.  
**For developers**: Familiarity with NumPy/SciPy for numerical computing.

---

## 1. Market Risk Engine: Volatility Dynamics & Coherent Measures
The Market Risk engine is designed to capture the non-normalities and time-varying nature of financial markets.

* **Volatility Clustering:** Implements **EWMA** and **GARCH(1,1)** estimation to account for heteroskedasticity. This ensures the model reacts to recent market "shocks" rather than assuming constant variance.
* **Multivariate Student-t Shocks:** The engine draws shocks from a **Multivariate Student-t distribution** ($df=3$) using the Gaussian mixture representation (mathematically equivalent to a t-copula). This explicitly models "Fat Tails" (leptokurtosis) observed in historical asset returns.
* **Covariance Estimation:** 
  * **GARCH(1,1):** Estimated via maximum likelihood using the `arch` package with default initialization
  * **EWMA:** Exponentially weighted with λ=0.97 (configurable)
* **Horizon Scaling:** 10-day risk measures are scaled to 1-year equivalents using the square-root-of-time rule:

$$
\text{VaR}_{\text{1Y}} = \text{VaR}_{\text{10D}} \times \sqrt{\frac{252}{10}}
$$

* **Simulation Paths:** 500,000 Monte Carlo paths (configurable via `n_paths` parameter).
* **Non-Linear P&L Mapping:** Linear (delta) + quadratic (gamma) approximation of position-level P&L from factor shocks. No full revaluation or path-dependent pricing is implemented.
* **Coherent Risk Measures:** * **Value at Risk (VaR):** The $\alpha$-quantile of the loss distribution.
    * **Expected Shortfall (ES):** Calculates the average loss beyond the VaR threshold. ES is a **coherent risk measure** because it satisfies the property of **subadditivity**, ensuring diversification is mathematically recognized.
* **Euler Allocation (Marginal ES):** Capital is attributed back to individual positions using the principle of marginal contribution:

$$
\text{Contribution}_i = E[L_i \mid L_{\text{portfolio}} \ge \text{VaR}_\alpha]
$$


---

## 2. Credit Risk Engine: Stochastic CCR with Structural WWR

The Credit engine has been enhanced from stylised approximations to a comprehensive stochastic framework capturing realistic tail dependencies.

### 2.1 Proper Derivative Pricing

**Black-Scholes Option Pricing:**
For European options, the engine implements the closed-form Black-Scholes formula:

$$
C(S,t) = S \cdot \Phi(d_1) - K e^{-r(T-t)} \cdot \Phi(d_2)
$$

$$
P(S,t) = K e^{-r(T-t)} \cdot \Phi(-d_2) - S \cdot \Phi(-d_1)
$$

where:

$$
d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)(T-t)}{\sigma\sqrt{T-t}}, \quad d_2 = d_1 - \sigma\sqrt{T-t}
$$

**DV01-based Swap Pricing:**
Interest rate swaps use a simplified DV01 approximation:

$$
\text{MTM}_{\text{swap}} = \text{DV01} \times (r_{\text{market}} - r_{\text{fixed}}) \times 10000
$$

$$
\text{DV01} = \frac{\text{Notional} \times \text{Tenor}}{10000}
$$

**Fallback to Stylised Pricing:**
For basic Trade objects without a `price()` method, the engine falls back to the linear + gamma approximation:

$$
\text{MTM}_t = w \cdot \frac{\Delta S}{S_0} + \frac{1}{2} \gamma \cdot \left(\frac{\Delta S}{S_0}\right)^2 + \text{add}
$$

### 2.2 Stochastic Credit Parameters

**Stochastic Loss Given Default (LGD):**
LGD is modeled via a Beta distribution to ensure values remain in [0,1]:

$$
\text{LGD} \sim \text{Beta}(\alpha, \beta)
$$

where α and β are calibrated to match:
- Mean: base LGD from data
- Variance: `lgd_volatility²` × (base LGD)²

**Stochastic Probability of Default (PD):**
PD follows a log-normal process with credit cycle sensitivity:

$$
\ln(\text{PD}_i) = \ln(\text{PD}_{\text{base}}) + \beta \cdot Z_{\text{sys}} + \sigma_{\text{idio}} \cdot Z_i
$$

where:
- $Z_{\text{sys}}$ = systematic credit factor (common across counterparties)
- $Z_i$ = idiosyncratic shock
- β = `pd_volatility` × `shock_scale` (default 0.35 × 0.10 = 0.035)

**Path-wise Loss Simulation:**
For each counterparty and each Monte Carlo path:

$$
L_{i,p} = \text{EAD}_i \times \text{LGD}_{i,p} \times \text{PD}_{i,p}
$$

Expected Loss and Unexpected Loss:

$$
\text{EL}_i = \mathbb{E}_p[L_{i,p}], \quad \text{UL}_i = \sqrt{\text{Var}_p(L_{i,p})}
$$

### 2.3 Time-Varying Volatility Term Structure

Market factor volatility decays from short-term to long-term equilibrium via exponential mean reversion:

$$
\sigma(t) = \sigma_{\infty} + (\sigma_0 - \sigma_{\infty}) \cdot e^{-\kappa t}
$$

where:
- $\sigma_0$ = `vol_short` (short-term volatility, default 30%)
- $\sigma_{\infty}$ = `vol_long` (long-term equilibrium, default 18%)
- κ = `mean_reversion` (speed of convergence, default 0.40)

This replaces constant volatility in market path simulation, capturing the empirical term structure of implied volatility.

### 2.4 Structural Wrong-Way Risk (WWR)

The engine implements a Merton-style framework for joint exposure-default modeling.

**Asset Value Process (Merton Model):**
Each counterparty's asset value follows a geometric Brownian motion:

$$
A_i = A_0 \exp\left[\left(\mu - \frac{\sigma_A^2}{2}\right) + \sigma_A \left(\sqrt{\rho} Z_{\text{sys}} + \sqrt{1-\rho} Z_i\right)\right]
$$

**Default Threshold:**
Default occurs when asset value falls below a threshold derived from PD:

$$
D_i = \Phi^{-1}(\text{PD}_i)
$$

**Exposure-Asset Correlation:**
Exposures are adjusted based on distance-to-default:

$$
\text{Exposure}_{i,p}^{\text{WWR}} = \text{EAD}_i \times \left[1 + 0.5 \cdot \max\left(-\frac{A_{i,p} - D_i}{\sigma_A}, 0\right)^2\right] \times (1 + 0.15 \cdot Z_{i,p}^{\text{corr}})
$$

where:

$$
Z_{i,p}^{\text{corr}} = \rho_{\text{EA}} \cdot \frac{A_{i,p} - \bar{A}_i}{\sigma_A} + \sqrt{1 - \rho_{\text{EA}}^2} \cdot Z_{\text{expo}}
$$

and $\rho_{\text{EA}}$ = `correlation_expo_asset` (default -0.40, negative = wrong-way).

**Conditional PD (Vasicek Model):**

$$
\text{PD}_i^{\text{cond}}(Z_{\text{sys}}) = \Phi\left(\frac{\Phi^{-1}(D_i) - \sqrt{\rho} Z_{\text{sys}}}{\sqrt{1-\rho}}\right)
$$

**Backward Compatibility:**
A simple heuristic WWR scaler is retained for legacy usage:

$$
\text{EL}^{\text{WWR}} = \text{EL} \times (1 + s \times \max(Z_{\text{factor}}, 0))
$$

where s = `sensitivity` (typically 0.3–0.7).

### 2.5 Exposure Profiles & CSA Mechanics

* **Variation Margin (VM):** Applied at configurable frequencies (daily, weekly, annual) with threshold and MTA checks
* **Initial Margin (IM):** Static buffer added to collateral balance
* **Exposure Metrics:**
  - Expected Exposure: $\text{EE}(t) = \mathbb{E}[\max(\text{MTM}(t) - \text{Collateral}(t), 0)]$
  - Potential Future Exposure: $\text{PFE}_\alpha(t) = \text{quantile}_\alpha[\max(\text{MTM}(t) - \text{Collateral}(t), 0)]$
  - Expected Positive Exposure: $\text{EPE} = \frac{1}{T} \int_0^T \text{EE}(t) \, dt$
  - Exposure at Default: $\text{EAD} = \alpha_{\text{factor}} \times \text{EPE}$ (regulatory alpha = 1.4)

### 2.6 Portfolio Aggregation

**Unexpected Loss (UL):**
Portfolio UL incorporates correlation structure:

$$
\text{UL}_{\text{portfolio}} = \sqrt{\mathbf{UL}^\top \mathbf{R} \, \mathbf{UL}}
$$

where **R** is the counterparty correlation matrix.

**Economic Capital:**

$$
\text{EC} = \text{EL}_{\text{total}} + \Phi^{-1}(\alpha) \times \text{UL}_{\text{portfolio}}
$$

Default confidence level: α = 99.9% (Student-t quantile with df=3).

---

## 3. Operational Risk Engine: Hybrid LDA & EVT
The Operational Risk engine follows the **Loss Distribution Approach (LDA)**, utilizing Extreme Value Theory to handle "Low Frequency, High Severity" events.

* **Frequency Modeling:** Modeled via a **Poisson process** ($\lambda$), capturing the discrete count of loss events over the annual capital horizon.
* **Hybrid Severity Model (Body + Tail):**
    * **Body:** A **Lognormal distribution** is used to model the high-frequency, low-impact "expected" losses.
    * **Tail:** **Extreme Value Theory (EVT)** is applied via the **Generalized Pareto Distribution (GPD)** for losses exceeding a specific threshold ($u$). This ensures the "Black Swan" tail is not underestimated.
* **Monte Carlo Aggregation:** The frequency and severity distributions are combined via a large number of Monte Carlo paths (default ~500k, configurable) to produce the aggregate annual loss distribution.
* **Expert Judgment & Scenarios:** Provides hooks to inject subjective scenario analysis into the simulation, allowing for the inclusion of tail events not present in historical data.

---

## 4. Firm-Wide Aggregation: Copula & Diversification
The simulator's "Master" layer aggregates the three risk stripes into a single, unified Economic Capital charge.

* **Student-t Copula ($df=3$):** Used to join the marginal distributions. Unlike a Gaussian copula, the Student-t copula possesses **Tail Dependence**, meaning it captures the empirical reality that in a market crash, correlations between Market, Credit, and OpRisk tend to spike simultaneously.
* **Diversification Benefit:** Quantified as the capital relief achieved by moving from a simple sum of standalone risks to a correlation-aware aggregate.

$$
\text{Diversification} = \left(\sum \text{EC}_{\text{standalone}}\right) - \text{EC}_{\text{diversified}}
$$

* **ICAAP Integration:** The framework supports the calculation of **RAROC** (Risk-Adjusted Return on Capital) by providing a rigorous denominator for performance measurement.

---

## 5. Calibration & Model Validation

### Parameter Selection

#### Market Risk Parameters

- **Student-t degrees of freedom (df=3)**: Shocks are generated from a multivariate Student-t distribution with ν=3 via the Gaussian mixture representation (equivalent to a t-copula). This choice reflects heavy tails commonly observed in financial returns during crises.
- **Covariance estimation**:
  - Default method: EWMA with decay λ=0.97 (~33-day effective memory).
  - Alternatives: sample covariance or GARCH(1,1) univariate volatilities × sample correlation matrix (via `arch` package when available).
- **Horizon scaling**: 10-day VaR/ES scaled to 1-year using √(252/10) rule.
- **Monte Carlo paths**: Default 500,000 (configurable via `n_paths`); 750,000 used in firm-wide aggregation when t-copula is active.

#### Credit Risk Parameters

**Core Risk Parameters:**
- **Systemic correlation (ρ)**: Default 0.2 in 1-factor Gaussian credit model (used in `simulate_credit_factors`)
- **Alpha factor (α)**: 1.4 applied to cumulative EPE → EAD (regulatory multiplier in `ExposureEngine`)
- **Base recovery rate**: Default 40% (LGD=60%) used as baseline for stochastic LGD simulation
- **Confidence level**: 99.9% for economic capital (Student-t df=3 quantile)
- **Heuristic WWR sensitivity**: 0.3–0.7 range (available as fallback when `use_structural_model=false`)

**Pricing Models:**
- **Use proper models flag**: `use_proper_models` (default `true`) enables Black-Scholes/DV01 pricing
- **Black-Scholes parameters**: Risk-free rate 2%, configurable per option
- **Swap DV01**: Based on notional and tenor (5 years default)

**Stochastic Credit Parameters:**
- **LGD volatility**: Default 20% (Beta distribution variance)
- **PD volatility**: Default 35% (log-normal idiosyncratic component)
- **Shock scale**: Default 0.10 (systematic PD sensitivity multiplier)
- **LGD bounds**: [10%, 90%] hard limits
- **PD bounds**: [0.01%, 50%] hard limits

**Volatility Term Structure:**
- **Short-term vol (σ₀)**: Default 30%
- **Long-term vol (σ∞)**: Default 18%
- **Mean reversion (κ)**: Default 0.40
- **Formula**: σ(t) = σ∞ + (σ₀ - σ∞)·exp(-κt)

**Structural WWR:**
- **Asset volatility**: Default 35%
- **Asset correlation**: Default 0.25 (systemic component)
- **Exposure-asset correlation**: Default -0.40 (negative = wrong-way)
- **Distance-to-default scaling**: Quadratic multiplier (max 0.5x)

#### Operational Risk Parameters

- **Frequency model**: Poisson with λ = sample mean of historical counts; fallback λ=DEFAULT_LAMBDA (1.0) if data issues occur.
- **Severity model**:
  - Body: Lognormal fitted via MLE on log(losses) below threshold.
  - Tail: Generalized Pareto (GPD) fitted to excesses above threshold using `scipy.optimize.minimize` (L-BFGS-B).
  - Default threshold: 99th percentile or user-specified; minimum ~10 exceedances recommended (not enforced).
  - GPD shape ξ typically constrained in optimization bounds (e.g. [-0.25, 0.35]) for realism/stability.
- **Insurance mitigation**: Per-loss deductible + limit + coinsurance; aggregate deductible + limit with pro-rata scaling when aggregate constraints bind.
- **Monte Carlo paths**: Inherited from global setting (default 500,000+); configurable per run.

#### Firm-Wide Aggregation

- **Copula**: Student-t with df=3 (default); draws correlated uniform variates → marginal inverse CDFs.
- **Inter-risk correlations** (configurable matrix in `aggregate.py`):
  - Market ↔ Credit: 0.3
  - Market ↔ OpRisk: 0.1
  - Credit ↔ OpRisk: 0.2
- **Simulation size**: Typically 750,000 paths for portfolio-level joint loss distribution.

### Validation Framework

#### Model Fitting & Diagnostics

- **GPD**: Negative log-likelihood optimization; no automatic threshold selection (user should inspect mean excess plot).
- **Lognormal body**: No formal normality test enforced (Jarque-Bera or similar can be added by user).
- **GARCH**: Falls back to sample/EWMA covariance if `arch` fit fails or package unavailable.
- **Numerical safeguards**:
  - Covariance matrices receive small jitter (≈1e-8 × trace) before Cholesky.
  - Losses clipped to [0, 1e15].
  - Array shapes validated via `validate_shape()`.
  - NaN/Inf replaced or avoided where possible.

#### Stress Testing & Scenario Analysis

- **Market Risk**: Deterministic mean shifts (e.g. -40% equity, +200 bps rates) while keeping covariance structure.
- **Operational Risk**:
  - Scenario set supports frequency multipliers, severity location/scale shifts.
  - Includes expert judgment, stochastic random scenarios, and deterministic shocks (cyber, pandemic, etc.).
  - Parallel execution via `ProcessPoolExecutor`.
- Historical scenario replay: Not implemented (framework allows custom mean-shift vectors).

#### Monte Carlo & Statistical Properties

- **Convergence**: Relies on large sample size (≥500k paths); no automated standard error reporting.
- **Reproducibility**: Global seed (default 42) propagated via `np.random.default_rng(seed)`.
- **Variance reduction**: None implemented (no antithetic variates, control variates, etc.).

#### Backtesting & Out-of-Sample Validation

- **Backtesting**: Not automated (no Kupiec, Christoffersen, traffic-light tests implemented).
- **Cross-validation**: Not enforced — full historical data used for OpRisk severity fitting and GARCH calibration.
- **Recommended practice** (user responsibility):
  - Reserve recent period for out-of-sample VaR/ES coverage tests.
  - Monitor number of 99.9% breaches in 250–500 day windows.

### Known Limitations

- **Static correlations**: Inter-risk correlations are fixed parameters, not dynamically estimated from data.
- **No model selection**: GARCH(1,1) and Lognormal-GPD are hardcoded choices; alternative specifications (e.g., EGARCH, Student-t severity) not automated.
- **Threshold sensitivity**: OpRisk GPD results sensitive to threshold choice; no automated threshold optimization.
- **Insurance mitigation**: Pro-rata scaling applied when aggregate limits bind; more sophisticated risk transfer modeling (e.g., layer optimization) not implemented.
- **Credit risk simplifications**: No path-dependent exotic derivatives; CSA logic simplified (bilateral vs. unilateral not distinguished), parameters use defaults or config files; no automated MLE/Bayesian calibration to market data for credit parameters.
- **No variance reduction techniques**: Monte Carlo simulations use raw sampling without antithetic variates, control variates, importance sampling, or quasi-Monte Carlo methods.

## 6. References & Standards

### Regulatory Framework

- Basel Committee on Banking Supervision (2019). *Minimum capital requirements for market risk* (FRTB).
- Basel Committee (2006). *International Convergence of Capital Measurement and Capital Standards* (Basel II) — Advanced Measurement Approach for operational risk.
- EBA Guidelines on ICAAP and ILAAP (2020 onwards).

### Academic & Technical References

- McNeil, A.J., Frey, R., Embrechts, P. (2015). *Quantitative Risk Management: Concepts, Techniques and Tools*. Princeton University Press.
- Embrechts, P., Klüppelberg, C., Mikosch, T. (1997). *Modelling Extremal Events for Insurance and Finance*. Springer.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.
- Chavez-Demoulin, V., Embrechts, P., Nešlehová, J. (2006). Quantitative models for operational risk: extremes, dependence and aggregation. *Journal of Banking & Finance*, 30(10), 2635–2658.
- Gregory, J. (2012). *Counterparty Credit Risk and Credit Value Adjustment*. Wiley Finance.
- Black, F., Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*, 81(3), 637–654.
- Merton, R.C. (1974). On the Pricing of Corporate Debt: The Risk Structure of Interest Rates. *Journal of Finance*, 29(2), 449–470.
- Vasicek, O. (1987). Probability of Loss on Loan Portfolio. *KMV Corporation*.
- Pykhtin, M., Zhu, S. (2007). A Guide to Modeling Counterparty Credit Risk. *GARP Risk Review*, July/August.

### Implementation Standards

- Code style: PEP 8 (enforced via (`ruff`) in pre-commit where configured).
- Reproducibility: Global seeding via (`default.yaml`); all random number generation uses NumPy's (`default_rng(seed)`).
- Numerical stability:
  - Small diagonal jitter (1e-8 x I) on covariance matrices.
  - Hard clipping of simulated losses ([0, 1e15]).
  - Explicit shape validation helper (`validate_shape()`).
- Testing: (`pytest`) suite covering core engine functionality (frequency models, severity models, exposure calculations, allocation logic).
- Dependencies kept minimal: numpy, pandas, scipy, openpyxl, yfinance, arch (optional).

---

## 7. Tech Stack & Quality Assurance
* **Computation:** `NumPy` and `SciPy` for vectorization of Monte Carlo paths.
* **Reporting:** `openpyxl` for generating automated, formatted Excel workbooks with conditional formatting and data validation.
* **Workflow:** `pre-commit` hooks for code formatting (`ruff`) and notebook maintenance (`nbstripout`).
* **Verification:** `pytest` suite for validating stochastic distributions and engine convergence.
