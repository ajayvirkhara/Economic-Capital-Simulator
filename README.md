# 🏦 Economic Capital Simulator
> Quantitative Risk Simulation Framework for Economic Capital: integrating Market, Credit, and Operational risk into one unified engine.

A modular Python framework for simulating **Economic Capital** across **Market, Credit, and Operational Risk** consistent with modern risk management and ICAAP principles.

This project provides a complete **Monte Carlo–based platform** for estimating portfolio-level risk, exposure, and capital requirements, designed for hedge funds, asset managers, and internal risk analytics functions.

---

## ⚙️ Overview

The Economic Capital Simulator models how a financial institution’s capital requirement evolves across three major risk stripes:

| Module | Purpose | Core Techniques |
|:---------|:----------|:----------------|
| **Market Risk** | Simulates correlated market shocks and portfolio P&L | Multivariate *t*-copula, Delta–Gamma VaR/ES, Stress Testing |
| **Credit Risk** | Models counterparty exposures under CSA and margining | Stylised MTM, Collateral Paths, Exposure Profiles (EE, EPE, PFE) |
| **Operational Risk** | Quantifies losses from operational failures | Loss Distribution Approach (LDA), EVT tails, Scenario/Insurance Adjustments |

Each risk type can be executed and validated independently, with shared utilities for logging, profiling, and configuration.

---

<details>
<summary>📖 Table of Contents</summary>

- [Overview](#️-overview)
- [Project Structure](#-project-structure)
- [Core Features](#-core-features)
- [Utilities](#-utilities)
- [Testing](#-testing)
- [Installation](#-installation)
- [Example Usage](#-example-usage)
- [License](#-license)
- [Author](#-author)

</details>

---

## 🧩 Project Structure

```bash

├── config/                             # Configuration files for risk engines
│   ├── default.yaml                              # Global default configuration
│   ├── credit_config.yaml                        # Parameters specific to credit_risk models
│   ├── market_config.yaml                        # Parameters specific to market_risk models
│   └── op_config.yaml                            # Parameters specific to op_risk models
├── docs/                               # Project documentation
│   └── README.md                                 # Detailed docs for the docs folder
├── econ_capital/                       # Primary Python source package (econ_capital)
│   ├── credit_risk/                              # Credit and Counterparty Credit Risk (CCR)
│   │   ├── allocation.py                                   # Capital allocation (Euler, proportional)
│   │   ├── ccr_engine.py                                   # Credit capital integration engine with stochastic modelling
│   │   ├── config.py                                       # Credit module config loader
│   │   ├── creditrisk_reporting.py                         # Regulatory-grade Excel report generation
│   │   ├── csa.py                                          # Credit Support Annex (CSA) logic
│   │   ├── data_loaders.py                                 # Credit spreads, indices, dummy data
│   │   ├── default_model.py                                # Default modeling with stochastic LGD/PD simulation
│   │   ├── demo_exposure.py                                # Demo for exposure profiles with term structure volatility
│   │   ├── exposure_engine.py                              # MTM + Collateral + Exposure with proper pricing support
│   │   ├── exposure_models.py                              # MTM computation with pricing model protocol
│   │   ├── generate_portfolio.py                           # Utility to generate realistic counterparty exposure CSV
│   │   ├── market_model.py                                 # Credit factors and term structure volatility
│   │   ├── run_creditrisk_report.py                        # Driver script for full Credit Risk simulation + report
│   │   ├── trade_models.py                                 # Trade classes with Black-Scholes and DV01 pricing
│   │   ├── wwr.py                                          # Structural Merton-style WWR model
│   │   ├── __init__.py                                     # Exposes the public API
│   │   └── __main__.py                                     # Module entry point
│   ├── market_risk/                              # Market Risk
│   │   ├── config.py                                       # Market module config loader
│   │   ├── covariance.py                                   # EWMA/GARCH covariance models
│   │   ├── data_loaders.py                                 # Factor and pricing data ingestion
│   │   ├── engine.py                                       # Monte Carlo risk engine
│   │   ├── shocks.py                                       # Multivariate t-copula simulation
│   │   ├── stats.py                                        # VaR, ES, and backtesting functions
│   │   ├── marketrisk_reporting.py                         # Regulatory-grade Excel report generation
│   │   ├── run_marketrisk_report.py                        # Driver script for full Market Risk simulation + report
│   │   ├── __init__.py                                     # Exposes the public API
│   │   └── __main__.py                                     # Module entry point
│   ├── op_risk/                                  # Operational Risk (OpRisk)
│   │   ├── data/                                           # Example/sample loss data files
│   │   │   ├── freq_data.csv                               # Sample frequency data
│   │   │   ├── sev_data.csv                                # Sample severity data
│   │   │   ├── empty_freq.csv                              # Empty template for frequency data
│   │   │   └── empty_sev.csv                               # Empty template for severity data
│   │   ├── config.py                                       # OpRisk module config loader
│   │   ├── data_loaders.py                                 # Incident/loss data ingestion
│   │   ├── dependencies.py                                 # Modeling dependencies (e.g., using copulas)
│   │   ├── frequency_models.py                             # Poisson / Negative Binomial frequency models
│   │   ├── insurance.py                                    # Risk transfer (insurance) adjustments
│   │   ├── lda_engine.py                                   # Loss Distribution Approach (LDA) simulation engine
│   │   ├── oprisk_reporting.py                             # Regulatory-grade Excel report generation
│   │   ├── run_oprisk_report.py                            # Driver script for full OpRisk simulation + stress tests + report
│   │   ├── scenarios.py                                    # Scenario-based stress tests
│   │   ├── severity_models.py                              # Lognormal, Pareto, EVT tail fitting
│   │   ├── stress_tests.py                                 # Scenario and sensitivity extensions
│   │   ├── utils.py                                        # Helper functions for data/plots
│   │   ├── __init__.py                                     # Exposes the public API
│   │   └── __main__.py                                     # Module entry point
│   ├── aggregate.py                              # Portfolio-level aggregation across risk stripes
│   ├── config_loader.py                          # Global + module-specific YAML config merging and defaults
│   ├── firmwide_reporting.py                     # Regulatory-grade firm-wide Excel report generation
│   ├── reporting_utils.py                        # Shared utilities for clean Excel chart styling and column autofitting
│   ├── run_full_ec.py                            # Master orchestrator for full firm-wide Economic Capital simulation
│   ├── utils.py                                  # Shared logging, profiling, and helpers
│   └── __init__.py                               # Package init and high-level API (run_full_simulation, etc.)
├── notebooks/                          # Demonstration and tutorial notebooks
│   ├── demo_credit.ipynb                         # Demo for Credit and CCR analysis
│   ├── demo_market.ipynb                         # Demo for Market Risk analysis
│   └── demo_oprisk.ipynb                         # Demo for Operational Risk analysis
├── tests/                              # Unit and integration tests using pytest
│   ├── credit_risk/                              # Tests for the credit_risk package
│   │   ├── test_cr_allocation.py
│   │   └── ... (other test files)
│   │
│   ├── market_risk/                              # Tests for the market_risk package
│   │   └── ... (test files)
│   │
│   ├── op_risk/                                  # Tests for the op_risk package
│   │   └── ... (test files)
│   ├── __init__.py                               # Makes tests a package (optional but present)
│   ├── conftest.py                               # Shared pytest fixtures
│   ├── test_aggregate.py                         # Tests diversification and aggregation logic
│   ├── test_config_loader.py                     # Tests YAML config merging and defaults
│   ├── test_reporting_utils.py                   # Unit tests for reporting_utils.py (chart styling, colors, labels, autofit)
│   └── test_run_full_simulation.py               # Tests full firm-wide orchestration
├── .coveragerc                         # Coverage configuration (omits demos, raises, etc.)
├── .gitattributes                      # Line ending normalization (LF for code/YAML, binary for Excel)
├── .gitignore                          # Specifies files/folders to ignore in Git
├── .pre-commit-config.yaml             # Pre-commit hooks (Ruff linter/formatter + pytest)
├── LICENSE                             # Project licensing information
├── pytest.ini                          # Configuration file for pytest
├── README.md                           # Project overview, installation, and usage
├── requirements.txt                    # Core runtime dependencies (minimal to run the project)
├── requirements-dev.txt                # Development tools, testing, linting & notebook support
└── setup.py                            # Setup script for packaging the project

```

---

## 🧠 Core Features

### 🔹 Credit Risk
* **Enhanced Derivative Pricing:**
  * **Proper pricing models:** Black-Scholes for European options, DV01-based pricing for interest rate swaps
  * **Fallback support:** Stylised linear + gamma approximation for basic trades (`MTM_t = w·ΔS/S₀ + 0.5·γ·(ΔS/S₀)² + add`)
  * **Configurable:** Toggle between proper vs. stylised pricing via `use_proper_models` flag

* **Stochastic Credit Parameters:**
  * **Stochastic LGD:** Beta distribution with configurable volatility (default 20%)
  * **Stochastic PD:** Log-normal model with credit cycle sensitivity (default 35% volatility)
  * **Path-wise simulation:** Monte Carlo loss distribution for tail risk estimation

* **Dynamic Volatility Term Structure:**
  * **Time-varying volatility:** Exponential decay from short-term to long-term equilibrium
  * **Mean reversion:** Configurable speed of convergence (default κ=0.40)
  * **Default term structure:** vol_short=30%, vol_long=18%

* **Structural Wrong-Way Risk (WWR):**
  * **Merton-style model:** Joint simulation of counterparty asset values and exposures
  * **Distance-to-default effects:** Exposure increases as creditworthiness deteriorates
  * **Negative correlation:** Configurable exposure-asset correlation (default -0.40)
  * **Fallback:** Simple heuristic WWR scaling for backward compatibility

* **CSA logic** with Variation & Initial Margin, thresholds, and flexible call schedules
* **Exposure metrics:** EE(t), PFE(t), EPE with alpha-factor adjustment to EAD
* **Demo**: `python -m econ_capital.credit_risk.demo_exposure`

### 🔹 Market Risk
* **Multivariate *t*-copula** for correlated market shocks
* **Delta–Gamma VaR/ES** estimation with decomposition
* **Stress testing** and historical shock replays
* **Covariance estimation** via EWMA or GARCH(1,1)
* **Euler allocation** for capital attribution

### 🔹 Operational Risk
* **Loss Distribution Approach (LDA)** combining Poisson frequency + hybrid Lognormal/GPD severity modelling
* **Lognormal, GDP, and hybrid** tail fitting
* **Expert scenario analysis, insurance mitigation, and stress testing add-ons**
* **Regulatory-style outputs** for ICAAP and Basel III/IV

### 🔹 Portfolio Aggregation
* **Capital Aggregation:** Simulates joint distribution of losses across all three risk stripes using a **fat-tailed t or Gaussian copula** to derive the final Diversified Economic Capital.
* **Diversification Benefit:** Quantifies the diversification effect ($EC_{\text{Total}} < \sum EC_{\text{Stripe}}$).

### 📊 Outputs & Regulatory-Grade Reporting

The framework generates professional **Excel reports** (via `openpyxl`) for each risk pillar and a consolidated firm-wide view. Reports include formatted tables, bar charts, styled headers, and multiple sheets for breakdowns.

All chart-heavy reports use shared styling utilities located in `econ_capital/reporting_utils.py`.  
This module provides:
- Consistent clean bar chart styling (color palette, no legend, rotated labels, £M formatting, manual layout)
- Automatic column autofitting with text wrapping and max-width capping

This ensures a professional, uniform look across Market, Credit, OpRisk, and Firm-Wide reports.

#### Market Risk Report
- **File**: `Market_Risk_EC_Report_*.xlsx` (in `econ_capital/market_risk/reports/`)
- **Key Sheets**:
  - Summary: 10-day & 1-year VaR/ES at 99.9%, stressed scenarios
  - Capital Breakdown: Top position contributions (Euler allocation on ES)
  - Positions: Full portfolio holdings table
  - Charts: Bar chart of largest contributors

#### Credit Risk Report
- **File**: `Credit_Risk_EC_Report_*.xlsx` (in `econ_capital/credit_risk/reports/`)
- **Key Sheets**:
  - Portfolio Summary: Total EL, UL, EC (99.9%)
  - Counterparty Breakdown: Per-counterparty EAD, PD, LGD, marginal EC
  - Exposure Profiles: EE, PFE (97.5%), EPE over horizon
  - Full Data: Detailed table with WWR adjustments

#### Operational Risk Report
- **File**: `OpRisk_EC_Report_*.xlsx` (in `econ_capital/op_risk/reports/`)
- **Key Sheets**:
  - Baseline Metrics: Expected Loss, VaR/ES at multiple levels
  - Stress Test Results: Capital under each scenario (frequency/severity shocks)
  - Scenario Details: Expert/deterministic scenario impacts
  - Charts: Bar chart comparing baseline vs. stressed capital

#### Firm-Wide Economic Capital Report
- **File**: `FirmWide_EC_Report_*.xlsx` (in `econ_capital/reports/`)
- **Key Sheets**:
  - Summary: Standalone vs. diversified EC, diversification benefit
  - Risk Contributions: Marginal EC by risk type (Market, Credit, OpRisk)
  - Confidence Levels: EC/VaR at 99%, 99.5%, 99.9%
  - Correlation Assumptions
  - Sensitivity/Stress Outputs
  - Detailed breakdowns from each pillar
  - Charts: Bar chart of EC by confidence level

All reports are automatically generated when running the individual risk drivers or the full aggregation script (`run_full_ec.py`). JSON summaries are also saved for programmatic consumption.

---

## 🔧 Utilities

### 🪵 Logging
Unified logging configuration across all modules:

```python
import logging
logger = logging.getLogger("econ_capital")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
```

* Consistent output formatting
* Logs runtime summaries and parameter setups

### ⏱️ Profiling

Lightweight execution timing is provided via the `timed_section` context manager in `econ_capital.utils`.

Usage example:
```python
from econ_capital.utils import timed_section

with timed_section("Market Risk Simulation"):
    # Your long-running code here
    results = engine.run()
```
  
### 📊 Excel Reporting Helpers

Shared utilities in econ_capital/reporting_utils.py provide:

* Clean, consistent styling for bar charts (colors, layout, labels, rotation)
* Automatic column width adjustment with text wrapping

Used by all risk pillar reporting modules.

---

## 🧪 Testing

All modules include comprehensive unit tests under the tests/ directory with >90% code coverage:

```bash
pytest -v --cov=econ_capital --cov-report=term-missing
```

Example output:

```bash
tests/credit_risk/test_cr_exposure_engine.py::test_vm_and_im_effects PASSED
tests/credit_risk/test_cr_trade_models.py::test_vanilla_swap_price PASSED
tests/credit_risk/test_cr_trade_models.py::test_european_option_price_call PASSED
tests/credit_risk/test_cr_market_model.py::test_simulate_term_structure_volatility_basic PASSED
tests/market_risk/test_mr_engine.py::test_engine_run_outputs PASSED
tests/op_risk/test_or_lda_engine.py::test_severity_simulation PASSED

---------- coverage: platform win32, python 3.11.5 -----------
Name                                      Stmts   Miss  Cover   Missing
-----------------------------------------------------------------------
econ_capital/credit_risk/default_model.py   145      8    94%   89-92, 156-159
econ_capital/credit_risk/trade_models.py     87      5    94%   45-47, 98-100
econ_capital/credit_risk/market_model.py     52      3    94%   78-80
-----------------------------------------------------------------------
TOTAL                                       2847    147    95%
```

---

## 🧰 Installation

### For Users
To run the simulations and generate reports:
```bash
git clone https://github.com/ajayvirkhara/Economic-Capital-Simulator.git
cd Economic-Capital-Simulator
pip install -r requirements.txt
pip install -e .
```

### For Developers
To contribute, run tests, or use Jupyter notebooks:
```bash
git clone https://github.com/ajayvirkhara/Economic-Capital-Simulator.git
cd Economic-Capital-Simulator
pip install -r requirements-dev.txt
pip install -e .
pre-commit install
```

### Core Runtime (requirements.txt)
───────────────────────────────────────────────

These packages are required to run the risk engines and generate reports.

#### Core Scientific Stack
* **numpy** (==2.1.1)
* **pandas** (==2.2.2)
* **scipy** (==1.14.1)

#### Risk/Finance Specific
* **yfinance** (==0.2.40)
* **arch** (==6.3.0)
* **pyyaml** (==6.0.2)
* **pandas-datareader** (==0.10.0)

#### Reporting & Excel
* **openpyxl** (==3.1.5)

#### Utilities
* **tqdm** (==4.66.5)

### Development & Quality (requirements-dev.txt)
───────────────────────────────────────────────

Additional tools for contributors / local development.

#### Testing
* **pytest** (==8.4.2)
* **pytest-cov** (==4.0.0)

#### Code Quality & Formatting
* **ruff**
* **pre-commit** (==3.8.0)

#### Notebook/Interactive Development
* **pytest** (>=1.0.0)
* **ipython**

#### Notebook tools
* **nbstripout** (>=0.6.0)

#### Build & Editable Installation
* **setuptools** (>=70.0.0)
* **-e .**

---

## 🚀 Example Usage

**Credit Risk Exposure**

```bash
python -m econ_capital.credit_risk
```

**Market Risk VaR Simulation**

```bash
python -m econ_capital.market_risk
```

**Operational Risk Capital**

```bash
python -m econ_capital.op_risk
```

**Full Firm-Wide Economic Capital (All Risks + Diversification)**

```bash
python -m econ_capital.run_full_ec
```

---

## 🪪 License

Licensed under the **MIT License** - free for educational and research use with attribution.

---

## 👤 Author

**Ajayvir Khara**
*MSc Finance | Passed FRM Part I and Part II*

* LinkedIn: [LinkedIn](https://www.linkedin.com/in/ajayvirkhara)
* GitHub: [github.com/ajayvirkhara](https://github.com/ajayvirkhara)
