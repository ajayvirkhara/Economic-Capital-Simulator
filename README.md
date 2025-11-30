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

├── config/                     # Configuration files for risk engines
│ ├── default.yaml            # Global default configuration
│ ├── credit_config.yaml      # Parameters specific to credit_risk models
│ ├── market_config.yaml      # Parameters specific to market_risk models
│ └── op_config.yaml          # Parameters specific to op_risk models
│
├── docs/                       # Project documentation (e.g., Sphinx build)
│ └── README.md               # Detailed docs for the docs folder
│
├── econ_capital/               # Primary Python source package (econ_capital)
│ ├── credit_risk/            # Credit and Counterparty Credit Risk (CCR)
│ │ ├── allocation.py         # Capital allocation (Euler, proportional)
│ │ ├── ccr_engine.py         # Credit capital integration engine
│ │ ├── config.py             # Credit module config loader (e.g., config checks)
│ │ ├── csa.py                # Credit Support Annex (CSA) logic
│ │ ├── data_loaders.py       # Credit spreads, indices, dummy data
│ │ ├── default_model.py      # Default and recovery modelling
│ │ ├── demo_exposure.py      # Demo for exposure profiles and CSA dynamics
│ │ ├── exposure_engine.py    # Combines MTM + Collateral + Exposure summary
│ │ ├── exposure_models.py    # Stylised MTM and collateral mechanics
│ │ ├── market_model.py       # Credit spread and macro driver integration
│ │ ├── trade_models.py       # Stylised trades and netting sets
│ │ ├── wwr.py                # Wrong-way risk (WWR) extensions
│ │ ├── __init__.py           # Exposes the public API
│ │ └── __main__.py           # Module entry point
│ │
│ ├── market_risk/            # Market Risk
│ │ ├── config.py             # Market module config loader
│ │ ├── covariance.py         # EWMA/GARCH covariance models
│ │ ├── data_loaders.py       # Factor and pricing data ingestion
│ │ ├── engine.py             # Monte Carlo risk engine
│ │ ├── shocks.py             # Multivariate t-copula simulation
│ │ ├── stats.py              # VaR, ES, and backtesting functions
│ │ ├── __init__.py           # Exposes the public API
│ │ └── __main__.py           # Module entry point
│ │
│ ├── op_risk/                # Operational Risk (OpRisk)
│ │ ├── data/                 # Example/sample loss data files
│ │ │ ├── freq_data.csv       # Sample frequency data
│ │ │ └── sev_data.csv        # Sample severity data
│ │ ├── config.py             # OpRisk module config loader
│ │ ├── data_loaders.py       # Incident/loss data ingestion
│ │ ├── dependencies.py       # Modeling dependencies (e.g., using copulas)
│ │ ├── frequency_models.py   # Poisson / Negative Binomial frequency models
│ │ ├── insurance.py          # Risk transfer (insurance) adjustments
│ │ ├── lda_engine.py         # Loss Distribution Approach (LDA) simulation engine
│ │ ├── reporting.py          # Capital summary generation
│ │ ├── scenarios.py          # Scenario-based stress tests
│ │ ├── severity_models.py    # Lognormal, Pareto, EVT tail fitting
│ │ ├── stress_tests.py       # Scenario and sensitivity extensions
│ │ ├── utils.py              # Helper functions for data/plots
│ │ ├── __init__.py           # Exposes the public API
│ │ └── __main__.py           # Module entry point
│ │
│ ├── aggregate.py            # Portfolio-level aggregation across risk stripes
│ ├── utils.py                # Shared logging, profiling, and helpers
│ └── __init__.py
│
├── notebooks/                  # Demonstration and tutorial notebooks
│ ├── demo_credit.ipynb       # Demo for Credit and CCR analysis
│ ├── demo_market.ipynb       # Demo for Market Risk analysis
│ └── demo_oprisk.ipynb       # Demo for Operational Risk analysis
│
├── tests/                      # Unit and integration tests using pytest
│ ├── credit_risk/            # Tests for the credit_risk package
│ │ ├── test_cr_allocation.py
│ │ └── ... (other test files)
│ │
│ ├── market_risk/            # Tests for the market_risk package
│ │ └── ... (test files)
│ │
│ ├── op_risk/                # Tests for the op_risk package
│ │ └── ... (test files)
│ │
│ └── conftest.py             # Shared pytest fixtures
│
├── .gitignore                  # Specifies files/folders to ignore in Git
├── .pylintrc                   # Configuration file for Pylint static analysis
├── LICENSE                     # Project licensing information
├── pytest.ini                  # Configuration file for pytest
├── README.md                   # Project overview, installation, and usage
├── requirements.txt            # List of required Python packages
└── setup.py                    # Setup script for packaging the project

```

---

## 🧠 Core Features

### 🔹 Credit Risk
* **Stylised MTM model** for linear and convex trades

```python
MTM_t = w·ΔS/S₀ + 0.5·γ·(ΔS/S₀)² + add
```
* **CSA logic** with Variation & Initial Margin, thresholds, and call schedules
* **Exposure metrics:** EE(t), PFE(t), and cumulative EPE
* **Wrong-way risk module** for exposure–credit correlation
* **Demo**: `python -m econ_capital.credit_risk.demo_exposure`

### 🔹 Market Risk
* **Multivariate *t*-copula** for correlated market shocks
* **Delta–Gamma VaR/ES** estimation with decomposition
* **Stress testing** and historical shock replays
* **Covariance estimation** via EWMA or GARCH(1,1)
* **Euler allocation** for capital attribution

### 🔹 Operational Risk
* **Loss Distribution Approach (LDA)** combining frequency and severity modelling
* **Lognormal, GDP, and hybrid** tail fitting
* **Scenario analysis, insurance, and stress add-ons**
* **Regulatory-style outputs** for ICAAP and Basel III/IV

### 🔹 Portfolio Aggregation
* **Capital Aggregation:** Simulates joint distribution of losses across all three risk stripes using an **aggregation copula** to derive the final Diversified Economic Capital.
* **Diversification Benefit:** Quantifies the diversification effect ($EC_{\text{Total}} < \sum EC_{\text{Stripe}}$).

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

Profiling utilities are embedded within test scripts (e.g. `@profile_test` decorator) using Python’s `cProfile` and `pstats` modules to identify performance bottlenecks.

## 🧪 Testing

All modules include unit tests under the tests/ directory:

```bash
pytest -v
```

Example output:

```bash
tests/credit_risk/test_cr_exposure_engine.py::test_vm_and_im_effects PASSED
tests/market_risk/test_mr_engine.py::test_engine_run_outputs PASSED
tests/op_risk/test_or_lda_engine.py::test_severity_simulation PASSED
```

---

## 🧰 Installation

```bash
git clone https://github.com/ajayvirkhara/Economic-Capital-Simulator.git
cd Economic-Capital-Simulator
pip install -r requirements.txt
```

Requirements:

### Core Scientific Stack
* **numpy** ($\ge 2.1.1$)
* **pandas** ($\ge 2.2.2$)
* **scipy** ($\ge 1.14.1$)

### Risk/Finance Specific
* **yfinance** ($\ge 0.2.40$)
* **arch** ($\ge 6.3.0$)
* **pyyaml** ($\ge 6.0.2$)

### Development & Testing
* **pytest** ($\ge 8.4.2$)
* **pylint** ($\ge 3.2.7$)
* **black** ($\ge 24.8.0$) - (Used for code formatting)

### Optional
* **jupyter** - (Required to run the demo notebooks in the `notebooks/` directory)

---

## 🚀 Example Usage

**Credit Risk Exposure**

```bash
python -m econ_capital.credit_risk.demo_exposure
```

**Market Risk VaR Simulation**

```bash
python -m econ_capital.market_risk.__main__
```

**Operational Risk Capital**

```bash
python -m econ_capital.op_risk.__main__
```

**Full Economic Capital Aggregation**

```bash
python -m econ_capital.aggregate
```

---

## 🪪 License

Licensed under the **MIT License** - free for educational and research use with attribution.

---

## 👤 Author

**Ajayvir Khara**
*MSc Finance | FRM Level II Candidate*

* LinkedIn: [LinkedIn](https://www.linkedin.com/in/ajayvirkhara)
* GitHub: [github.com/ajayvirkhara](https://github.com/ajayvirkhara)
