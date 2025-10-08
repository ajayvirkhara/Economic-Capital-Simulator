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
| **Operational Risk** | Quantifies losses from operational failures | Loss Distribution Approach (LDA), EVT tails, Scenario Aggregation |

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
econ_capital/
│
├── credit_risk/
│ ├── allocation.py         # Capital allocation (Euler, proportional)
│ ├── ccr_engine.py         # Credit capital integration engine
│ ├── config.py             # Module-level config loader
│ ├── credit_risk.py        # Orchestration script
│ ├── csa.py                # CSA logic (VM/IM thresholds, call frequency)
│ ├── data_loaders.py       # Credit spreads, indices, dummy data
│ ├── default_model.py      # Default and recovery modelling
│ ├── demo_exposure.py      # Demo for exposure profiles and CSA dynamics
│ ├── exposure_engine.py    # Combines MTM + Collateral + Exposure summary
│ ├── exposure_models.py    # Stylised MTM and collateral mechanics
│ ├── market_model.py       # Credit spread and macro driver integration
│ ├── trade_models.py       # Stylised trades and netting sets
│ ├── wwr.py                # Wrong-way risk extensions
│ └── __init__.py
│
├── market_risk/
│ ├── config.py             # Parameter config
│ ├── covariance.py         # EWMA/GARCH covariance models
│ ├── data_loaders.py       # Factor and pricing data ingestion
│ ├── engine.py             # Monte Carlo risk engine
│ ├── market_risk.py        # Driver script
│ ├── shocks.py             # Multivariate t-copula simulation
│ ├── stats.py              # VaR, ES, and backtesting functions
│ └── __init__.py
│
├── op_risk/
│ ├── data_loaders.py       # Incident/loss data ingestion
│ ├── frequency_models.py   # Poisson / Negative Binomial frequency models
│ ├── severity_models.py    # Lognormal, Pareto, EVT tail fitting
│ ├── lda_engine.py         # LDA simulation for OpRisk capital
│ ├── stress_tests.py       # Scenario and sensitivity extensions
│ ├── insurance.py          # Risk transfer adjustments
│ ├── reporting.py          # Capital summary generation
│ ├── utils.py              # Helper functions for data/plots
│ └── __init__.py
│
├── utils.py                # Shared logging, profiling, and helpers
├── aggregate.py            # Portfolio-level aggregation across risk stripes
└── __init__.py
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
* **Loss Distribution Approach (LDA)** combining frequency and severity
* **Extreme Value Theory (EVT)** for tail fitting
* **Scenario analysis and insurance mitigation**
* **Regulatory-style outputs** for ICAAP and Basel III/IV

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

### 🧪 Testing

All modules include unit tests under the tests/ directory:

```bash
pytest -v
```

Example output:

```bash
tests/test_credit_risk.py::test_linear_trade_mtm_increases_with_price PASSED
tests/test_credit_risk.py::test_vm_and_im_effects PASSED
tests/test_market_risk.py::test_var_computation PASSED
tests/test_op_risk.py::test_lda_capital_estimate PASSED
```

---

## 🧰 Installation

```bash
git clone https://github.com/ajayvirkhara/Economic-Capital-Simulator.git
cd Economic-Capital-Simulator
pip install -r requirements.txt
```

Requirements:
* numpy, pandas, pandas_datareader
* scipy, matplotlib
* pytest, pylint
* (optional) jupyter for demo notebooks

---

## 🚀 Example Usage

**Credit Risk Exposure**

```bash
python -m econ_capital.credit_risk.demo_exposure
```

**Market Risk VaR Simulation**

```python
from econ_capital.market_risk.engine import run_var_simulation
run_var_simulation(config_path="config/market_config.yaml")
```

**Operational Risk Capital**

```python
from econ_capital.op_risk.lda_engine import simulate_op_capital
simulate_op_capital(config="config/op_config.yaml")
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