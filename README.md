# Financial Risk Toolkit

A Python package for computing Value-at-Risk (VaR), Expected Shortfall (ES), backtests, and Black–Scholes option pricing.

---

## 📦 Installation

```bash
# 1. Clone the repo
git clone https://github.com/LeoWang805/Risk_Project.git
cd Risk_Project

# 2. Create & activate a venv
python3 -m venv .venv
# on macOS / Linux:
source .venv/bin/activate  
# on Windows PowerShell:a
# .venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install in editable mode
pip install -e .

# 5. Run tests to verify
pytest -q
```

---

## 🗂 Project Structure

Risk_Project/
├── data/                   # Input CSVs (e.g. AAPL-bloomberg.csv)
├── notebooks/              # Jupyter walkthroughs
│   ├── 01_data_loader.ipynb
│   ├── 02_calibration.ipynb
│   ├── 03_var_es.ipynb
│   ├── 04_backtest.ipynb
│   └── 05_black_scholes.ipynb
├── src/
│   └── risk_project/       # Python package itself
│       ├── __init__.py
│       ├── backtest.py
│       ├── black_scholes.py
│       ├── calibration.py
│       ├── config.py
│       ├── data_loader.py
│       ├── monte_carlo.py
│       └── var_es.py
├── tests/                  # pytest unit tests
│   ├── test_backtest.py
│   ├── test_black_scholes.py
│   ├── test_monte_carlo.py
│   ├── test_var_es.py
│   └── conftest.py
├── requirements.txt        # pinned dependencies
├── setup.py                # package install script
└── README.md               # ← you’re here
---

## 🎯 Usage

### From Python

```python
from risk_project.data_loader    import load_price_series
from risk_project.calibration    import estimate_mu_sigma, estimate_covariance_matrix
from risk_project.var_es         import parametric_var_es, historical_var_es
from risk_project.monte_carlo    import monte_carlo
from risk_project.backtest       import compute_portfolio_pnl, compute_exceptions, kupiec_test
from risk_project.black_scholes  import bs_call, bs_put

# 1) Load data
series   = load_price_series(["data/AAPL-bloomberg.csv", "data/AMZN-bloomberg.csv"])
positions = {"AAPL": 100_000 / series["AAPL"].iloc[-1],
             "AMZN": 100_000 / series["AMZN"].iloc[-1]}

# 2) Calibrate
mu, sigma = {}, {}
for sym, ps in series.items():
    m, _ = estimate_mu_sigma(ps)
    mu[sym] = m
cov = estimate_covariance_matrix(series)

# 3) Compute Parametric VaR & ES
var, es = parametric_var_es(positions, series, mu, cov, p=0.99)
print(f"Parametric 1-day 99% VaR: ${var:,.0f}, ES: ${es:,.0f}")

# 4) Price a vanilla call
price = bs_call(S=100, K=100, vol=0.2, r=0.01, T=1.0)
print(f"Call price: ${price:.2f}")
```
---

## 4) Add “Notebooks” section

**Paste** this **next**:

```markdown
---

## 📝 Notebooks

Each file under `notebooks/` is a step-by-step Jupyter walkthrough:

1. **01_data_loader.ipynb**   — load CSVs into pandas  
2. **02_calibration.ipynb**   — estimate μ, σ, and covariance  
3. **03_var_es.ipynb**        — compute static VaR & ES  
4. **04_backtest.ipynb**      — rolling‐window backtests of VaR  
5. **05_black_scholes.ipynb** — Black–Scholes option pricing checks

Launch with:

```bash
jupyter lab   # or `jupyter notebook`

---

## 5) Add “Testing” section

**Paste** this **afterwards**:

```markdown
---

## ✅ Testing

We use [pytest]. To run the full suite:

```bash
pytest -q

---

## 6) Add “License & Authors”

Finally, **paste** at the bottom:

```markdown
---
