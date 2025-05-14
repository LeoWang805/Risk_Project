Here’s a fully revised **README.md** that incorporates all of your new backtest graphs and testing features. Feel free to drop this into your repo, replacing the old README:

````markdown
# Financial Risk Toolkit

A Python package and Jupyter Lab walkthrough for computing Value-at-Risk (VaR), Expected Shortfall (ES), backtests, and Black–Scholes option pricing — now with extended robustness checks, P&L attribution, hedge effectiveness analysis, and data-integrity testing.

---

## 📦 Installation

```bash
# 1. Clone the repo
git clone https://github.com/LeoWang805/Risk_Project.git
cd Risk_Project

# 2. Create & activate a venv
python3 -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install in editable mode
pip install -e .

# 5. Run tests to verify
pytest -q
````

---

## 📁 Project Structure

```
Risk_Project/
├── data/                     # CSV price histories
├── notebooks/                # Jupyter walkthroughs
│   ├── 01_data_loader.ipynb
│   ├── 02_calibration.ipynb
│   ├── 03_var_es.ipynb
│   ├── 04_backtest.ipynb     # rolling-window backtests, extended plots & tests
│   └── 05_black_scholes.ipynb
├── src/
│   └── risk_project/         # Python package
│       ├── __init__.py
│       ├── backtest.py
│       ├── black_scholes.py
│       ├── calibration.py
│       ├── config.py
│       ├── data_loader.py
│       ├── monte_carlo.py
│       └── var_es.py
├── tests/                    # pytest suites
│   ├── test_backtest.py
│   ├── test_black_scholes.py
│   ├── test_monte_carlo.py
│   └── test_var_es.py
├── requirements.txt          # pinned dependencies
├── pyproject.toml            # build/config metadata
├── .gitignore
└── README.md                 # this file
```

---

## 🎯 Usage

### As a library

```python
from risk_project.data_loader    import load_price_series
from risk_project.calibration    import estimate_mu_sigma, estimate_covariance_matrix
from risk_project.var_es         import parametric_var_es, historical_var_es
from risk_project.monte_carlo    import monte_carlo
from risk_project.backtest       import compute_portfolio_pnl, compute_exceptions, kupiec_test
from risk_project.black_scholes  import bs_call, bs_put

# 1) Load data
series    = load_price_series(["data/AAPL-bloomberg.csv", "data/AMZN-bloomberg.csv"])
positions = {"AAPL": 100_000 / series["AAPL"].iloc[-1],
             "AMZN": 100_000 / series["AMZN"].iloc[-1]}

# 2) Calibrate
mu, sigma = {}, {}
for sym, ps in series.items():
    m, _  = estimate_mu_sigma(ps)
    mu[sym]   = m
cov        = estimate_covariance_matrix(series)

# 3) Compute Parametric VaR & ES
var, es   = parametric_var_es(positions, series, mu, cov, p=0.99)
print(f"Parametric 1-day 99% VaR: ${var:,.0f}, ES: ${es:,.0f}")

# 4) Price a vanilla call
price = bs_call(S=100, K=100, vol=0.2, r=0.01, T=1.0)
print(f"Call price: ${price:.2f}")
```

---

## 📝 Notebooks

Interactive guides under `notebooks/`:

1. **01\_data\_loader.ipynb**
   Load CSVs into Pandas, handle dates & missing data.

2. **02\_calibration.ipynb**
   Estimate drift (μ), volatility (σ), and covariance matrices.

3. **03\_var\_es.ipynb**
   Compute static (one-shot) VaR & ES via Parametric, Historical, and Monte Carlo methods.

4. **04\_backtest.ipynb**
   Rolling-window backtests with enhanced analytics and visualizations:

   * **1-Day 99% VaR backtest** with breach markers and Kupiec tests
   * **5-Day 99% VaR & 97.5% ES comparison** (Parametric vs Historical vs Monte Carlo)
   * **Bootstrap 95% CI** around Parametric 5-day VaR for robustness
   * **Monte Carlo convergence** and **monotonicity vs horizon** (optional cells)
   * **P\&L Attribution**: actual vs predicted vs residual daily P\&L
   * **Hedge Effectiveness**: rolling‐beta hedge analysis and volatility reduction
   * **Data Integrity checks**: missing dates, outlier counts, standardized-returns histograms

5. **05\_black\_scholes.ipynb**
   Black–Scholes call/put pricing and boundary‐case validations.

Start them with:

```bash
jupyter lab   # or `jupyter notebook`
```

---

## ✅ Testing & Quality

* **Unit tests** in `tests/` cover every module via pytest.

* **Run all tests**:

  ```bash
  pytest -q
  ```

* **Continuous integration** ensures code correctness on each push.

---

## 📈 Visualization & Reporting

* Publication-quality plots with Matplotlib (clear legends, gridlines, 2-year ticks).
* Drop-in code cells for each new analysis or test, making it easy to extend.
* Use this toolkit for both **research** and **production**-style risk reports.

---
