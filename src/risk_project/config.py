# src/config.py

# ─── Data files ─────────────────────────────────────────────────────────────
# List the CSV files (relative to project root/data/) you want to load
STOCK_FILES = [
    "data/AAPL-bloomberg.csv",
    "data/AMZN-bloomberg.csv",
]

# ─── Portfolio sizing ──────────────────────────────────────────────────────
# Dollars you intend to allocate to each symbol
TARGET_NOTIONAL = 100_000

# ─── VaR/ES configuration ──────────────────────────────────────────────────
# Confidence levels
P_VAR = 0.99   # e.g. 99% VaR
P_ES  = 0.975  # e.g. 97.5% ES (the expectation beyond the 2.5% tail)

# Horizon for risk metrics (in days)
HORIZON_DAYS = 1

# ─── Rolling window lengths ────────────────────────────────────────────────
# How many past days to use for rolling calibration/backtest
WINDOW = 250

# ─── Monte Carlo settings ─────────────────────────────────────────────────
MC_PATHS = 10_000  # number of simulated paths in Monte Carlo
SEED     = 42      # RNG seed for reproducibility

# ─── Trading calendar ─────────────────────────────────────────────────────
# Trading days per year (used to annualize/inverse‐annualize)
TRADING_DAYS_YR = 252

# ─── Backtest date boundaries (optional) ──────────────────────────────────
# Set to None to use the full range of available data
BACKTEST_START = None  # e.g. "2000-01-01"
BACKTEST_END   = None  # e.g. "2024-05-01"

# ── Option‐pricing parameters ───────────────────────────────────────────────────
# Annual continuously‐compounded risk‐free rate (e.g. 2%)
RISK_FREE_RATE   = 0.02

# Time to maturity for examples (in years, e.g. 1 year)
TIME_TO_MATURITY = 1.0

