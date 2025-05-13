import pandas as pd
import numpy as np
import pytest
from risk_project.monte_carlo import monte_carlo

def make_series(n=300):
    dates   = pd.date_range("2020-01-01", periods=n)
    rets    = np.full(n, 0.01)
    prices  = 100 * np.cumprod(1 + rets)
    return pd.Series(prices, index=dates)

def test_monte_carlo_var_long_zero_loss():
    df  = pd.DataFrame({"X": make_series()})
    pos = {"X": 1.0}
    v = monte_carlo(
        price_df     = df,
        positions    = pos,
        idx          = 250,
        is_long      = True,
        is_var       = True,
        p            = 0.99,
        horizon_days = 1,
        window       = 250,
        trading_days = 252,
        n_sims       = 500,
        seed         = 0
    )
    # A strictly upward series never produces a loss on a long:
    assert pytest.approx(v, abs=1e-8) == 0.0

def test_monte_carlo_es_short_constant_loss():
    df  = pd.DataFrame({"X": make_series()})
    pos = {"X": 1.0}
    idx = 250
    es = monte_carlo(
        price_df     = df,
        positions    = pos,
        idx          = idx,
        is_long      = False,
        is_var       = False,
        p            = 0.99,
        horizon_days = 1,
        window       = 250,
        trading_days = 252,
        n_sims       = 500,
        seed         = 0
    )
    # For a short, P&L = -ΔP = price_today * 1%, so ES ≈ price[idx] * 0.01
    expected = df["X"].iloc[idx] * 0.01
    # allow ~5% relative MC noise
    assert pytest.approx(es, rel=0.05) == expected
