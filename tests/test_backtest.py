import pandas as pd
import numpy as np
import pytest

from risk_project.backtest import compute_portfolio_pnl, compute_exceptions, kupiec_test

def test_compute_portfolio_pnl_simple():
    # Two days of prices → PnL on second day = (102*pos)−(100*pos) = +2
    dates = pd.date_range("2020-01-01", periods=2)
    price = pd.Series([100.0, 102.0], index=dates)
    series = {"A": price}
    positions = {"A": 1.0}
    pnl = compute_portfolio_pnl(series, positions, horizon_days=1)
    # First day has no prior day → NaN
    assert np.isnan(pnl.iloc[0])
    # Second day PnL = +2.0
    assert pytest.approx(pnl.iloc[1], abs=1e-8) == 2.0

def test_compute_exceptions_and_kupiec_perfect():
    # Build a PnL series where true exceptions = losses > VaR
    pnl = pd.Series([0.0, -5.0, -1.0, -10.0])
    var = pd.Series([1.0, 1.0, 1.0, 1.0])
    exc = compute_exceptions(pnl, var)
    # Only entries 1 and 3 have loss > 1 → 2 exceptions
    assert exc.sum() == 2

    # Kupiec test: 2 exceptions in 4 days at p=0.95
    lr_stat, p_val = kupiec_test(exc.sum(), len(exc), p=0.95)
    assert lr_stat >= 0
    # p_val should be between 0 and 1
    assert 0.0 <= p_val <= 1.0

def test_kupiec_no_exceptions_yields_p1():
    # If exceptions=0, p-value should be 1.0
    lr_stat, p_val = kupiec_test(0, 10, p=0.99)
    assert lr_stat >= 0
    assert pytest.approx(p_val, abs=1e-8) == 1.0
