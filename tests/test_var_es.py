import pandas as pd
import pytest
from risk_project.var_es import historical_var_es

def make_linear_series(n=100):
    # prices go up $1/day
    dates = pd.date_range("2020-01-01", periods=n)
    return pd.Series(data=range(n), index=dates)

def test_historical_var_es_long():
    # long $1 of the asset → loss when price falls, but price always rises → no losses
    ser = make_linear_series()
    positions = {"X": 1.0}
    var, es = historical_var_es(positions, {"X": ser}, p=0.95, horizon_days=1, is_long=True)
    assert var == 0.0
    assert es == 0.0

def test_historical_var_es_short():
    # short $1 of the asset → loss when price rises → loss of $1 every day
    ser = make_linear_series()
    positions = {"X": 1.0}
    var, es = historical_var_es(positions, {"X": ser}, p=0.95, horizon_days=1, is_long=False)
    # 95th percentile of constant loss=1 is 1
    assert pytest.approx(var) == 1.0
    assert pytest.approx(es)  == 1.0
