import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from typing import Dict, Tuple

def compute_portfolio_pnl(
    series: Dict[str, pd.Series],
    positions: Dict[str, float],
    horizon_days: int = 1
) -> pd.Series:
    """
    Compute historical P&L series for a portfolio.

    Parameters
    ----------
    series : dict[str, pd.Series]
        Mapping ticker -> price series.
    positions : dict[str, float]
        Share counts per ticker.
    horizon_days : int, default 1
        Holding period to compute P&L.

    Returns
    -------
    pd.Series
        Portfolio P&L indexed by date; first horizon_days entries are NaN.
    """
    price_df = pd.DataFrame(series)
    prior    = price_df.shift(horizon_days)
    pnl_df   = (price_df - prior).multiply(pd.Series(positions), axis=1)
    # skipna=False ensures first rows with any NaN produce NaN in the sum
    pnl      = pnl_df.sum(axis=1, skipna=False)
    return pnl


def compute_exceptions(
    pnl: pd.Series,
    var_series: pd.Series
) -> pd.Series:
    """
    Flag days where loss > VaR.

    Parameters
    ----------
    pnl : pd.Series
        Portfolio P&L series.
    var : pd.Series
        VaR series aligned with pnl.

    Returns
    -------
    pd.Series
        Boolean series where True indicates an exception.
    """
    # loss = -pnl; exception if loss > VaR
    return (-pnl) > var_series


def kupiec_test(
    n_exceptions: int,
    n_obs: int,
    p: float
) -> Tuple[float, float]:
    """
    Kupiec Unconditional Coverage test (proportion-of-failures).

    Parameters
    ----------
    n_exceptions : int
        Number of VaR exceptions observed.
    n_obs : int
        Total number of observations (days).
    p : float
        VaR confidence level (e.g. 0.99 → p0=0.01 expected failure rate).

    Returns
    -------
    lr_stat : float
        Likelihood‐ratio statistic.
    p_value : float
        p‐value from chi‐square(1) test.
    """
    # perfect fit
    if n_exceptions == 0 or n_exceptions == n_obs:
        return 0.0, 1.0

    x  = n_exceptions
    n  = n_obs
    p0 = 1.0 - p   # expected exception probability

    # log‐likelihood under H1 (empirical failure rate x/n)
    l1 = x * np.log(x / n) + (n - x) * np.log((n - x) / n)
    # log‐likelihood under H0 (failure rate = p0)
    l0 = x * np.log(p0)   + (n - x) * np.log(1.0 - p0)

    lr_stat = 2.0 * (l1 - l0)
    p_value = 1.0 - chi2.cdf(lr_stat, df=1)
    return lr_stat, p_value
