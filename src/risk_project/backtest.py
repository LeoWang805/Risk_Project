import numpy as np
import pandas as pd
from scipy.stats import chi2
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
    Perform Kupiec proportion-of-failures test.

    Parameters
    ----------
    k : int
        Number of exceptions observed.
    n : int
        Total number of trials/days.
    p : float, default 0.99
        Target non-exception probability.

    Returns
    -------
    lr_stat : float
        Kupiec likelihood-ratio statistic.
    p_value : float
        p-value for the test.

    Raises
    ------
    ValueError
        If n <= 0 or k not in [0, n].
    """
    # edge‐cases: no exceptions or all exceptions → perfect fit
    if n_exceptions == 0 or n_exceptions == n_obs:
        return 0.0, 1.0

    pi = n_exceptions / n_obs
    # log‐likelihood ratio
    L0 = (1 - p)**(n_obs - n_exceptions) * p**n_exceptions
    L1 = (1 - pi)**(n_obs - n_exceptions) * pi**n_exceptions
    lr = -2 * np.log(L0 / L1)
    p_value = 1 - chi2.cdf(lr, df=1)
    return lr, p_value
