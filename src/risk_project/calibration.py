# src/calibration.py
import numpy as np
import pandas as pd
from typing import Dict, Tuple

def compute_log_returns(
    price_series: pd.Series
) -> pd.Series:
    """
    Compute daily log returns of a price series.

    Parameters
    ----------
    price_series : pd.Series
        Time series of prices indexed by date.

    Returns
    -------
    pd.Series
        Daily log returns (first entry will be NaN).
    """
    return np.log(price_series / price_series.shift(1)).dropna()

def estimate_mu_sigma(
    price_series: pd.Series,
    trading_days_per_year: int = 252
) -> Tuple[float, float]:
    """
    Estimate annualized drift (mu) and volatility (sigma) from daily log returns.

    Parameters
    ----------
    price_series : pd.Series
        Time series of prices indexed by date.
    trading_days_per_year : int, default 252
        Number of trading days per year for annualization.

    Returns
    -------
    mu_ann : float
        Annualized mean return.
    sigma_ann : float
        Annualized volatility.

    Raises
    ------
    ValueError
        If the series has fewer than 2 data points.
    """
    log_rets = compute_log_returns(price_series)
    mu = log_rets.mean() * trading_days_per_year
    sigma = log_rets.std(ddof=1) * np.sqrt(trading_days_per_year)
    return mu, sigma

def estimate_covariance_matrix(
    series_dict: Dict[str, pd.Series],
    trading_days_per_year: int = 252
) -> pd.DataFrame:
    """
    Estimate annualized covariance matrix of log returns for multiple assets.

    Parameters
    ----------
    series_dict : dict[str, pd.Series]
        Mapping ticker -> price series.
    trading_days_per_year : int, default 252
        Number of trading days per year for annualization.

    Returns
    -------
    pd.DataFrame
        Annualized covariance matrix of log returns.

    Raises
    ------
    ValueError
        If any input series has fewer than 2 data points.
    """
    # build DataFrame of log returns
    rets = {
        sym: compute_log_returns(ps)
        for sym, ps in series_dict.items()
    }
    df = pd.DataFrame(rets)
    cov_daily = df.cov()
    return cov_daily * trading_days_per_year
