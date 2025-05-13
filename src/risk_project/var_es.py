# src/var_es.py

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, Tuple
from risk_project.config import P_VAR, HORIZON_DAYS, TRADING_DAYS_YR, MC_PATHS, SEED

def compute_weights(
    positions: Dict[str, float],
    price_series: Dict[str, pd.Series]
) -> Tuple[np.ndarray, float]:
    syms        = list(positions.keys())
    last_prices = np.array([price_series[s].iloc[-1] for s in syms])
    holdings    = np.array([positions[s] for s in syms])
    values      = holdings * last_prices
    V0          = values.sum()
    w           = values / V0
    return w, V0

def parametric_var_es(
    positions: Dict[str, float],
    price_series: Dict[str, pd.Series],
    mu_ann: Dict[str, float],
    cov_ann: pd.DataFrame,
    p: float = P_VAR,
    horizon_days: int = HORIZON_DAYS,
    trading_days: int = TRADING_DAYS_YR
) -> Tuple[float, float]:
    """
    Compute parametric VaR and ES for a stock-only portfolio under normal assumption.

    Parameters
    ----------
    positions : dict[str, float]
        Number of shares held for each ticker.
    price_series : dict[str, pd.Series]
        Historical price series for each ticker.
    mu_ann : dict[str, float]
        Annualized drifts for each ticker.
    cov_ann : pd.DataFrame
        Annualized covariance matrix among tickers.
    p : float, default 0.99
        Confidence level for VaR (e.g. 0.99 for 99%).
    horizon_days : int, default 1
        Holding period in days.
    trading_days : int, default 252
        Trading days per year.

    Returns
    -------
    var : float
        Dollar Value-at-Risk at level p.
    es : float
        Expected Shortfall beyond VaR.

    Raises
    ------
    ValueError
        If any input dimensions mismatch.
    """
    w, V0 = compute_weights(positions, price_series)
    syms   = list(positions.keys())

    # annual → daily → horizon scaling
    mu_daily  = np.array([mu_ann[s] / trading_days for s in syms])
    cov_daily = cov_ann.loc[syms, syms] / trading_days
    mu_h      = mu_daily * horizon_days
    cov_h     = cov_daily.values * horizon_days

    # portfolio moments
    mu_p    = w.dot(mu_h)
    sigma_p = np.sqrt(w.dot(cov_h).dot(w))

    # VaR
    z     = norm.ppf(1 - p)
    var_r = -(mu_p + z * sigma_p)
    var    = var_r * V0

    # ES
    phi    = norm.pdf(z)
    es_r   = -mu_p + sigma_p * phi / (1 - p)
    es     = es_r * V0

    return var, es

def historical_var_es(
    positions: Dict[str, float],
    price_series: Dict[str, pd.Series],
    p: float = P_VAR,
    horizon_days: int = HORIZON_DAYS,
    is_long: bool = True
) -> Tuple[float, float]:
    """
    Compute historical VaR and ES from P&L distribution.

    Parameters
    ----------
    positions : dict[str, float]
        Number of shares held for each ticker.
    price_series : dict[str, pd.Series]
        Historical price series for each ticker.
    p : float, default 0.99
        Confidence level for VaR.
    horizon_days : int, default 1
        Holding period in days.

    Returns
    -------
    var : float
        Historical VaR in dollar terms.
    es : float
        Historical ES in dollar terms.

    Raises
    ------
    ValueError
        If fewer than horizon_days+1 observations.
    """
    df        = pd.DataFrame({s: price_series[s] for s in positions})
    port_vals = df.multiply(pd.Series(positions)).sum(axis=1)
    pnl       = port_vals.diff(periods=horizon_days).dropna()

    raw = -pnl if is_long else pnl
    losses = raw.clip(lower=0.0)

    var = losses.quantile(p)
    es  = losses[losses >= var].mean()
    return var, es

def monte_carlo_var_es(
    positions: Dict[str, float],
    price_series: Dict[str, pd.Series],
    mu_ann: Dict[str, float],
    cov_ann: pd.DataFrame,
    p: float = P_VAR,
    horizon_days: int = HORIZON_DAYS,
    n_sims: int = MC_PATHS,
    trading_days: int = TRADING_DAYS_YR,
    seed: int = SEED
) -> Tuple[float, float]:
    """
    Monte Carlo simulation of VaR and ES under multivariate normal.

    Parameters
    ----------
    positions : dict[str, float]
        Number of shares held per ticker.
    price_series : dict[str, pd.Series]
        Historical price series per ticker.
    mu_ann : dict[str, float]
        Annualized drifts.
    cov_ann : pd.DataFrame
        Annualized covariance.
    p : float, default 0.99
        Confidence level.
    horizon_days : int, default 1
        Holding period in days.
    n_sims : int, default 10000
        Number of Monte Carlo trials.
    trading_days : int, default 252
        Trading days per year.
    seed : int | None
        RNG seed for reproducibility.

    Returns
    -------
    var_mc : float
        Simulated VaR.
    es_mc : float
        Simulated ES.

    Raises
    ------
    ValueError
        If covariance matrix is not positive definite.
    """
    rng       = np.random.default_rng(seed)
    w, V0     = compute_weights(positions, price_series)
    syms      = list(positions.keys())

    mu_daily  = np.array([mu_ann[s] / trading_days for s in syms])
    cov_daily = cov_ann.loc[syms, syms] / trading_days
    mu_h      = mu_daily * horizon_days
    cov_h     = cov_daily.values * horizon_days

    sims      = rng.multivariate_normal(mu_h, cov_h, size=n_sims)
    port_rets = sims.dot(w)
    losses    = -port_rets * V0

    var = np.quantile(losses, p)
    es  = losses[losses >= var].mean()
    return var, es
