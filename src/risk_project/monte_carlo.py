# src/monte_carlo.py

import numpy as np
import pandas as pd
from typing import Dict
from risk_project.calibration import estimate_mu_sigma, estimate_covariance_matrix

def monte_carlo(
    price_df: pd.DataFrame,
    positions: Dict[str, float],
    idx: int,
    is_long: bool,
    is_var: bool,
    p: float,
    horizon_days: int,
    window: int,
    trading_days: int,
    n_sims: int,
    seed: int = None
) -> float:
    """
    Unified rolling Monte Carlo VaR or ES estimator.

    Parameters
    ----------
    price_df : pd.DataFrame
        DataFrame of prices (columns = tickers).
    positions : dict[str, float]
        Share counts per ticker.
    idx : int
        Current index at which to compute risk (end of window).
    is_long : bool
        True for long portfolio, False for short.
    is_var : bool
        True to return VaR, False to return ES.
    p : float
        Confidence level.
    horizon_days : int
        Holding period.
    window : int
        Rolling window length.
    trading_days : int
        Trading days per year.
    n_sims : int
        Number of Monte Carlo trials.
    seed : int
        RNG seed.

    Returns
    -------
    float
        VaR or ES at date index `idx`.

    Raises
    ------
    ValueError
        If idx < window or DataFrame too small.
    """
    if idx < window:
        raise IndexError(f"idx {idx} < window {window}")
    # 1) historical slice
    hist = price_df.iloc[idx-window : idx]

    # 2) annual μ & Σ on that slice
    mu_ann = {
        s: estimate_mu_sigma(hist[s], trading_days_per_year=trading_days)[0]
        for s in hist.columns
    }
    cov_ann = estimate_covariance_matrix(
        {s: hist[s] for s in hist.columns},
        trading_days_per_year=trading_days
    )

    # 3) convert to daily & horizon
    syms     = list(positions.keys())
    mu_daily = np.array([mu_ann[s]/trading_days for s in syms])
    cov_daily= cov_ann.loc[syms,syms]/trading_days
    mu_h     = mu_daily * horizon_days
    cov_h    = cov_daily.values * horizon_days

    # 4) simulate log-returns
    rng  = np.random.default_rng(seed)
    sims = rng.multivariate_normal(mu_h, cov_h, size=n_sims)  # shape (n_sims, n_assets)

    # 5) portfolio weighting & V0 at date idx
    last_prices = np.array([price_df[s].iloc[idx] for s in syms])
    holdings    = np.array([positions[s] for s in syms])
    values      = holdings * last_prices
    V0          = values.sum()
    w           = values / V0

    # 6) portfolio log-return sims → discrete returns → P&L
    port_log_rets = sims.dot(w)                # each sim’s log-return
    port_discrete = np.expm1(port_log_rets)    # exp(log) - 1
    pnl_sims      = port_discrete * V0

    # 7) define losses & clip negatives for longs
    raw_losses = -pnl_sims if is_long else pnl_sims
    losses     = np.clip(raw_losses, a_min=0.0, a_max=None)

    # 8) VaR or ES
    var = np.quantile(losses, p)
    es  = losses[losses >= var].mean()

    return var if is_var else es
