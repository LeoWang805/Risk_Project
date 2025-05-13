import numpy as np
from scipy.stats import norm

def bs_call(S: float, K: float, vol: float, r: float, T: float) -> float:
    """
    Black–Scholes price for European call.

    Parameters
    ----------
    S : float
        Current spot price.
    K : float
        Strike price.
    vol : float
        Implied volatility (annualized).
    r : float
        Risk-free rate (annualized).
    T : float
        Time to maturity (years).

    Returns
    -------
    float
        Call option price.

    Raises
    ------
    ValueError
        If inputs are non-positive or vol<0.
    """
    if np.isclose(vol, 0.0):
        return max(S - K, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S: float, K: float, vol: float, r: float, T: float) -> float:
    """
    Black–Scholes price for European put.

    Parameters
    ----------
    S : float
        Current spot price.
    K : float
        Strike price.
    vol : float
        Implied volatility.
    r : float
        Risk-free rate.
    T : float
        Time to maturity.

    Returns
    -------
    float
        Put option price.
    """
    if np.isclose(vol, 0.0):
        return max(K * np.exp(-r * T) - S, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
