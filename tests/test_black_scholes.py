import numpy as np
import pytest

from risk_project.black_scholes import bs_call, bs_put

def test_vol0_atm_payoffs():
    # At-the-money, vol=0 ⇒ call = max(S−K,0), put = max(Ke^(−rT)−S,0)
    S, K, r, T = 100.0, 100.0, 0.05, 1.0
    call0 = bs_call(S, K, 0.0, r, T)
    put0  = bs_put (S, K, 0.0, r, T)
    assert pytest.approx(call0, abs=1e-8) == max(S - K, 0)
    expected_put = max(K * np.exp(-r*T) - S, 0)
    assert pytest.approx(put0,  abs=1e-8) == expected_put

def test_put_call_parity():
    # C − P = S − K e^(−rT)
    S, K, vol, r, T = 100.0, 95.0, 0.2, 0.02, 0.5
    C = bs_call(S, K, vol, r, T)
    P = bs_put (S, K, vol, r, T)
    lhs = C - P
    rhs = S - K * np.exp(-r*T)
    assert pytest.approx(lhs, rel=1e-6) == rhs

def test_bs_known_values():
    # Known from standard tables (e.g. Hull): S=100,K=100,r=5%,T=1,σ=20%
    S, K, vol, r, T = 100.0, 100.0, 0.20, 0.05, 1.0
    C = bs_call(S, K, vol, r, T)
    P = bs_put (S, K, vol, r, T)
    # Rough reference: C ≈ 10.4506, P ≈ 5.5735
    assert pytest.approx(C, rel=1e-3) == 10.4506
    assert pytest.approx(P, rel=1e-3) ==  5.5735
