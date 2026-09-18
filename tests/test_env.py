import pandas as pd
import numpy as np
import math
from rl_trader.env import SingleTickerEnv

def test_env_accounting_basic():
    idx = pd.date_range("2024-01-01", periods=100, freq="D", tz="EST")
    prices_data = np.linspace(100, 110, len(idx))
    prices = pd.DataFrame({"Close": prices_data, "Open": prices_data}, index=idx)
    feats = pd.DataFrame({"x": np.zeros(len(idx))}, index=idx)

    env = SingleTickerEnv(prices, feats, initial_equity=1000.0, spread_bps=0.0, spread_std_bps=0.0, slippage_std_bps=0.0)
    obs, _ = env.reset()
    obs, r, d, tr, info = env.step([1.0]) # Signal 100%
    obs, r, d, tr, info = env.step([1.0]) # Execute 100%
    assert env.shares > 0 and env.cash >= 0
    # Hold (keep targeting 100%)
    for _ in range(10):
        obs, r, d, tr, info = env.step([1.0])
    assert info["equity"] > 1000.0  # price rose, equity should rise
    print(f"Final equity: {info['equity']}")

def test_position_sizing():
    idx = pd.date_range("2024-01-01", periods=10, freq="D", tz="EST")
    prices_data = np.linspace(100, 100, len(idx))
    prices = pd.DataFrame({"Close": prices_data, "Open": prices_data}, index=idx)
    feats = pd.DataFrame({"x": np.zeros(len(idx))}, index=idx)

    # Test 50% max position
    env = SingleTickerEnv(prices, feats, initial_equity=10000.0, max_position_pct=0.5, spread_bps=0.0, spread_std_bps=0.0, slippage_std_bps=0.0,
                          fee_kwargs={"model": "fixed", "per_share": 0.0, "min_per_order": 0.0, "sec_fee_per_dollar": 0.0, "taf_fee_per_share": 0.0})
    obs, _ = env.reset()

    # Target 100% of the allowed size, which is 50% of the portfolio
    env.step([1.0])  # decide
    obs, r, d, tr, info = env.step([1.0])  # execute

    expected_shares = math.floor((0.5 * 10000.0) / 100.0)
    assert env.shares == expected_shares, f"Expected {expected_shares} shares, got {env.shares}"

    # Check cash is roughly half
    assert 4900 < env.cash < 5100
