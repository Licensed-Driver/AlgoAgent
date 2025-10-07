import pandas as pd
import numpy as np
from rl_trader.env import SingleTickerEnv

def test_env_accounting_basic():
    idx = pd.date_range("2024-01-01", periods=100, freq="T", tz="EST")
    prices = pd.Series(np.linspace(100, 110, len(idx)), index=idx, name="Close")
    feats = pd.DataFrame({"x": np.zeros(len(idx))}, index=idx)

    env = SingleTickerEnv(prices, feats, initial_equity=1000.0, spread_bps=0.0)
    obs, _ = env.reset()
    # Buy full at first step
    obs, r, d, tr, info = env.step([1.0])
    assert env.shares > 0 and env.cash >= 0
    # Hold
    for _ in range(10):
        obs, r, d, tr, info = env.step([1.0])
    assert info["equity"] > 1000.0  # price rose, equity should rise
