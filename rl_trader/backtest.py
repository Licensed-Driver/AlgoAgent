import numpy as np
import pandas as pd
from .env import SingleTickerEnv

def run_backtest(model, prices: pd.Series, features: pd.DataFrame, initial_equity=10_000.0,
                 spread_bps=2.0, slippage_bps=0.0, max_position_pct=1.0, reward_mode="pnl"):
    env = SingleTickerEnv(prices, features, initial_equity, spread_bps, slippage_bps, max_position_pct, reward_mode)
    obs, _ = env.reset()
    done = False
    eq_hist = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        eq_hist.append(info.get("equity", np.nan))
    eq = pd.Series(eq_hist, index=features.index[1:len(eq_hist)+1])
    return eq
